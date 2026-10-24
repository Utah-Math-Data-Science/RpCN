import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from PIL import Image

from IR_reward import rm_load


TTG_SRC = Path(__file__).resolve().parents[1] / "RpCN_updated" / "RpCN" / "src"
sys.path.insert(0, str(TTG_SRC))

from ttg import Gaussian, make_evaluate_torch, on_x0  # noqa: E402
from ttg.backend import TorchBackend  # noqa: E402


BATCH_SIZE = 4
NUM_INFERENCE_STEPS = 100

MODEL_IDS = {
    "v1.4": "CompVis/stable-diffusion-v1-4",
    "v1.5": "runwayml/stable-diffusion-v1-5",
    "v2.1": "sd2-community/stable-diffusion-2-1",
}


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--img_id",
        default="004971-0071",
        type=str,
        help="Prompt ID of the image to create (e.g. 004971-0071)",
    )
    parser.add_argument(
        "--model",
        default="v1.5",
        type=str,
        help="Stable Diffusion model (v1.4, v1.5, or v2.1)",
    )


def tensors_to_pil(decoded_tensors, reorder=False):
    """Convert decoded image tensors in [0, 1] to PIL images."""
    images = torch.as_tensor(decoded_tensors).detach().float().cpu()
    if reorder:
        images = images.permute(0, 2, 3, 1)
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(image) for image in images]


def get_prompt_by_id(json_path, img_id):
    with open(json_path) as f:
        prompts = json.load(f)
    return next(
        (entry.get("prompt") for entry in prompts if entry.get("id") == img_id),
        None,
    )


class StableDiffusionContext:
    """Models and metadata shared by the ttg samplers."""

    def __init__(
        self,
        prompt,
        model="v1.5",
        device="cuda",
        dtype=torch.float16,
        batch_size=BATCH_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=7.5,
        check_every=10,
    ):
        try:
            model_id = MODEL_IDS[model.lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported model: {model}") from error

        self.device = device
        self.dtype = dtype
        self.prompt = prompt
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.check_every = check_every

        self.ir_model = rm_load(device=device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=device,
            dtype=torch.float32,
        )

        sample_size = self.pipe.unet.config.sample_size
        spatial_shape = (
            (sample_size, sample_size)
            if isinstance(sample_size, int)
            else tuple(sample_size)
        )
        self.data_dim = (self.pipe.unet.config.in_channels, *spatial_shape)
        self.prompt_embeds, self.negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )

    def decode_latents(self, latents):
        latents = latents / self.pipe.vae.config.scaling_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        return (images / 2 + 0.5).clamp(0, 1)


class DiffusersSchedule:
    """Expose a Diffusers inference grid through the ttg Schedule API."""

    def __init__(self, scheduler, device, dtype):
        model_timesteps = scheduler.timesteps.flip(0)
        self.model_timesteps = model_timesteps.to(device=device)
        self.alphas = scheduler.alphas_cumprod[model_timesteps.long()].to(
            device=device,
            dtype=dtype,
        )
        self.n_steps = len(self.alphas)

    def time(self, u):
        return torch.round(u * (self.n_steps - 1)).clamp(0, self.n_steps - 1)

    def _alpha(self, t):
        return self.alphas[t.long()]

    def model_timestep(self, t):
        return self.model_timesteps[t.long()]

    def mean(self, t):
        return self._alpha(t).sqrt()

    def std(self, t):
        return (1.0 - self._alpha(t)).sqrt()

    def log_snr(self, t):
        alpha = self._alpha(t)
        return torch.log(alpha) - torch.log1p(-alpha)

    def log_snr_diff(self, t_a, t_b):
        return self.log_snr(t_a) - self.log_snr(t_b)

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        mean = torch.zeros((*leading_shape, *event_shape), dtype=dtype, device=device)
        var = torch.ones(leading_shape, dtype=dtype, device=device)
        return Gaussian(mean, var)


def build_evaluate(image_class, schedule):
    def eps_network(latents, t):
        model_t = schedule.model_timestep(t)
        scaled = image_class.scheduler.scale_model_input(latents, model_t)
        latent_input = torch.cat([scaled, scaled], dim=0).to(image_class.dtype)

        batch_size = latents.shape[0]
        negative = image_class.negative_prompt_embeds[:1].expand(batch_size, -1, -1)
        prompt = image_class.prompt_embeds[:1].expand(batch_size, -1, -1)
        encoder_hidden_states = torch.cat([negative, prompt], dim=0)

        noise_uncond, noise_text = image_class.pipe.unet(
            latent_input,
            torch.cat([model_t, model_t]),
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0].chunk(2)
        return noise_uncond + image_class.guidance_scale * (
            noise_text - noise_uncond
        )

    def to_x0_eps(x, t, model_output, sch):
        mean = sch.mean(t).reshape((-1,) + (1,) * (x.ndim - 1))
        std = sch.std(t).reshape((-1,) + (1,) * (x.ndim - 1))
        if image_class.scheduler.config.prediction_type == "epsilon":
            return (x - std * model_output) / mean, model_output
        if image_class.scheduler.config.prediction_type == "v_prediction":
            return mean * x - std * model_output, mean * model_output + std * x
        raise ValueError(
            f"Unsupported prediction type: {image_class.scheduler.config.prediction_type}"
        )

    def reward(latent_x0):
        images = image_class.decode_latents(latent_x0.to(dtype=image_class.dtype))
        prompts = [image_class.prompt] * images.shape[0]
        image_list = [image.float() for image in images]
        return image_class.ir_model.score_batched(prompts, image_list).reshape(-1)

    return make_evaluate_torch(eps_network, to_x0_eps, schedule, on_x0(reward))


def build_sampling_components(image_class):
    backend = TorchBackend(device=image_class.device)
    schedule = DiffusersSchedule(
        image_class.scheduler,
        image_class.device,
        image_class.dtype,
    )
    return backend, schedule, build_evaluate(image_class, schedule)


def compute_seed(img_id, seed=256):
    return seed + int(img_id[:6])


def decode_and_score(image_class, prompt, latents):
    img_array = image_class.decode_latents(latents.to(dtype=image_class.dtype))
    decoded_images = tensors_to_pil(img_array, reorder=True)
    rewards = image_class.ir_model.score_batched(
        [prompt] * len(decoded_images), [image.float() for image in img_array]
    )
    return decoded_images, rewards


def save_rewards_to_csv(csv_path, rewards):
    """Save one reward for each generated image."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().reshape(-1).tolist()

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_index", "reward"])
        for idx, reward in enumerate(rewards):
            writer.writerow([idx, reward])


def save_results(model, stem, decoded_images, rewards, label):
    img_dir = f"generated/{model}/{label}/images"
    csv_dir = f"generated/{model}/{label}/csv"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    print(f"Saving {label} results to: {csv_dir}")
    save_rewards_to_csv(f"{csv_dir}/{stem}_rewards.csv", rewards)

    print(f"Saving images to: {img_dir}")
    for index, image in enumerate(decoded_images):
        image.save(f"{img_dir}/{stem}_{index}.png")
