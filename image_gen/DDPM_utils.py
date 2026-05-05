import torch 
from diffusers import StableDiffusionPipeline, DDPMScheduler, StableDiffusionXLPipeline
from PIL import Image
from IR_reward import rm_load
import numpy as np 
import json 

def tensors_to_pil(decoded_tensors, reorder=False):
    """
    decoded_tensors: Tensor [B, 3, H, W], range [0, 1]
    returns: List[PIL.Image]
    """
    if isinstance(decoded_tensors, np.ndarray):
        decoded_tensors = torch.from_numpy(decoded_tensors).float()
    
    else:
        decoded_tensors = decoded_tensors.float()

    if reorder == True: 
        images = decoded_tensors.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    
    else: 
        images = decoded_tensors.detach().cpu().float().numpy()

    # Scale to 8-bit color range
    images = (images * 255).round().astype("uint8")
    
    # Create the list of PIL images
    return [Image.fromarray(img) for img in images]



def get_prompt_by_id(json_path, img_id):    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if entry.get('id') == img_id:
            return entry.get('prompt')
            
    return None  # Return None if the ID isn't found



class DDPM_helpers():
    def __init__(
        self,
        model="v1.5",
        device="cuda",
        num_inference_steps=100,
        guidance_scale=7.5,
        height=512,
        width=512,
        dtype=torch.float16,
        check_every=10,
        prompt ="a photo of a cat",
        batch_size = 4
    ):
        self.device = device
        self.guidance_scale = guidance_scale

        self.height = height
        self.width = width
        self.check_every = check_every
        self.dtype=torch.float16
        self.prompt = prompt

        self.ir_model = rm_load(device=device)
        
        # models to load 
        self.pipe, self.scheduler = self.load_pipeline(model)
        self.scheduler.set_timesteps(num_inference_steps)
        self.data_dim =  (self.pipe.unet.config.in_channels, self.height //8, self.width //8)
        self.batch_size = batch_size
        self.prompt_embeds, self.negative_prompt_embeds = self.load_prompts(self.prompt)
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device, dtype=torch.float32)

    def reward_fn(self, images):
        prompt = self.prompt 
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float().to(self.device)
            images = images.permute(0, 3, 1, 2)
            
        rewards = self.ir_model.score_batched([prompt]*self.batch_size, [images[i].to(dtype=torch.float32) for i in range(images.shape[0])])
        return rewards

    def load_pipeline(self, model):
        model = model.lower()

        if model == "v1.4":
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=self.dtype,
            ).to(self.device)
        elif model == "v1.5":
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=self.dtype,
            ).to(self.device)
        elif model == "v2.1":
            pipe = StableDiffusionPipeline.from_pretrained(
                "sd2-community/stable-diffusion-2-1",
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        return pipe, scheduler
    

    def process_latents(self, latents, t):
        scaled_inputs = self.scheduler.scale_model_input(latents, t) # fk steering does this
        # scaled_inputs = scaled_model_input.flatten(0, 1) # need to flatten across chains 
        latent_model_input = torch.cat([scaled_inputs] * 2) # duplicate for CFG (fk steering also does this )
        return latent_model_input.to(dtype=self.dtype)

    def prepare_prompt_for_cfg(self):
        # you need to process prompts 
        # uncond_rep = self.uncond_embeds.repeat_interleave(self.n_chains, dim=0)
        # text_rep = self.text_embeds.repeat_interleave(self. n_chains, dim=0)
        encoder_hidden_states = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)
        return encoder_hidden_states
    

    @torch.no_grad()
    def predict_cfg_noise(self, latents, t, resize=True):

        latent_model_input = self.process_latents(latents, t)
        encoder_hidden_states = self.prepare_prompt_for_cfg()


        noise_pred = self.pipe.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states = encoder_hidden_states,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False
            )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_pred
        

    def get_coeffecients_by_t(self, t, t_prev):
        
        
        alpha_bar_t = self.alphas_cumprod[t.long()]
        # If t_prev is negative (last step), alpha_bar_prev is 1.0
        alpha_bar_t_prev = self.alphas_cumprod[t_prev.long()] if t_prev >= 0 else torch.tensor(1.0).to(self.device)
        
        # alpha_t is the ratio of the cumulative products
        alpha_t = alpha_bar_t / alpha_bar_t_prev
        beta_t = 1 - alpha_t
        
        return alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t
    

    def decode_latents(self, latents):
        latents = latents / self.pipe.vae.config.scaling_factor
        images = self.pipe.vae.decode(latents, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def std_fn(self, t, t1=None):
        if t1 is None:
            return t * 0 + 1 
        if t1 < 0:
            return torch.tensor(0.0)
        else: 
            alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, t1)
            sigma_t = (((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t).sqrt()
            return sigma_t
        
    @torch.no_grad()
    def expected_next_fn(self, t, t_prev, latents):
        
        noise_pred = self.predict_cfg_noise(latents, t)
        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, t_prev)
        mu_tm1 = (1 / alpha_t.sqrt()) * (latents - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred)
        return mu_tm1
    
    @torch.no_grad()
    def get_tweedie(self,t, latents):
        alpha_bar_t, alpha_bar_t_prev, alpha_t, beta_t = self.get_coeffecients_by_t(t, self.scheduler.timesteps[-1])
        noise_pred = self.predict_cfg_noise(latents, t)
        latent_x0 =  (1/alpha_bar_t.sqrt()) * (latents - (1  - alpha_bar_t).sqrt() * noise_pred)
        # breakpoint()
        return self.decode_latents(latent_x0)

    def load_prompts(self, prompt, negative_prompt=None):
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=self.batch_size,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt

            )

        return prompt_embeds, negative_prompt_embeds





    



    