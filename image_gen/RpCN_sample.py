import spt_guidance.torch.rpcn as rpcn
from DDIM_utils import DDIM_helpers, tensors_to_pil, get_prompt_by_id 
from DDPM_utils import DDPM_helpers, tensors_to_pil, get_prompt_by_id 

import torch 
import json 
import argparse
import csv
import os

def save_rewards_to_csv(csv_path, rewards):
    """
    Saves batch indices and rewards to a specified CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Move to CPU and convert to list if it's a torch tensor
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().numpy()

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_index", "reward"])
        
        for idx, reward in enumerate(rewards):
            writer.writerow([idx, reward[0]])

def compute_seed(id, seed=256):
    partial_id = int(id[:6])
    return seed + partial_id


if __name__ == "__main__":
    n_chains = 4
    batch_size = 4

    parser = argparse.ArgumentParser(description="Retrieve a prompt from a JSON file by ID.")
    parser.add_argument("--img_id", default = '004971-0071', type=str, help="The prompt ID of the image to create (e.g., 004971-0071, 000562-0011)")
    parser.add_argument("--model", default='v1.5', type=str, help="The stable diffusion model you want to use (e.g., v1.4, v1.5, vXL)")
    parser.add_argument("--resample_freq", default = 20, type=int, help="How often do you want to resample?")
    parser.add_argument("--pCN_ct", default = 10, type=int, help="How many pCN steps do you want to take?")
    args = parser.parse_args()

    prompt = get_prompt_by_id(json_path = "prompts/prompts.json", img_id = args.img_id)

    torch.manual_seed(compute_seed(args.img_id, 31))
    
    if args.model == 'v1.4':
        image_class = DDPM_helpers(prompt=prompt, batch_size=batch_size, num_inference_steps=100, check_every=args.resample_freq, model=args.model) 

    else: 
        image_class = DDIM_helpers(prompt=prompt, batch_size=batch_size, num_inference_steps=100, check_every=args.resample_freq, model=args.model) 

    kernel = rpcn.KernelPreconditionedCrankNicolson(
        image_class.data_dim,  # shape of one data sample, not a batch
        device = image_class.device,
        dtype = image_class.dtype
    )

    rpcn = rpcn.ResidualMCMCSteering(
        kernel,
        torch.linspace(torch.pi / 2 - 1e-3, 0.05, n_chains, device=image_class.device).sin(),  # pCN kernel step size
        reward_fn=image_class.reward_fn,  # function be be maximized (reward function)
        expected_next_fn=image_class.expected_next_fn,  # function that computes the next state's mean given the current state
        std_fn=image_class.std_fn,  # function that returns the standard deviation of each dimension of the current or next state (see docstring)
        intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
        inverse_temperature=torch.linspace(0., 10., n_chains, device=image_class.device),  # inverse temperatures
        kwargs_intermediate_reward=dict(
            expected_sample_fn=image_class.get_tweedie,  # function to return the expected final samples (at time=1) of the current particles
        ),
    )

    def pCN_update_fn(timestep, t0, t1, reward):
        # print(f"pCN update at t={t0}, reward={reward}")
        return timestep % image_class.check_every == 0


    with torch.no_grad():
        print(f'Generating reward aligned samples for prompt: {args.img_id}')
        final_samples = rpcn.run(
            # key,  # PRNG key if using Jax
            image_class.scheduler.timesteps.to(image_class.device),  # sampling time interval discretization
            pcn_should_update_fn=pCN_update_fn,  # function to indicate whether the pCN updates should run
            # pcn_should_replica_exchange_fn=lambda t0, t1, reward: t0 % image_class.check_every == 0,  # function to indicate whether to do replica exchange
            pcn_update_count=args.pCN_ct,  # number of pcn steps to take per update
            batch_size=image_class.batch_size,  # number of target samples to generate at once
        )


        img_array = image_class.pipe.decode_latents(final_samples[-1].detach())
        decoded_images = tensors_to_pil(img_array)
        rewards = image_class.reward_fn(img_array)


        img_dir = f"generated/{args.model}/images"
        csv_dir = f"generated/{args.model}/csv"

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        csv_name = f"{csv_dir}/{args.img_id}_rewards.csv"
        print(f'Saving IR results to: {csv_dir}')
        save_rewards_to_csv(csv_name, rewards)

        # save coldest images     
        print(f'Saving images to: {img_dir}')
        for i, img in enumerate(decoded_images):
            img.save(f"{img_dir}/{args.img_id}_{i}.png")

