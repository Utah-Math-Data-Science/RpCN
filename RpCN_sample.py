import spt_guidance.torch.rpcn as rpcn
import torch 
from inverse_utils import inverse_utils
from utils import load_model_and_sampler
# reward_fn = operator.log_likelihood
from utils import normalize_tensor
from utils import plot_samples
import argparse
import os 
import time
import numpy as np



if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_id', type=int, default=49000, help="Which validation image do you want to sample?")
    parser.add_argument('--seed', type=int, default=31, help="Which validation image do you want to sample?")
    parser.add_argument('--operator', type=str, default="motion_blur", help="What forward operator are you using?")
    parser.add_argument(
        "--n_chains",
        type=int,
        default=5,
        help="Number of parallel-tempering chains.",
    )
    parser.add_argument(
        "--chain_chunk_size",
        type=int,
        default=1,
        help="Maximum chains evaluated together; lower values use less GPU memory.",
    )
    parser.add_argument(
        "--resampling_frequency",
        type=int,
        default=15,
        help="Run pCN updates every N diffusion steps.",
    )
    parser.add_argument(
        "--pcn_update_count",
        type=int,
        default=10,
        help="Number of pCN updates at each resampling step.",
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=0.03,
        help="Multiplier for posterior-scaled gradient guidance.",
    )
    parser.add_argument(
        '--grad_clip_norm',
        type=float,
        default=2.5,
        help="Maximum per-sample RMS norm for the reward gradient.",
    )
    parser.add_argument(
        '--guidance_residual_floor',
        type=float,
        default=None,
        help="Stop likelihood guidance after measurement MSE reaches this value.",
    )
    parser.add_argument(
        "--guidance_scaling",
        choices=("direct", "std", "variance"),
        default="std",
        help="Diffusion noise scale applied to gradient guidance.",
    )
    parser.add_argument(
        "--guidance_schedule",
        choices=(
            "none",
            "constant",
            "early",
            "middle",
            "late",
            "late_quadratic",
            "late_cubic",
            "late_quartic",
            "late_sextic",
            "late_octic",
            "last_half",
        ),
        default="late",
        help="Timestep schedule applied to gradient guidance.",
    )

    parser.add_argument(
        "--output_dir",
        default="generated/tensors",
        help="Directory used by --save.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="If provided, plot the final sample."
    )    
    parser.add_argument(
        "--save",
        action="store_true",
        help="If provided, save the unprocessed final sample."
    )    
    args = parser.parse_args()
    if args.n_chains < 1:
        parser.error("--n_chains must be at least 1")
    if args.chain_chunk_size < 1:
        parser.error("--chain_chunk_size must be at least 1")
    if args.resampling_frequency < 1:
        parser.error("--resampling_frequency must be at least 1")
    if args.pcn_update_count < 1:
        parser.error("--pcn_update_count must be at least 1")
    if args.guidance_residual_floor is not None and args.guidance_residual_floor < 0:
        parser.error("--guidance_residual_floor must be nonnegative")
    
    ### kwargs
    batch_size = 1

    path_to_validation = '~/DAPS/dataset/test-ffhq'

    def seed_function(seed, id):
        random = seed + 2435* id
        return random 

    effective_seed = seed_function(args.seed, args.img_id)
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed % (2 ** 32))

    model, ddpm = load_model_and_sampler()
    
    inverse_class = inverse_utils(
        model,
        ddpm,
        path_to_validation,
        batch_size,
        args.resampling_frequency,
        args.guidance_schedule,
        guidance_scale=args.guidance_scale,
        guidance_scaling=args.guidance_scaling,
        grad_clip_norm=args.grad_clip_norm,
        guidance_residual_floor=args.guidance_residual_floor,
        grad_norm=True,
        image_id=args.img_id,
        operator=args.operator,
        device="cuda",
    )


    kernel = rpcn.KernelPreconditionedCrankNicolson(
        inverse_class.data_dim,  # shape of one data sample, not a batch
        device = inverse_class.device,
        dtype = inverse_class.dtype
    )


    rpcn = rpcn.ResidualMCMCSteering(
        kernel,
        torch.linspace(torch.pi / 2 - 1e-3, 0.05, args.n_chains, device=inverse_class.device).sin(),  # pCN kernel step size
        reward_fn=inverse_class.reward_fn,  # function be be maximized (reward function)
        expected_next_fn=inverse_class.expected_next_fn,  # function that computes the next state's mean given the current state
        std_fn=inverse_class.std_fn,  # function that returns the standard deviation of each dimension of the current or next state (see docstring)
        intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
        inverse_temperature=torch.linspace(0., 1., args.n_chains, device=inverse_class.device),  # inverse temperatures
        chain_chunk_size=args.chain_chunk_size,
        kwargs_intermediate_reward=dict(
            expected_sample_fn=inverse_class.get_tweedie,  # function to return the expected final samples (at time=1) of the current particles
        ),
    )

    def pCN_update_fn(timestep, t0, t1, reward):
        if timestep % inverse_class.check_every != 0:
            return False
        if args.guidance_residual_floor is None:
            return True
        cold_reward = reward[-1].mean()
        measurement_size = inverse_class.y[0].numel()
        residual = -2 * inverse_class.operator.sigma ** 2 * cold_reward / measurement_size
        return residual.item() > args.guidance_residual_floor

    with torch.no_grad():

        print(f'Generating Reconstructed sample for FFHQ Validation Image: {args.img_id} under Operator: {args.operator}')
        final_samples = rpcn.run(
            # key,  # PRNG key if using Jax
            inverse_class.timesteps.to(inverse_class.device),  # sampling time interval discretization
            pcn_should_update_fn=pCN_update_fn,  # function to indicate whether the pCN updates should run
            # pcn_should_replica_exchange_fn=lambda t0, t1, reward: t0 % image_class.check_every == 0,  # function to indicate whether to do replica exchange
            pcn_update_count=args.pcn_update_count,  # number of pcn steps to take per update
            batch_size=inverse_class.batch_size,  # number of target samples to generate at once
        )
        reference = (inverse_class.ref_image * 0.5 + 0.5).clamp(0, 1)
        coldest_chain_index = torch.argmax(rpcn.inverse_temperature).item()
        final_state = final_samples[coldest_chain_index]
        terminal_t = torch.zeros(
            final_state.shape[0], device=final_state.device, dtype=torch.long
        )
        final_state = ddpm.p_mean_variance(model, final_state, terminal_t)["mean"]
        reconstruction = (final_state * 0.5 + 0.5).clamp(0, 1)
        mse = torch.mean((reconstruction - reference) ** 2)
        psnr = -10 * torch.log10(mse)
        measurement_mse = torch.mean(
            (inverse_class.operator(final_state) - inverse_class.y) ** 2
        )
        print(f'PSNR: {psnr.item():.4f} dB')
        print(f'Measurement MSE: {measurement_mse.item():.8f}')

        end= time.time()
        print(f'total time: {end - start}')
        if args.save:

            save_dir = args.output_dir
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{args.img_id}_{args.operator}_{effective_seed}.pt"
            )

            torch.save(final_state, save_path)

        if args.plot:
            plot_dir = "generated/figures"
            os.makedirs(plot_dir, exist_ok=True)

            img = normalize_tensor(final_state)
            plot_samples(img, args.img_id, args.operator)
