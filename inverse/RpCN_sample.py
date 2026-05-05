import spt_guidance.torch.rpcn as rpcn
import torch 
from inverse_utils import inverse_utils
from utils import load_model_and_sampler
# reward_fn = operator.log_likelihood
from utils import normalize_tensor
from utils import plot_samples
import argparse
import os 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_id', type=int, default=60040, help="Which validation image do you want to sample?")
    parser.add_argument('--seed', type=int, default=31, help="Which validation image do you want to sample?")
    parser.add_argument('--operator', type=str, default="inpainting_random", help="What forward operator are you using?")
    
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
    
    ### kwargs
    batch_size = 1
    n_chains = 3 
    resampling_frequency = 10 
    guidance_strength = 1.25

    guidance_schedule = "late"


    path_to_validation = '~/.cache/kagglehub/datasets/weideng056/ffhq-1k-valid/versions/1/samples'

    def seed_function(seed, id):
        random = seed + 2435* id
        return random 

    torch.manual_seed(seed_function(args.seed, args.img_id))

    model, ddpm = load_model_and_sampler()
    
    inverse_class = inverse_utils(model, ddpm, path_to_validation, batch_size, resampling_frequency, guidance_strength, guidance_schedule, True, args.img_id, args.operator, "cuda")


    kernel = rpcn.KernelPreconditionedCrankNicolson(
        inverse_class.data_dim,  # shape of one data sample, not a batch
        device = inverse_class.device,
        dtype = inverse_class.dtype
    )


    rpcn = rpcn.ResidualMCMCSteering(
        kernel,
        torch.linspace(torch.pi / 2 - 1e-3, 0.05, n_chains, device=inverse_class.device).sin(),  # pCN kernel step size
        reward_fn=inverse_class.reward_fn,  # function be be maximized (reward function)
        expected_next_fn=inverse_class.expected_next_fn,  # function that computes the next state's mean given the current state
        std_fn=inverse_class.std_fn,  # function that returns the standard deviation of each dimension of the current or next state (see docstring)
        intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
        inverse_temperature=torch.linspace(0., 1., n_chains, device=inverse_class.device),  # inverse temperatures
        kwargs_intermediate_reward=dict(
            expected_sample_fn=inverse_class.get_tweedie,  # function to return the expected final samples (at time=1) of the current particles
        ),
    )

    def pCN_update_fn(timestep, t0, t1, reward):
        return timestep % inverse_class.check_every == 0

    with torch.no_grad():
        print(f'Generating ...')
        final_samples = rpcn.run(
            # key,  # PRNG key if using Jax
            inverse_class.timesteps.to(inverse_class.device),  # sampling time interval discretization
            pcn_should_update_fn=pCN_update_fn,  # function to indicate whether the pCN updates should run
            # pcn_should_replica_exchange_fn=lambda t0, t1, reward: t0 % image_class.check_every == 0,  # function to indicate whether to do replica exchange
            pcn_update_count=50,  # number of pcn steps to take per update
            batch_size=inverse_class.batch_size,  # number of target samples to generate at once
        )

        if args.save:

            save_dir = "generated/tensors"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{args.img_id}_{args.operator}_{seed_function(args.seed, args.img_id)}.pt"
            )

            torch.save(final_samples[-1], save_path)

        if args.plot:
            plot_dir = "generated/figures"
            os.makedirs(plot_dir, exist_ok=True)

            img = normalize_tensor(final_samples[-1])
            plot_samples(img, args.img_id, args.operator)