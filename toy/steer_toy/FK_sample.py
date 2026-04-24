import spt_guidance.torch.fk_steering as fk_steering
import torch 
from log_reward import LogReward
from model.model import DiffusionModel
from analytic_helpers import analytic_helpers
from utils import plot_samples
from functools import partial

if __name__ == "__main__":
    device = "cuda"
    centers = torch.tensor([
        ( 8.0000,  0.0000),
        ( 5.6569,  5.6569),
        # ( 0.0000,  8.0000),
        # (-5.6569,  5.6569),
        (-8.0000,  0.0000),
        # (-5.6569, -5.6569),
        # ( 0.0000, -8.0000),
        ( 5.6569, -5.6569)
        ])

    reward_fn = LogReward(centers, 0.25)



    checkpoint = torch.load("model/diffusion_gmm_model.pt")
    model_state_dict = checkpoint['model_state_dict']
    alphas_cumprod = checkpoint['alphas_cumprod']
    betas = checkpoint['betas']
    alphas = 1.0 - betas

    # Reconstruct the model architecture
    model = DiffusionModel(input_dim=2, output_dim=2, hidden_dim=512, n_hidden=3).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    toy_class  = analytic_helpers(alphas_cumprod, model, batch_size=1024)


    fk = fk_steering.FeynmannKac(
        init_param_fn=partial(torch.randn, toy_class.data_dim, device=toy_class.device),  # data_dims is shape of one data sample, not a batch
        reward_fn=reward_fn.forward,  # function be be maximized (reward function)
        propose_fn=toy_class.get_tweedie,  # function that updates the current particles to the next time step
        resample_fn=fk_steering.ResampleFn.adaptive_resample,  # function to resample the particles based on their potential
        potential=fk_steering.Potential.MAX,  # how the potential of each particle is computed
        intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
        inverse_temperature=5.,  # the tilt of the target distribution
        kwargs_intermediate_reward=dict(
            expected_sample_fn=lambda t0, z: toy_class.get_tweedie(t0, 1., z),  # function to return the expected final samples (at time=1) of the current particles
        ),
    )
    final_samples = fk.run(
    # key,  # PRNG key if using Jax
    toy_class.timesteps,  # sampling time interval discretization
    4,  # number of FK Steering chains to use
    batch_size=toy_class.batch_size,  # number of target samples to generate at once
    )

    print(f'final samples size:{final_samples.size()}')