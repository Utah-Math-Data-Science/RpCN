from functools import partial

import spt_guidance.torch.fk_steering as fk_steering
import torch

from toy.steer_toy.analytic_helpers import analytic_helpers
from toy.steer_toy.log_reward import LogReward
from toy.steer_toy.model.model import DiffusionModel
from toy.steer_toy.utils import DIR_ROOT, get_reward_gmm_and_subset


if __name__ == "__main__":
    torch.manual_seed(31)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gmm, subset_indices = get_reward_gmm_and_subset(device)
    reward_fn = LogReward(gmm, subset_indices)



    checkpoint_path = DIR_ROOT / "src/toy/steer_toy/model/diffusion_gmm_model.pt"
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    alphas_cumprod = checkpoint['alphas_cumprod']
    betas = checkpoint['betas']
    alphas = 1.0 - betas

    # Reconstruct the model architecture
    model = DiffusionModel(input_dim=2, output_dim=2, hidden_dim=512, n_hidden=3).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    toy_class  = analytic_helpers(alphas_cumprod, model, batch_size=1024, device=device)

    def propose_step(t0, t1, z):
        mean = toy_class.expected_next_fn(t0, t1, z)
        if t1 >= 0:
            std = toy_class.std_fn(t0, t1)
            return mean + std * torch.randn_like(z)
        return mean

    def adaptive_resample_safe(time_step, ensemble, potentials):
        weights = torch.nan_to_num(potentials, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.clamp_min(0.0)
        weight_sum = weights.sum()
        safe_denominator = torch.clamp_min(weight_sum, torch.finfo(weights.dtype).tiny)
        normalized_probabilities = weights / safe_denominator
        uniform_probabilities = torch.ones_like(weights) / weights.numel()
        valid_probabilities = torch.isfinite(weight_sum) & (weight_sum > 0)
        probabilities = torch.where(valid_probabilities, normalized_probabilities, uniform_probabilities)

        effective_sample_size = 1.0 / torch.square(probabilities).sum()
        return torch.where(
            effective_sample_size < ensemble.shape[0] // 2,
            fk_steering.ResampleFn.do_resample(ensemble, probabilities),
            ensemble,
        )


    fk = fk_steering.FeynmannKac(
        init_param_fn=partial(torch.randn, toy_class.data_dim, device=toy_class.device),  # data_dims is shape of one data sample, not a batch
        reward_fn=reward_fn.forward,  # function be be maximized (reward function)
        propose_fn=propose_step,  # function that updates the current particles to the next time step
        resample_fn=adaptive_resample_safe,  # function to resample the particles based on their potential
        potential=fk_steering.Potential.MAX,  # how the potential of each particle is computed
        intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
        inverse_temperature=5.,  # the tilt of the target distribution
        kwargs_intermediate_reward=dict(
            expected_sample_fn=toy_class.get_tweedie,  # function to return the expected final samples (at time=1) of the current particles
        ),
    )
    final_samples = fk.run(
    # key,  # PRNG key if using Jax
    toy_class.timesteps,  # sampling time interval discretization
    4,  # number of FK Steering chains to use
    batch_size=toy_class.batch_size,  # number of target samples to generate at once
    )

    print(f'final samples size:{final_samples.size()}')
