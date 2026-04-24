import spt_guidance.torch.rpcn as rpcn
import torch 
from log_reward import LogReward
from model.model import DiffusionModel
from analytic_helpers import analytic_helpers
from utils import plot_samples

if __name__ == "__main__":
    device = "cuda"
    n_chains = 20



    torch.manual_seed(31)

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



    kernel = rpcn.KernelPreconditionedCrankNicolson(
        toy_class.data_dim,  # shape of one data sample, not a batch
        device = toy_class.device,
        dtype = toy_class.dtype
    )


    rpcn = rpcn.ResidualMCMCSteering(
        kernel,
        torch.linspace(torch.pi / 2 - 1e-3, 0.05, n_chains, device=toy_class.device).sin(),  # pCN kernel step size
        reward_fn=reward_fn.forward,  # function be be maximized (reward function)
        expected_next_fn=toy_class.expected_next_fn,  # function that computes the next state's mean given the current state
        std_fn=toy_class.std_fn,  # function that returns the standard deviation of each dimension of the current or next state (see docstring)
        intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
        inverse_temperature=torch.linspace(0., 1., n_chains, device=toy_class.device),  # inverse temperatures
        kwargs_intermediate_reward=dict(
            expected_sample_fn=toy_class.get_tweedie,  # function to return the expected final samples (at time=1) of the current particles
        ),
    )

    def pCN_update_fn(timestep, t0, t1, reward):
        # print(f"pCN update at t={t0}, reward={reward}")
        return timestep % toy_class.check_every == 0


    with torch.no_grad():
        final_samples = rpcn.run(
            # key,  # PRNG key if using Jax
            toy_class.timesteps.to(toy_class.device),  # sampling time interval discretization
            pcn_should_update_fn=pCN_update_fn,  # function to indicate whether the pCN updates should run
            # pcn_should_replica_exchange_fn=lambda t0, t1, reward: t0 % image_class.check_every == 0,  # function to indicate whether to do replica exchange
            pcn_update_count=50,  # number of pcn steps to take per update
            batch_size=toy_class.batch_size,  # number of target samples to generate at once
        )

        final_chain = final_samples[-1]
        plot_samples(final_chain, title="RpCN Steered Samples", name="final_chain.png")