from functools import partial
from pathlib import Path
from model.model import MeanFlowModel
import torch
from rewards import left, right, top, bottom, left_node
# from steering import SourceTemperingSampler
from plotting import plot_steered_vs_unguided
import argparse
import spt_guidance.torch.spt as spt
import spt_guidance.torch.fk_steering as fk_steering
import spt_guidance.torch.rpcn as rpcn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--distribution",
        help="please indicate target distribution",
        required=True,
        choices=["Mixture", "Circle"]
    )

    parser.add_argument(
        "--guidance",
        help="please indicate the guidance method",
        required=True,
        choices=["SPT", "FK", "RpCN"]
    )

    parser.add_argument(
        "--save",
        help="please indicate whether or not you want to save data",
        required=True,
        type=int,
        choices=[0, 1]
    )
    parser.add_argument(
        "--plot",
        help="please indicate whether or not you want to plot results",
        required=True,
        type=int,
        choices=[0, 1]
    )

    args =parser.parse_args()

    # params
    reward_fn = right # see rewards.py for options
    guidance_strength = 10.0
    mean_flow_steps = 10
    num_chains = 5
    batch_size = 1024 # number of points in chain


    model = MeanFlowModel(
        input_dim=2,
        output_dim=2,
        dim=256,
        n_hidden=2
    ).to(device)


    state_dict = torch.load(
        Path(__file__).parent/f"model/meanflow_{args.distribution}.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()




    # sampler = SourceTemperingSampler(
    #     model=model,
    #     reward_fn=reward_fn,
    #     beta = guidance_strength,          # strength of steering
    #     n_chains=num_chains,        # PT ladder
    #     n_time_steps=mean_flow_steps,    # MeanFlow steps
    #     device=device
    # )
    data_dim = 2
    grid = torch.linspace(0.0, 1.0, 20 + 1, device=device)

    @torch.no_grad()
    def transport(z):
        """
        Deterministic MeanFlow transport T(z) using Forward Euler.
        z: [batch, dim]
        """
        for i in range(1, len(grid)):

            t_prev = grid[i-1]
            t_curr =  grid[i]
            dt = t_curr - t_prev

            vel = model(
                z,
                t_curr.expand(z.shape[0], 1),
                t_prev.expand(z.shape[0], 1)
            )

            z = z - dt * vel
        return z

    if args.guidance == "SPT":
        @torch.no_grad()
        def get_energy(z):
            """
            Pullback energy: E(z) = -R(T(z))
            z: [batch_size, dim]
            Returns: [batch_size]
            """
            x = transport(z)

            rewards = reward_fn(x)
            energy = -rewards

            return energy

        kernel = spt.KernelPreconditionedCrankNicolson(data_dim)
        ptmcmc = spt.PTMCMC(
            kernel, get_energy,
            torch.linspace(0., 5., num_chains, device=device),
            torch.linspace(torch.pi / 2 - 1e-3, 0.05, num_chains, device=device).sin(),
        )
        last_chain = ptmcmc.run(100, batch_size=batch_size)[-1]
        transported_samples = transport(last_chain)
    elif args.guidance == "FK":
        """
        Note: In practice, a diffusion model should be used with FK steering.
        Not a mean flow model.
        """
        @torch.no_grad()
        def propose_fn(t0, t1, z):
            vel = model(
                z,
                (1 - t0).expand(z.shape[0], 1),
                (1 - t1).expand(z.shape[0], 1)
            )

            return z - (t1 - t0) * vel
        transported_samples = fk_steering.FeynmannKac(
            init_param_fn=partial(torch.randn, 2, device=device),
            reward_fn=reward_fn,
            propose_fn=propose_fn,
            resample_fn=fk_steering.ResampleFn.adaptive_resample,
            potential=fk_steering.Potential.MAX,
            # intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,
            intermediate_reward=fk_steering.IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD,
            inverse_temperature=5.,
            kwargs_intermediate_reward=dict(
                # expected_sample_fn=lambda t0, z: propose_fn(t0, torch.tensor(1., device=device), z),
                sub_ensemble_size=1,
                pushforward_fn=lambda t0, z: propose_fn(t0, torch.tensor(1., device=device), z),
            ),
        ).run(grid, 4, batch_size=batch_size)
    elif args.guidance == "RpCN":
        """
        Note: In practice, a diffusion model should be used with RpCN.
        Not a mean flow model.
        """
        @torch.no_grad()
        def propose_fn(t0, t1, z):
            vel = model(
                z,
                (1 - t0).expand(z.shape[0], 1),
                (1 - t1).expand(z.shape[0], 1)
            )

            return z - (t1 - t0) * vel
        kernel = rpcn.KernelPreconditionedCrankNicolson(data_dim, device=device)
        rpcn = rpcn.ResidualMCMCSteering(
            kernel,
            torch.linspace(torch.pi / 2 - 1e-3, 0.05, num_chains, device=device).sin(),  # pCN kernel step size
            reward_fn=reward_fn,  # function be be maximized (reward function)
            expected_next_fn=propose_fn,
            std_fn=lambda t0, t1=None: (t1**2 - t0**2).sqrt() if t1 is not None else t0,  # function that returns the standard deviation of each dimension of the current or next state (see docstring)
            intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
            inverse_temperature=torch.linspace(0., 5., num_chains, device=device),  # inverse temperatures
            kwargs_intermediate_reward=dict(
                expected_sample_fn=lambda t0, z: propose_fn(t0, torch.tensor(1., device=device), z),
            ),
        )
        transported_samples = rpcn.run(
           grid,  # sampling time interval discretization
           pcn_update_count=8,  # number of pcn steps to take per update
           batch_size=batch_size,  # number of target samples to generate at once
        )



    # if args.save:
    #     torch.save(
    #         {
    #             "z": last_chain.cpu(),
    #             "x": transported_samples.cpu(),
    #         },
    #         Path(__file__).parent/f'results/data/steered_meanflow_{args.distribution}_{reward_fn.__name__}.pt'
    #     )




    with torch.no_grad():
        z_untampered = torch.randn(batch_size, 2, device=device)
        x_untampered = transport(z_untampered).cpu()

    if args.plot:
        plot_steered_vs_unguided(x_untampered, transported_samples.cpu(), f"{args.distribution}_{reward_fn.__name__}")

