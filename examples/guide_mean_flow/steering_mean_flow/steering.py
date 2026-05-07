import torch
import math


class SourceTemperingSampler:
    """
    Source-space Parallel Tempering MCMC for a pretrained MeanFlow model.
    Non-adaptive pCN + Parallel Tempering.
    """

    def __init__(
        self,
        model,
        reward_fn,
        beta=5.0,
        n_chains=10,
        n_time_steps=20,
        device="cpu",
        data_dim = 2
    ):
        self.model = model.to(device)
        self.model.eval()

        self.reward_fn = reward_fn
        self.device = device
        self.n_chains = n_chains

        self.betas = torch.linspace(0.0, beta, n_chains, device=device)

        self.thetas = torch.linspace(
            math.pi / 2, 0.05, n_chains, device=device
        )

        self.grid = torch.linspace(
            0.0, 1.0, n_time_steps + 1, device=device
        )

        self.data_dim = data_dim

    @torch.no_grad()
    def transport(self, z):
        """
        Deterministic MeanFlow transport T(z) using Forward Euler.
        z: [n_chains, batch, dim]
        """
        orig_shape = z.shape

        # Flatten chains and batches to process through the model efficiently
        
        z_flat = z.view(-1, self.data_dim)
        all_batches = z_flat.shape[0]

        for i in range(1, len(self.grid)):

            t_prev = self.grid[i-1]
            t_curr =  self.grid[i]
            dt = t_curr - t_prev

            vel = self.model(
                z_flat, 
                t_curr.expand(all_batches, 1), 
                t_prev.expand(all_batches, 1)
            )

            z_flat = z_flat - dt * vel
        #reshape back to [n_chains, batch, dim]
        return z_flat.view(orig_shape)


    @torch.no_grad()
    def get_energy(self, z):
        """
        Pullback energy: E(z) = R(T(z))
        z: [n_chains, batch_size, dim]
        Returns: [n_chains, batch_size]
        """
        # 1. Transport latent z to data space x (maintains shape [K, B, D])
        x = self.transport(z)

        # 2. Flatten for reward function if needed: [K*B, D]
        x_flat = x.reshape(-1, x.shape[-1])

        # 3. Compute rewards on flattened x: [K*B]
        rewards = self.reward_fn(x_flat)

        # 4. Return with proper batch structure: [K, B]
        return rewards.reshape(z.shape[0], z.shape[1])


    @torch.no_grad()
    def propose_updates(self, z, energies):
        """
        Propose PCN updates for each chain and accept/reject.
        """
        theta = self.thetas.view(-1, 1, 1)
        beta_k = self.betas.view(-1, 1)

        xi = torch.randn_like(z)

        # pCN proposal across all chains and batches shape ([K, B, D])
        z_prop = torch.cos(theta) * z + torch.sin(theta) * xi

        E_prop = self.get_energy(z_prop)

        # Metropolis acceptance
        # Equivalent to: log( P(accept) ) = beta * (Reward_new - Reward_old)
        log_alpha = -beta_k * (E_prop - energies)

        accept = torch.rand_like(log_alpha).log() < log_alpha

        # accept.unsqueeze(-1) has shape [K, B,1], 
        # (for each chain and point in the batch, should the proposal or should original point be taken?)
        
        z_new = torch.where(accept.unsqueeze(-1), z_prop, z)
        energies_new = torch.where(accept, E_prop, energies)

        # now return the samples and energies
        return z_new, energies_new

    # ------------------------------------------------------------------
    # Replica exchange
    # ------------------------------------------------------------------

    @torch.no_grad()


    def swap_between_chains(self, z, energies, ):

        for k in range(self.n_chains - 1):
            delta_beta = self.betas[k + 1] - self.betas[k]
            delta_E = energies[k + 1] - energies[k]

            log_alpha = -delta_beta * delta_E
            # for each chain, look at all batch points and decide whether to swap or not
            accept = torch.rand(z.shape[1], device=self.device).log() < log_alpha

            # print(f"Replica exchange between chains {k} and {k+1}:")
            num_accepted = accept.sum().item()
            # print(f"Accepted {num_accepted} out of {z.shape[1]} samples.")
            
            if accept.any():
                z_k = z[k].clone()
                E_k = energies[k].clone()

                # swap the accepted points
                z[k, accept] = z[k + 1, accept]
                energies[k, accept] = energies[k + 1, accept]

                z[k + 1, accept] = z_k[accept]
                energies[k + 1, accept] = E_k[accept]
        return z, energies

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n_iterations,
        batch_size,
    ):
        """
        Run the sampler and return samples from the coldest chain.
        """
        # Initialize from prior
        z = torch.randn(
            self.n_chains, batch_size, self.data_dim, device=self.device
        )
        
        energies = self.get_energy(z)

        for it in range(n_iterations):
            z, energies = self.propose_updates(z, energies)

 
            z, energies = self.swap_between_chains(
                z, energies
            )

        # z has shape [n_chains, batch_size, data_dim], so take the last chain as the coldest
        last_chain = z[-1]
        transported_samples = self.transport(last_chain)

        return last_chain, transported_samples