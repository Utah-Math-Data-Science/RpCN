import torch
from rewards.DAPS_operators import get_operator
from utils import load_image_by_id


def guidance_schedule(t, num_timesteps, kind="late_quad", start_frac=0.5):
    """
    Returns a scalar coeff in [0, 1] for guidance strength at timestep t.
    Reverse loop convention: t goes from (T-1) ... 0.
    Coeff increases as t decreases (later denoising => stronger guidance).
    """
    T = int(num_timesteps)
    if T <= 1:
        return 1.0
    # u: 0 at start (t=T-1), 1 at end (t=0)
    u = 1.0 - (float(t) / float(T - 1))
    u = max(0.0, min(1.0, u))
    if kind == "linear":
        return u
    if kind == "quad":
        return u * u
    if kind == "late":
        # hard turn-on after start_frac of the trajectory
        return 1.0 if u >= float(start_frac) else 0.0
    if kind == "late_quad":
        # smooth late-only ramp; 0 until start_frac, then quadratic to 1
        sf = float(start_frac)
        if u <= sf:
            return 0.0
        v = (u - sf) / (1.0 - sf)
        return v * v
    raise ValueError(f"Unknown kind: {kind}")


class NoisePCNSampler:
    def __init__(
        self,
        model,
        sampler,
        n_chains=2,
        batch_size=3,
        resampling=10,
        guidance_strength=1,
        guidance_schedule="linear",
        grad_norm=True,
        image_id=60000,
        operator="gaussian_blur",
        device="cuda",
    ):
        self.model = model
        self.sampler = sampler
        self.device = torch.device(device)

        self.ref_image = load_image_by_id(image_id, device=device).detach()

        self.operator = get_operator(operator)
        self.y = self.operator(self.ref_image).detach()

        self.n_chains = n_chains
        self.batch_size = batch_size

        self.betas = torch.linspace(
            0.0, 1, n_chains, device=device, dtype=torch.float16
        )
        self.thetas = torch.linspace(
            torch.pi / 2, 0.01, n_chains, device=device, dtype=torch.float16
        )
        self.resampling = resampling

        self.guidance_strength = guidance_strength
        self.guidance_schedule = guidance_schedule
        self.grad_norm = grad_norm

    def get_reward_gradient(self, xt, t_batch):
        with torch.enable_grad():
            xt = xt.detach().requires_grad_(True)
            t_in = self.sampler._scale_timesteps(t_batch)
            model_out = self.model(xt, t_in)

            noise_pred, _ = torch.split(model_out, xt.shape[1], dim=1)
 

            x0_raw = self.sampler.mean_processor.predict_xstart(xt, t_batch, noise_pred)
            x0 = self.sampler.mean_processor.process_xstart(x0_raw)

            reward = self.operator.log_likelihood(x0, self.y)
            grad = torch.autograd.grad(reward.sum(), xt)[0]

        if not self.grad_norm:
            return grad.detach()

        grad_flat = grad.flatten(1)
        grad_norm = grad_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return (grad_flat / grad_norm).view_as(grad).detach()

    @torch.no_grad()
    def sample_initial_latents(self, batch_size, n_chains, flat=False):
        if flat == True:
            latents = torch.randn(
                (n_chains * batch_size, 3, 256, 256), device=self.device
            )
        else:
            latents = torch.randn(
                (n_chains, batch_size, 3, 256, 256), device=self.device
            )

        return latents

    @torch.no_grad()
    def pCN_proposal(self, eps, theta):
        """
        pCN proposal on zero-mean Gaussian residual eps.
        """
        xi = torch.randn_like(eps)
        eps_prop = torch.cos(theta) * eps + torch.sin(theta) * xi
        return eps_prop

    @torch.no_grad()
    def pCN_step(self, mu, sigma_t, t_idx: int, beta, theta):
        # mu, sigma_t: [B, C, H, W] for ONE chain
        B = mu.shape[0]
        t_batch = torch.full((B,), int(t_idx), device=mu.device, dtype=torch.long)
        eps_curr = torch.randn_like(mu)
        x_curr = mu + sigma_t * eps_curr
        tweedie_curr = self.sampler.p_mean_variance(
            model=self.model, x=x_curr, t=t_batch
        )["pred_xstart"]

        reward_curr = self.operator.log_likelihood(tweedie_curr, self.y)  # [B]
        for _ in range(10):
            eps_prop = self.pCN_proposal(
                eps_curr, theta
            )  # theta is scalar for this chain
            x_prop = mu + sigma_t * eps_prop
            tweedie_prop = self.sampler.p_mean_variance(
                model=self.model, x=x_prop, t=t_batch
            )["pred_xstart"]
            reward_prop = self.operator.log_likelihood(tweedie_prop, self.y)  # [B]
            log_alpha = beta * (reward_prop - reward_curr)  # beta scalar
            accept = torch.log(torch.rand_like(reward_curr)) < log_alpha  # [B]
            mask = accept[:, None, None, None]  # broadcast to [B,C,H,W]
            x_curr = torch.where(mask, x_prop, x_curr)
            eps_curr = torch.where(mask, eps_prop, eps_curr)
            reward_curr = torch.where(accept, reward_prop, reward_curr)
        return x_curr, reward_curr  # x_curr: [B,C,H,W], reward_curr: [B]

    def pt_swap(self, latents, rewards):
        # Iterate through adjacent chains
        for n in range(self.n_chains - 1):
            # Metropolis-Hastings swap ratio
            # (beta_low - beta_high) * (reward_high - reward_low)
            delta_beta = self.betas[n] - self.betas[n + 1]
            delta_reward = rewards[n + 1] - rewards[n]

            log_alpha_swap = delta_beta * delta_reward
            accept_swap = torch.log(torch.rand_like(log_alpha_swap)) < log_alpha_swap

            # Swap latents and rewards for the batch items that accepted
            idx = accept_swap.nonzero().squeeze(-1)
            if idx.numel() > 0:
                latents[n, idx], latents[n + 1, idx] = (
                    latents[n + 1, idx].clone(),
                    latents[n, idx].clone(),
                )
                rewards[n, idx], rewards[n + 1, idx] = (
                    rewards[n + 1, idx].clone(),
                    rewards[n, idx].clone(),
                )

        return latents, rewards

    @torch.no_grad()
    def transport_parallel(self):
        x = self.sample_initial_latents(self.batch_size, self.n_chains, True)

        for t in reversed(range(self.sampler.num_timesteps)):

            t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)

            out = self.sampler.p_mean_variance(model=self.model, x=x, t=t_batch)
            mu_ug = out["mean"]
            sigma_t = out["variance"] ** 0.5
            grad = self.get_reward_gradient(x, t_batch)

            sched = guidance_schedule(
                t, self.sampler.num_timesteps, self.guidance_schedule
            )

            mu = mu_ug + self.guidance_strength * sched * grad

            if t % self.resampling == 0:
                mu_c = mu.view(self.n_chains, self.batch_size, 3, 256, 256)
                sig_c = sigma_t.view(self.n_chains, self.batch_size, 3, 256, 256)
                betas = self.betas.to(dtype=mu.dtype)
                thetas = self.thetas.to(dtype=mu.dtype)
                vmapped_pcn = torch.vmap(
                    self.pCN_step,
                    in_dims=(
                        0,
                        0,
                        None,
                        0,
                        0,
                    ),  # map over chains for mu/sigma/beta/theta
                    randomness="different",
                )
                x_c, rewards = vmapped_pcn(
                    mu_c, sig_c, t, betas, thetas
                )  # x_c: [n_chains,B,C,H,W]
                x_c, rewards = self.pt_swap(x_c, rewards)
                x = x_c.view(self.n_chains * self.batch_size, 3, 256, 256)
            else:
                eps = torch.randn_like(mu) if t > 0 else torch.zeros_like(mu)
                x = mu + sigma_t * eps

        return x, rewards
