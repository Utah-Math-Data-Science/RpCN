from functools import partial

from einops import reduce
import torch
import torch.distributions
from torch.distributions.utils import _sum_rightmost


class Euler:
    @staticmethod
    def stochastic(noise_level, noise_level_next):
        if noise_level_next == 0:
            return torch.zeros_like(noise_level_next)
        return noise_level.square() - noise_level_next.square()

    @classmethod
    @torch.no_grad()
    def step(cls, denoiser, noise_level, noise_level_next, x, noise_sampler=None):
        """Take one reverse-VE-SDE Euler step in normalized coordinates.

        ``x`` is distributed as ``x_0 + noise_level * eps``.  ``denoiser`` must
        be callable as ``denoiser(noise_level, x)`` and return an estimate of
        ``x_0``.  With ``noise_sampler=None`` the step is deterministic.  To
        sample the reverse SDE, pass a callable such as ``torch.randn``;
        it receives ``x.shape`` and must return noise with the same shape.
        """
        denoised = denoiser(noise_level, x)
        if noise_level_next == 0:
            return denoised
        variance_step = cls.stochastic(noise_level, noise_level_next)
        score = (denoised - x) / noise_level.square()
        noise = 0 if noise_sampler is None else noise_sampler(x.shape)
        return x + variance_step * score + variance_step.sqrt() * noise


@torch.no_grad()
def metropolis_hastings_sample(
    log_target_fn,
    num_chains,
    num_samples,
    dim,
    burn_in=0,
    proposal_std=1.0,
    thinning=1,
    initial_x=None,
    device=None,
    dtype=torch.float32,
    global_proposal_std=None,
    global_proposal_probability=0.0,
):
    if num_chains <= 0:
        raise ValueError("num_chains must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if proposal_std <= 0:
        raise ValueError("proposal_std must be positive")
    if not 0.0 <= global_proposal_probability <= 1.0:
        raise ValueError("global_proposal_probability must be between zero and one")
    if global_proposal_probability > 0.0 and (
        global_proposal_std is None or global_proposal_std <= 0
    ):
        raise ValueError(
            "global_proposal_std must be positive when global proposals are enabled"
        )
    if thinning <= 0:
        raise ValueError("thinning must be positive")

    if initial_x is not None:
        if initial_x.shape != (num_chains, dim):
            raise ValueError("initial_x must have shape [num_chains, dim]")
        x = initial_x
        if device is None:
            device = x.device
        if dtype is None:
            dtype = x.dtype
        x = x.to(device=device, dtype=dtype)
    else:
        if device is None:
            device = torch.device("cpu")
        x = torch.zeros((num_chains, dim), device=device, dtype=dtype)

    logp = log_target_fn(x)
    if logp.shape != (num_chains,):
        raise ValueError("log_target_fn must return shape [num_chains]")

    total_steps = burn_in + num_samples * thinning
    samples = []
    accepted = 0
    proposed = 0
    local_accepted = 0
    local_proposed = 0
    global_accepted = 0
    global_proposed = 0
    use_global_proposals = global_proposal_probability > 0.0

    for step_idx in range(total_steps):
        if use_global_proposals:
            global_mask = (
                torch.rand((num_chains,), device=x.device, dtype=x.dtype)
                < global_proposal_probability
            )
            proposal_scales = torch.where(
                global_mask,
                torch.as_tensor(global_proposal_std, device=x.device, dtype=x.dtype),
                torch.as_tensor(proposal_std, device=x.device, dtype=x.dtype),
            )
            proposal = x + proposal_scales[:, None] * torch.randn_like(x)
        else:
            global_mask = torch.zeros(num_chains, device=x.device, dtype=torch.bool)
            proposal = x + proposal_std * torch.randn_like(x)
        logp_proposal = log_target_fn(proposal)
        if logp_proposal.shape != (num_chains,):
            raise ValueError("log_target_fn must return shape [num_chains]")

        log_alpha = torch.clamp(logp_proposal - logp, max=0.0)
        log_u = torch.log(torch.rand((num_chains,), device=x.device, dtype=x.dtype))
        accept = log_u < log_alpha

        x = torch.where(accept[:, None], proposal, x)
        logp = torch.where(accept, logp_proposal, logp)

        accepted += int(accept.sum().item())
        proposed += num_chains
        global_accepted += int((accept & global_mask).sum().item())
        global_proposed += int(global_mask.sum().item())
        local_accepted += int((accept & ~global_mask).sum().item())
        local_proposed += int((~global_mask).sum().item())

        if step_idx >= burn_in and ((step_idx - burn_in) % thinning == 0):
            samples.append(x.clone())

    if len(samples) != num_samples:
        raise RuntimeError("Internal sampling error: collected unexpected number of samples")

    stacked_samples = torch.stack(samples, dim=0).permute(1, 0, 2)
    diagnostics = {
        "acceptance_rate": accepted / proposed,
        "accepted": accepted,
        "proposed": proposed,
        "local_acceptance_rate": (
            local_accepted / local_proposed if local_proposed else None
        ),
        "local_accepted": local_accepted,
        "local_proposed": local_proposed,
        "global_acceptance_rate": (
            global_accepted / global_proposed if global_proposed else None
        ),
        "global_accepted": global_accepted,
        "global_proposed": global_proposed,
    }
    return stacked_samples, diagnostics


class Independent(torch.distributions.Independent):
    def log_prob_unnormalized(self, value):
        log_prob_unnormalized = self.base_dist.log_prob_unnormalized(value)
        return _sum_rightmost(log_prob_unnormalized, self.reinterpreted_batch_ndims)


class Normal(torch.distributions.Normal):
    def log_prob_unnormalized(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        return -((value - self.loc) ** 2) / (2 * var)


class GaussianPath:
    """
    Gaussian conditional probability path.
    """
    def __init__(self, cfg, target_distribution_at_time_1=False):
        r"""
        Parameters
        ----------
        cfg: conf.prob_path.DiffusionPath
            The configuration of the Gaussian probability path.

        target_distribution_at_time_1: bool
            True if the probability path approximates the target distribution
            when :math:`t = 1` and is the noise distribution at :math:`t = 0`.
            Otherwise, the path approximates the target distribution when
            :math:`t = 0`.
        """
        self.cfg = cfg
        self.target_distribution_at_time_1 = target_distribution_at_time_1

    @staticmethod
    def rng_normal(shape, std=1., **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.float32
        if std > 0:
            noise = torch.randn(size=shape, **kwargs) * std
        else:
            noise = 0.
        return noise


class Bao2024EnsembleScoreMatching(GaussianPath):
    def _reverse_time(self, t):
        r"""
        Reflects the time `t` in the interval :math:`[t_\min, 1]`.
        """
        return 1 - t

    def sample_time(self, t_shape, **kwargs):
        return self.rng_random(t_shape, self.rng_time, **kwargs)

    def linspace_time(self, time_step_count, **kwargs):
        t = torch.linspace(1., 0., time_step_count, **kwargs)
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return t

    def alpha(self, t):
        return 1 - t * (1 - self.cfg.epsilon_alpha)

    def dt_log_alpha(self, t):
        return -(1 - self.cfg.epsilon_alpha) / self.alpha(t)

    def beta(self, t):
        return (self.cfg.epsilon_beta + t * (1 - self.cfg.epsilon_beta))**(1/2)

    def dt_squared_beta(self, t):
        return 1 - self.cfg.epsilon_beta

    def mean(self, t, data):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return self.alpha(t) * data

    def dt_mean(self, t, data):
        raise NotImplementedError()

    def std(self, t, data):
        r"""
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return self.beta(t)

    def dt_std(self, t, data):
        raise NotImplementedError()

    def f(self, t, data):
        return self.dt_log_alpha(t) * data

    def g(self, t):
        if self.target_distribution_at_time_1:
            raise NotImplementedError()
        return (self.dt_squared_beta(t) - 2 * self.dt_log_alpha(t) * self.beta(t).square()).sqrt()

    def sample_noise(self, t, data):
        """
        Sample noise with mean zero and standard deviation at time :math:`t`.
        This method makes sense when :math:`t` is the time where the
        probability path is closest to the noise distribution.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        if self.target_distribution_at_time_1:
            raise NotImplementedError()
        noise = self.rng_normal(data.shape, device=data.device)
        if self.cfg.renormalize_sampled_noise:
            noise = (noise - noise.mean(0)) / noise.std(0)
        if self.cfg.sample_noise_scale_std:
            noise = noise * self.std(t, data)
        if self.cfg.sample_noise_add_mean:
            noise = self.mean(t, data) + noise
        return noise


class EnsembleDenoiser:
    @staticmethod
    def log_prob_to_weight_safe(log_prob, failed_to_sum_to_one_threshold=1e-9):
        weights = (log_prob - log_prob.max()).exp()
        weights = torch.where(weights.sum() > failed_to_sum_to_one_threshold, weights, 1.)
        weights_sum = weights.sum()
        weights /= weights_sum
        return weights

    @classmethod
    def denoise(cls, training_samples, noise_level, particle):
        conditional_prob_path = Independent(
            Normal(loc=training_samples, scale=noise_level, validate_args=False),
            1,
        )
        log_conditional_probs = conditional_prob_path.log_prob_unnormalized(particle)
        weights = cls.log_prob_to_weight_safe(log_conditional_probs)
        return reduce(weights[:, None] * training_samples, 'member ... -> ...', 'sum')

    # @classmethod
    # @torch.no_grad()
    # def forward(cls, prob_path, training_samples, noise_level, particles, vmap_chunk_size=None):
    #     """Return the ensemble posterior mean in normalized EDM coordinates."""
    #     return torch.vmap(
    #         lambda particle: cls._denoise_particle(training_samples, noise_level, particle),
    #         chunk_size=vmap_chunk_size,
    #     )(particles)

    @staticmethod
    def next_fn(denoiser, prob_path, t0, t1, particles, noise_sampler=None):
        normalized_next = Euler.step(
            denoiser,
            prob_path.noise_level(t0), prob_path.noise_level(t1),
            particles / prob_path.scale(t0),
            noise_sampler=noise_sampler,
        )
        return normalized_next * prob_path.scale(t1)

    @staticmethod
    def tweedie(denoiser, prob_path, t, particles):
        return denoiser(prob_path.noise_level(t), particles / prob_path.scale(t))

    @staticmethod
    def std_fn(prob_path, t0, t1=None):
        noise_level = prob_path.noise_level(t0)
        if t1 is None:
            return noise_level * prob_path.scale(t0)
        noise_level_next = prob_path.noise_level(t1)
        if noise_level_next == 0:
            return torch.zeros_like(noise_level_next)
        variance_step = Euler.stochastic(noise_level, noise_level_next)
        return variance_step.sqrt() * prob_path.scale(t1)

    @classmethod
    def sample(cls, denoiser, prob_path, training_samples, solver, count, i0, i1, steps, generator=None):
        device, dtype = training_samples.device, training_samples.dtype
        i = torch.linspace(i0, i1, steps, device=device, dtype=dtype)
        t = prob_path.t(i)
        noise_levels = prob_path.noise_level(t)
        x = noise_levels[0] * torch.randn((count, *training_samples.shape[1:]), device=device, dtype=dtype, generator=generator)
        for noise_level, noise_level_next in noise_levels.unfold(0, 2, 1):
            x = solver(
                denoiser,
                noise_level, noise_level_next,
                x,
                noise_sampler=partial(torch.randn, device=device, dtype=dtype, generator=generator)
            )
        return denoiser(noise_levels[-1], x)
