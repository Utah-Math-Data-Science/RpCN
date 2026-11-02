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


class EnsembleDenoiser:
    @staticmethod
    def log_prob_to_weight_safe(log_prob, failed_to_sum_to_one_threshold=1e-9):
        weights = (log_prob - log_prob.max()).exp()
        weights = torch.where(weights.sum() > failed_to_sum_to_one_threshold, weights, 1.)
        weights_sum = weights.sum()
        weights /= weights_sum
        return weights

    @classmethod
    def denoise(
        cls,
        training_samples,
        noise_level, particle,
        smoothing_std=0.0, smoothing_samples=1,
        generator=None,
    ):
        conditional_prob_path = Independent(
            Normal(loc=training_samples, scale=noise_level, validate_args=False),
            1,
        )
        if smoothing_std > 0:
            perturbed = particle + smoothing_std * torch.randn((smoothing_samples, *particle.shape), device=particle.device, dtype=particle.dtype, generator=generator)
            log_conditional_probs = conditional_prob_path.log_prob_unnormalized(perturbed[:, None])
            weights = torch.vmap(cls.log_prob_to_weight_safe)(log_conditional_probs).mean(0)
        else:
            log_conditional_probs = conditional_prob_path.log_prob_unnormalized(particle)
            weights = cls.log_prob_to_weight_safe(log_conditional_probs)
        return reduce(weights[:, None] * training_samples, 'member ... -> ...', 'sum')

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
