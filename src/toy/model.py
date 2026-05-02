from einops import reduce
import torch
import torch.distributions
from torch.distributions.utils import _sum_rightmost


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


class EnsembleScoreModel:
    def __init__(self, cfg, prob_path):
        self.cfg = cfg
        self.prob_path = prob_path
        self.training_data = None

    @staticmethod
    def log_prob_to_weight_safe(log_prob, failed_to_sum_to_one_threshold=1e-9):
        weights = (log_prob - log_prob.max()).exp()
        weights = torch.where(weights.sum() > failed_to_sum_to_one_threshold, weights, 1.)
        weights_sum = weights.sum()
        weights /= weights_sum
        return weights

    def score(self, t, xt, means, std):
        conditional_prob_path = Independent(Normal(loc=means, scale=std, validate_args=False), 1)
        log_conditional_probs = conditional_prob_path.log_prob_unnormalized(xt)
        weights = self.log_prob_to_weight_safe(log_conditional_probs)
        conditional_scores = -(xt - means) / std.square()
        return reduce(weights[:, None] * conditional_scores, 'member dim -> dim', 'sum')

    def set_training_data(self, training_data):
        self.training_data = training_data

    def _score_particles(self, t, particles):
        if self.training_data is None:
            raise ValueError("training_data is not set. Call set_training_data first.")

        means = self.prob_path.mean(t, self.training_data)
        std = self.prob_path.std(t, self.training_data)
        score = torch.vmap(
            lambda member: self.score(t, member, means, std),
            chunk_size=self.cfg.vmap_chunk_size,
        )(particles)
        return score.clamp(min=-self.cfg.sampling_max_score_norm, max=self.cfg.sampling_max_score_norm)

    def expected_next_fn(self, t0, t1, latents):
        dt = t1 - t0
        score = self._score_particles(t0, latents)
        g = self.prob_path.g(t0)
        drift = self.prob_path.f(t0, latents) - g.square() * score
        return latents + dt * drift

    def std_fn(self, t0, t1=None):
        if t1 is None:
            return torch.tensor(1.0, device=self.training_data.device, dtype=self.training_data.dtype)
        dt = t1 - t0
        return dt.abs().sqrt() * self.prob_path.g(t0)

    @torch.no_grad()
    def get_tweedie(self, t, latents):
        score = self._score_particles(t, latents)
        alpha_t = self.prob_path.alpha(t)
        beta_t = self.prob_path.beta(t)
        return (latents + beta_t.square() * score) / alpha_t

    def sample(self, time_step, training_data):
        path_time = self.prob_path.linspace_time(self.cfg.sampling_time_step_count, device=training_data.device)
        minus_time_step_size = path_time[1] - path_time[0]
        minus_time_step_size_abs_sqrt = minus_time_step_size.abs().sqrt()
        ensemblet = self.prob_path.sample_noise(path_time[0], training_data)
        for time_step, t_now_and_next in enumerate(path_time.unfold(0, 2, 1)):
            t_now = t_now_and_next[0]
            means = self.prob_path.mean(t_now, training_data)
            std = self.prob_path.std(t_now, training_data)
            score = torch.vmap(
                lambda member: self.score(t_now, member, means, std),
                chunk_size=self.cfg.vmap_chunk_size,
            )(ensemblet)
            score = score.clamp(min=-self.cfg.sampling_max_score_norm, max=self.cfg.sampling_max_score_norm)
            g = self.prob_path.g(t_now)
            state_drift = ensemblet + minus_time_step_size * (self.prob_path.f(t_now, ensemblet) - g.square() * score)
            noise = torch.randn(
                ensemblet.shape,
                device=ensemblet.device,
                dtype=ensemblet.dtype,
            )
            ensemblet = state_drift + minus_time_step_size_abs_sqrt * g * noise
            ensemble = ensemblet
        return ensemble
