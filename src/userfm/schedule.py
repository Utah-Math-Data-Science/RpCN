"""ttg schedules for this project's probability paths."""
from dataclasses import dataclass

import jax.numpy as jnp
import ttg


@dataclass(frozen=True)
class VarianceExploding:
    r"""
    ttg schedule for :class:`userfm.sde_diffusion.VarianceExploding`.

    The native time :math:`\tau` is the diffusion time, :math:`\tau = 1 - t`
    for the model time :math:`t` (data at :math:`t = 1`).
    Then :math:`x_\tau = x_0 + \sigma(\tau) \epsilon` with
    :math:`\sigma(\tau) = \sigma_{\min} \sqrt{(\sigma_{\max} / \sigma_{\min})^{2\tau} - 1}`.
    ``time`` maps the schedule coordinate :math:`u \in [0, 1]` linearly onto
    :math:`[\tau_{\min}, \tau_{\max}]`, so a uniform :math:`u` grid gives
    uniformly spaced model times.
    """
    sigma_min: float
    sigma_max: float
    t_min: float
    t_max: float = 1.

    @property
    def _log_ratio(self):
        return jnp.log(self.sigma_max / self.sigma_min)

    def time(self, u):
        return self.t_min + u * (self.t_max - self.t_min)

    def mean(self, t):
        return jnp.ones_like(t)

    def std(self, t):
        return self.sigma_min * jnp.sqrt(jnp.expm1(2 * self._log_ratio * t))

    def dt_mean(self, t):
        return jnp.zeros_like(t)

    def dt_std(self, t):
        L = self._log_ratio
        return self.sigma_min * L * jnp.exp(2 * L * t) / jnp.sqrt(jnp.expm1(2 * L * t))

    def prior(self, leading_shape, event_shape, *, xp, dtype, device=None):
        return ttg.Gaussian(
            xp.zeros((*leading_shape, *event_shape), dtype=dtype),
            xp.full(tuple(leading_shape), self.std(jnp.asarray(self.t_max, dtype=dtype))**2, dtype=dtype),
        )

    def log_snr(self, t):
        return -2 * jnp.log(self.std(t))

    def log_snr_diff(self, t_a, t_b):
        # log(sigma(t_b)^2 / sigma(t_a)^2), computed through t_a - t_b to stay accurate for small gaps
        L = self._log_ratio
        return jnp.log1p(-jnp.exp(2 * L * t_b) * jnp.expm1(2 * L * (t_a - t_b)) / jnp.expm1(2 * L * t_a))
