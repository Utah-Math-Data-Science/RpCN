import math

import torch
import torch.nn as nn
from torch import Tensor


class ProbabilityPath(nn.Module):
    def __init__(self, data_std=1.):
        super().__init__()
        self.data_std = data_std

    def t(self, i):
        raise NotImplementedError()

    def scale(self, t: Tensor) -> Tensor:
        raise NotImplementedError()

    def noise_level(self, t: Tensor) -> Tensor:
        raise NotImplementedError()

    def c_in(self, noise_level):
        return 1 / (self.data_std**2 + noise_level.square()).sqrt()

    def c_skip(self, noise_level):
        data_std_2 = self.data_std**2
        return data_std_2 / (data_std_2 + noise_level.square())

    def c_out(self, noise_level):
        return self.data_std * noise_level / (self.data_std**2 + noise_level.square()).sqrt()

    def c_noise(self, noise_level):
        return noise_level.log() / 4

    def mean(self, t: Tensor, x: Tensor) -> Tensor:
        return self.scale(t) * x

    def loss_weight(self, noise_level: Tensor) -> Tensor:
        return 1 / self.c_out(noise_level).square()


class VarianceExploding(ProbabilityPath):
    def __init__(
        self,
        data_std,
        sigma_min,
        sigma_max,
    ):
        super().__init__(data_std=data_std)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def t(self, i):
        i = i.to(torch.float64)
        sigma_min_2 = self.sigma_min**2
        t = sigma_min_2 * (
            (self.sigma_max**2 / sigma_min_2)**i
        )
        return t.float()

    def scale(self, t):
        return torch.ones_like(t)

    def noise_level(self, t):
        return t.sqrt()

    # def c_in(self, noise_level):
    #     return torch.ones_like(noise_level)
    #
    # def c_skip(self, noise_level):
    #     return torch.ones_like(noise_level)
    #
    # def c_out(self, noise_level):
    #     return noise_level
    #
    # def c_noise(self, noise_level):
    #     return (0.5 * noise_level).log()


class EDM(ProbabilityPath):
    """
    """

    def __init__(
        self,
        data_std,
        sigma_min,
        sigma_max,
        rho: float = 7.0,
    ):
        super().__init__(data_std=data_std)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def t(self, i):
        i = i.to(torch.float64)
        t = (
            self.sigma_min ** (1 / self.rho)
            + i * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))
        ) ** self.rho
        return t.float()

    def scale(self, t):
        return torch.ones_like(t)

    def noise_level(self, t):
        return t


class VariancePreserving(ProbabilityPath):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2

    Arguments:
        alpha: The choice of :math:`\alpha(t)`.
        alpha_min: The signal scale at ``t=1``.
        t_min: The minimum time used when mapping normalized indices to time.
    """
    def __init__(
        self,
        data_std=1.,
        alpha: str = 'cos',
        alpha_min: float = 1e-3,
        t_min: float = 1e-3,
    ):
        super().__init__(data_std=data_std)
        self.t_min = t_min
        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - alpha_min) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(alpha_min)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(alpha_min) * t**2)
        else:
            raise ValueError()

    def t(self, i):
        return self.t_min + (1 - self.t_min) * i

    def scale(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def noise_level(self, t: Tensor) -> Tensor:
        alpha = self.alpha(t)
        return (1 - alpha.square()).clamp_min(0).sqrt() / alpha
