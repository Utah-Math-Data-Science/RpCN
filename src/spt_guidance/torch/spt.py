import torch


class Kernel:
    """
    shape : tuple[int]
        Shape the param
    """
    def __init__(self, shape, dtype=None, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def propose(self, param, **kwargs):
        raise NotImplementedError()

    def init_param(self):
        raise NotImplementedError()

    def get_acceptance(self, energy, new_energy):
        raise NotImplementedError()

    def parallel_tempering_swap_probability(
        energy_hot, inverse_temperature_hot,
        energy_cold, inverse_temperature_cold,
    ):
        raise NotImplementedError()


class KernelPreconditionedCrankNicolson(Kernel):
    r"""
    Preconditioned Crank-Nicolson kernel.

    Proposals :math:`X_{n + 1}'` are sampled as

    .. math::

        X_{n + 1}' \sim \mathcal{\sqrt{1 - \beta^2}X_n,\beta^2 I}

    where :math:`\beta \in (0, 1)` is the step size.
    The acceptance probability is

    .. math::

        \alpha(X, X') = \min(1, \exp(-\Delta E(X, X')))

    where :math:`\Delta E = \phi(X') - \phi(X)` is the difference between the
    energy of the proposal and the current state.

    Args
    ----
    shape : tuple[int]
        Shape the param
    std : scalar
        The diagonal element of the covariance.
    """
    def __init__(self, shape, std=1):
        super().__init__(shape)
        self.std = std

    def propose(self, param, step_size, drift=None):
        r"""
        step_size : scalar, default None
            Kernel step size :math:`\beta`
        """
        if drift is None:
            drift = torch.sqrt(1 - torch.square(step_size))
        return (
            drift * param
            + step_size * self._sample_noise(param.shape, dtype=param.dtype, device=param.device)
        )

    def init_param(self, dtype=None, device=None):
        return self._sample_noise(self.shape, dtype=dtype, device=device)

    def _sample_noise(self, shape, dtype=None, device=None):
        return torch.randn(shape, dtype=dtype, device=device) * self.std

    @staticmethod
    def get_acceptance(energy, new_energy):
        delta_energy = new_energy - energy
        acceptance_probability = torch.clip(torch.exp(-delta_energy), max=1)
        accept = torch.bernoulli(acceptance_probability).to(torch.bool)
        return accept

    @staticmethod
    def parallel_tempering_swap_probability(
        energy_hot, inverse_temperature_hot,
        energy_cold, inverse_temperature_cold,
    ):
        delta_energy = energy_cold - energy_hot
        delta_inverse_temperature = inverse_temperature_cold - inverse_temperature_hot
        swap_probability = torch.clip(torch.exp(delta_inverse_temperature * delta_energy), max=1)
        return swap_probability


class PTMCMC:
    r"""
    Parallel Tempering Markov Chain Monte Carlo.

    Args
    ----
    kernel : Kernel
        Kernel that updates the params of each chain.
    potential_fn : Callable, default None
        Computes the potential energy of a data sample, which is to be minimized.
    inverse_temperature : torch.Tensor
        One-dimensional array of inverse temperatures :math:`\beta_k = 1/T_k`.
    *args_kernel_propose : torch.Tensor, default None
        One-dimensional arrays of matching the shape of ``inverse_temperature``
        to be passed to the prose function of the kernel.
    """
    def __init__(
        self,
        kernel, potential_fn,
        inverse_temperature,
        /,
        *args_kernel_propose,
    ):
        self.kernel = kernel
        self.potential_fn = potential_fn
        self.inverse_temperature = inverse_temperature
        self.chain_count = torch.numel(inverse_temperature)
        for i, v in enumerate(args_kernel_propose):
            if (shape := v.shape) not in (tuple(), self.inverse_temperature.shape):
                raise ValueError(f"Additional argument {i} to be passed to the propose function of kernel {type(kernel).__name__} must be a scalar (shape ()) or have shape {shape}.")
        self.init_param = torch.vmap(torch.vmap(lambda *_: kernel.init_param(), randomness='different'), randomness='different')
        self.potential_fn_vmap = torch.vmap(potential_fn)
        self.args_kernel_propose = args_kernel_propose
        self.update_chain_jit = torch.vmap(
            self.update_chain,
            in_dims=(0, 0, 0, *([0] * len(args_kernel_propose))),
            randomness='different',
        )

    def run(
        self,
        iters,
        param=None,
        *,
        batch_size=1,
    ):
        r"""
        Parallel Tempering Markov Chain Monte Carlo.

        Args
        ----
        iters : int
            Number of sample updates and chain swaps (replica exchanges).

        Keyword Args
        ------------
        param : torch.Tensor, optional
            The initial param to start running from.
        batch_size : int, optional
            The batch size of params to sample.

        Returns
        -------
        torch.Tensor
            Final sampled params from all chains. The params array shape is (chain_count, batch_size, ...).
        """
        if param is None:
            param = self.init_param(torch.zeros(1).expand(self.chain_count, batch_size)).to(self.inverse_temperature.device)
        energy_with_unit_temperature = self.potential_fn_vmap(param)
        for _ in range(iters):
            param, energy_with_unit_temperature = self.update_chain_jit(
                param, energy_with_unit_temperature,
                self.inverse_temperature,
                *self.args_kernel_propose,
            )
            for k in range(self.chain_count - 1):
                param, energy_with_unit_temperature = self.swap_chain(param, energy_with_unit_temperature, k)
        return param

    def update_chain(self, param, energy_with_unit_temperature, inverse_temperature, *args_kernel_propose):
        new_param = self.kernel.propose(param, *args_kernel_propose)
        new_energy_with_unit_temperature = self.potential_fn(new_param)
        accept = self.kernel.get_acceptance(
            inverse_temperature * energy_with_unit_temperature,
            inverse_temperature * new_energy_with_unit_temperature,
        )
        updated_energy_with_unit_temperature = torch.where(accept, new_energy_with_unit_temperature, energy_with_unit_temperature)
        accept = accept.reshape(*accept.shape, *[1 for _ in range(param.ndim - accept.ndim)])
        updated_param = torch.where(accept, new_param, param)
        return updated_param, updated_energy_with_unit_temperature

    def swap_chain(self, param, energy_with_unit_temperature, k):
        swap_probability = self.kernel.parallel_tempering_swap_probability(
            energy_hot=energy_with_unit_temperature[k], inverse_temperature_hot=self.inverse_temperature[k],
            energy_cold=energy_with_unit_temperature[k + 1], inverse_temperature_cold=self.inverse_temperature[k + 1],
        )
        swap = torch.bernoulli(swap_probability).to(torch.bool)
        energy_temp_1_kp1 = torch.where(swap, energy_with_unit_temperature[k], energy_with_unit_temperature[k + 1])
        energy_temp_1_k = torch.where(swap, energy_with_unit_temperature[k + 1], energy_with_unit_temperature[k])
        energy_with_unit_temperature[k] = energy_temp_1_k
        energy_with_unit_temperature[k + 1] = energy_temp_1_kp1
        swap = swap.reshape(*swap.shape, *[1 for _ in range(param.ndim - 1 - swap.ndim)])
        param_kp1 = torch.where(swap, param[k], param[k + 1])
        param_k = torch.where(swap, param[k + 1], param[k])
        param[k] = param_k
        param[k + 1] = param_kp1
        return param, energy_with_unit_temperature
