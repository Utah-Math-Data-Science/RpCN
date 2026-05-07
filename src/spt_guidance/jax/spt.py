from functools import partial

import jax
import jax.random
import jax.numpy as jnp


class Kernel:
    """
    shape : tuple[int]
        Shape the param
    """
    def __init__(self, shape):
        self.shape = shape

    def propose(self, key, param, **kwargs):
        raise NotImplementedError()

    def init_param(self, key):
        raise NotImplementedError()

    def get_acceptance(self, key, energy, new_energy):
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

    @partial(jax.jit, static_argnames=['self'])
    def propose(self, key, param, step_size, drift=None):
        r"""
        step_size : scalar, default None
            Kernel step size :math:`\beta`
        """
        if drift is None:
            drift = jnp.sqrt(1 - jnp.square(step_size))
        return (
            drift * param
            + step_size * self._sample_noise(key, param.shape, dtype=param.dtype)
        )

    def init_param(self, key, dtype=None):
        return self._sample_noise(key, self.shape, dtype=dtype)

    def _sample_noise(self, key, shape, dtype=None):
        return jax.random.normal(key, shape, dtype) * self.std

    @staticmethod
    @jax.jit
    def get_acceptance(key, energy, new_energy):
        (
            key_accept,
        ) = jax.random.split(key, 1)
        delta_energy = new_energy - energy
        acceptance_probability = jnp.clip(jnp.exp(-delta_energy), max=1)
        accept = jax.random.bernoulli(key_accept, acceptance_probability).astype(jnp.bool)
        return accept

    @staticmethod
    @jax.jit
    def parallel_tempering_swap_probability(
        energy_hot, inverse_temperature_hot,
        energy_cold, inverse_temperature_cold,
    ):
        delta_energy = energy_cold - energy_hot
        delta_inverse_temperature = inverse_temperature_cold - inverse_temperature_hot
        swap_probability = jnp.clip(jnp.exp(delta_inverse_temperature * delta_energy), max=1)
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
    inverse_temperature : jnp.Array
        One-dimensional array of inverse temperatures :math:`\beta_k = 1/T_k`.
    *args_kernel_propose : jnp.Array, default None
        One-dimensional arrays of matching the shape of ``inverse_temperature``
        to be passed to the prose function of the kernel.
    """
    def __init__(
        self,
        kernel, potential_fn,
        inverse_temperature,
        *args_kernel_propose,
    ):
        self.kernel = kernel
        self.potential_fn = potential_fn
        self.inverse_temperature = inverse_temperature
        self.chain_count = jnp.size(inverse_temperature)
        for i, v in enumerate(args_kernel_propose):
            if (shape := v.shape) not in (tuple(), self.inverse_temperature.shape):
                raise ValueError(f"Additional argument {i} to be passed to the propose function of kernel {type(kernel).__name__} must be a scalar (shape ()) or have shape {shape}.")
        self.init_param = jax.jit(jax.vmap(jax.vmap(kernel.init_param)))
        self.potential_fn_vmap = jax.jit(jax.vmap(potential_fn))
        self.args_kernel_propose = args_kernel_propose
        self.update_chain_jit = jax.jit(jax.vmap(
            self.update_chain,
            in_axes=(0, 0, 0, 0, *([0] * len(args_kernel_propose))),
        ))

    def run(
        self,
        key,
        iters,
        param=None,
        *,
        batch_size=1,
    ):
        r"""
        Parallel Tempering Markov Chain Monte Carlo.

        Args
        ----
        key : jax.random.PRNGKey
            Pseudo-random number generator key.
        iters : int
            Number of sample updates and chain swaps (replica exchanges).

        Keyword Args
        ------------
        param : jnp.Array, optional
            The initial param to start running from.
        batch_size : int, optional
            The batch size of params to sample.

        Returns
        -------
        jnp.Array
            Final sampled params from all chains. The params array shape is (chain_count, batch_size, ...).
        """
        (
            key_initial_sample,
            key_loop,
        ) = jax.random.split(key, 2)
        if param is None:
            param = self.init_param(
                jax.random.split(key_initial_sample, (self.chain_count, batch_size)),
            )
        energy_with_unit_temperature = self.potential_fn_vmap(param)
        for _ in range(iters):
            (
                key_update_chains,
                key_swap_chains,
                key_loop,
            ) = jax.random.split(key_loop, 3)
            param, energy_with_unit_temperature = self.update_chain_jit(
                jax.random.split(key_update_chains, self.chain_count),
                param, energy_with_unit_temperature,
                self.inverse_temperature,
                *self.args_kernel_propose,
            )
            for k in range(self.chain_count - 1):
                (
                    key_swap,
                    key_swap_chains,
                ) = jax.random.split(key_swap_chains)
                param, energy_with_unit_temperature = self.swap_chain(key_swap, param, energy_with_unit_temperature, k)
        return param

    @partial(jax.jit, static_argnames=['self'])
    def update_chain(self, key, param, energy_with_unit_temperature, inverse_temperature, *args_kernel_propose):
        (
            key_proposer,
            key_accept,
        ) = jax.random.split(key, 2)
        new_param = self.kernel.propose(key_proposer, param, *args_kernel_propose)
        new_energy_with_unit_temperature = self.potential_fn(new_param)
        accept = self.kernel.get_acceptance(
            key_accept,
            inverse_temperature * energy_with_unit_temperature,
            inverse_temperature * new_energy_with_unit_temperature,
        )
        updated_energy_with_unit_temperature = jnp.where(accept, new_energy_with_unit_temperature, energy_with_unit_temperature)
        accept = accept.reshape(*accept.shape, *[1 for _ in range(param.ndim - accept.ndim)])
        updated_param = jnp.where(accept, new_param, param)
        return updated_param, updated_energy_with_unit_temperature

    @partial(jax.jit, static_argnames=['self'])
    def swap_chain(self, key, param, energy_with_unit_temperature, k):
        swap_probability = self.kernel.parallel_tempering_swap_probability(
            energy_hot=energy_with_unit_temperature[k], inverse_temperature_hot=self.inverse_temperature[k],
            energy_cold=energy_with_unit_temperature[k + 1], inverse_temperature_cold=self.inverse_temperature[k + 1],
        )
        key_chain_swap, key = jax.random.split(key)
        swap = jax.random.bernoulli(key_chain_swap, swap_probability).astype(jnp.bool)
        energy_temp_1_kp1 = jnp.where(swap, energy_with_unit_temperature[k], energy_with_unit_temperature[k + 1])
        energy_temp_1_k = jnp.where(swap, energy_with_unit_temperature[k + 1], energy_with_unit_temperature[k])
        energy_with_unit_temperature = energy_with_unit_temperature.at[k].set(energy_temp_1_k)
        energy_with_unit_temperature = energy_with_unit_temperature.at[k + 1].set(energy_temp_1_kp1)
        swap = swap.reshape(*swap.shape, *[1 for _ in range(param.ndim - 1 - swap.ndim)])
        param_kp1 = jnp.where(swap, param[k], param[k + 1])
        param_k = jnp.where(swap, param[k + 1], param[k])
        param = param.at[k].set(param_k)
        param = param.at[k + 1].set(param_kp1)
        return param, energy_with_unit_temperature
