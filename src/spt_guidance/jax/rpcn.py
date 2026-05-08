import enum
from functools import partial

import jax
import jax.random
import jax.numpy as jnp

import spt_guidance.jax.spt as spt


class IntermediateReward(str, enum.Enum):
    EXPECTED_SAMPLE = 'expected_sample'
    SUB_ENSEMBLE_PUSHFORWARD = 'pushforward'


class KernelPreconditionedCrankNicolson(spt.Kernel):
    r"""
    Preconditioned Crank-Nicolson kernel.

    Proposals :math:`X_{n + 1}'` are sampled as

    .. math::

        X_{n + 1}' \sim \mathcal{\sqrt{1 - \beta^2}X_n,\beta^2 I}

    where :math:`\beta \in (0, 1)` is the step size.
    The acceptance probability is

    .. math::

        \alpha(X, X') = \min(1, \exp(\Delta E(X, X')))

    where :math:`\Delta E = \phi(X') - \phi(X)` is the difference between the
    reward of the proposal and the current state.

    Args
    ----
    shape : tuple[int]
        Shape the param
    """
    @partial(jax.jit, static_argnames=['self'])
    def propose(self, key, param, step_size, std, drift=None):
        r"""
        step_size : scalar, default None
            Kernel step size :math:`\beta`
        std : scalar
            The diagonal element of the covariance.
        """
        if drift is None:
            drift = jnp.sqrt(1 - jnp.square(step_size))
        return (
            drift * param
            + step_size * self._sample_noise(key, param.shape, std, dtype=param.dtype)
        )

    def init_param(self, key, std, dtype=None):
        return self._sample_noise(key, self.shape, std, dtype=dtype)

    def _sample_noise(self, key, shape, std, dtype=None):
        return jax.random.normal(key, shape, dtype) * std

    @staticmethod
    @jax.jit
    def get_acceptance(key, reward, new_reward):
        (
            key_accept,
        ) = jax.random.split(key, 1)
        delta_reward = new_reward - reward
        acceptance_probability = jnp.clip(jnp.exp(delta_reward), max=1)
        accept = jax.random.bernoulli(key_accept, acceptance_probability).astype(jnp.bool)
        return accept

    @staticmethod
    @jax.jit
    def parallel_tempering_swap_probability(
        reward_hot, inverse_temperature_hot,
        reward_cold, inverse_temperature_cold,
    ):
        delta_reward = reward_cold - reward_hot
        delta_inverse_temperature = inverse_temperature_cold - inverse_temperature_hot
        swap_probability = jnp.clip(jnp.exp(-delta_inverse_temperature * delta_reward), max=1)
        return swap_probability


class ResidualMCMCSteering:
    r"""
    Residual MCMC

    Args
    ----
    kernel : Kernel
        Kernel that updates the params of each chain.
    *args_kernel_propose : jnp.Array, default None
        One-dimensional arrays of matching the shape of ``inverse_temperature``
        to be passed to the prose function of the kernel.

    Keyword Args
    ------------
    inverse_temperature : jnp.Array
        One-dimensional array of inverse temperatures :math:`\beta_k = 1/T_k`.
    reward_fn : Callable[param: jnp.Array[batch_size, ...]]
        Computes the reward of a param at the end time, which is to be maximized.
    expected_next_fn : Callable[t0: scalar, t1: scalar, param: jnp.Array[batch_size, ...]]
        Returns the expected param at time ``t1`` given the param at ``t0``.
    std_fn : Callable[t0: scalar, t1: scalar = None]
        If ``t1`` is ``None``, returns the standard deviation of all dimensions of the param at time ``t0``.
        Otherwise, returns the standard deviation of all dimensions of the param at time ``t1`` given the param at ``t0``.
    intermediate_reward : IntermediateReward
        One of ``IntermediateReward`` or its string value. Specificies the
        intermediate reward calculation to use.
    kwargs_intermediate_reward : dict
        Additional keyword arguments for building the intermediate reward function.

        - If ``intermediate_reward == IntermediateReward.EXPECTED_SAMPLE``,
            - expected_sample_fn : Callable[key: jax.random.PRNGKey, t0: scalar, param: jnp.Array[batch_size, ...]]
                Returns the expected param at the end time given the param at ``t0``.
        - If ``intermediate_reward == IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD``
            - sub_ensemble_size : int
                The number of particles to sample starting from the param at ``t0``.
            - pushforward_fn : Callable[key: jax.random.PRNGKey, t0: scalar, param: jnp.Array[batch_size, ...]]
                Returns a param at the end time given a param at ``t0``.
    """
    def __init__(
        self,
        kernel,
        *args_kernel_propose,
        inverse_temperature,
        reward_fn,
        expected_next_fn,
        std_fn,
        intermediate_reward,
        kwargs_intermediate_reward=None,
    ):
        try:
            intermediate_reward = IntermediateReward(intermediate_reward)
        except ValueError:
            raise ValueError(f'intermediate_reward must be one of {[e.value for e in IntermediateReward]}, not {intermediate_reward!r}.')
        self.kwargs_intermediate_reward = kwargs_intermediate_reward or {}

        self.kernel = kernel
        self.args_kernel_propose = args_kernel_propose
        self.init_param = jax.jit(jax.vmap(jax.vmap(
            kernel.init_param,
            in_axes=(
                0,     # key
                None,  # std
            ),
        ), in_axes=(0, None)))
        self.reward_fn = reward_fn
        self.intermediate_reward_fn = jax.jit(self.get_intermediate_reward_fn(intermediate_reward, reward_fn))
        self.intermediate_reward_fn_vmap = jax.jit(jax.vmap(
            self.intermediate_reward_fn,
            in_axes=(
                0,     # key
                None,  # t0
                0,     # param batch
            ),
        ))
        self.expected_next_fn = jax.jit(jax.vmap(expected_next_fn, in_axes=(None, None, 0)))
        self.std_fn = std_fn
        self.inverse_temperature = inverse_temperature
        self.chain_count = jnp.size(inverse_temperature)
        self.update_chain_jit = jax.vmap(
            self.update_chain,
            in_axes=(0, 0, 0, None, 0, 0, None, *([0] * len(args_kernel_propose))),
        )

    def run(
        self,
        key,
        times,
        *,
        pcn_should_update_fn=lambda *_: True,
        pcn_should_replica_exchange_fn=lambda *_: True,
        pcn_update_count=1,
        batch_size=1,
    ):
        """
        Args
        ----
        key : jax.random.PRNGKey
            Pseudo-random number generator key.
        times : jnp.Array
            One-dimensional array of times to filter over.
        stds : jnp.Array
            One-dimensional array of standard deviations.
        ensemble_size : int
            The number of particles (i.e., chains).

        Keyword Args
        ------------
        pcn_should_update_fn : Callable[time_step: int, t0: scalar, t1: scalar, reward: torch.Tensor[chain_count, batch_size]]
            Returns boolean indicating whether the pCN update routine should run.
        pcn_should_replica_exchange_fn : Callable[time_step: int, t0: scalar, t1: scalar, pcn_step: int, reward: torch.Tensor[chain_count, batch_size]]
            Returns boolean indicating whether to do replica exchange.
        pcn_update_count : int
            Number of sample updates and chain swaps (replica exchanges).
        batch_size : int, default 1
            The batch size of params to sample.

        Returns
        -------
        jnp.Array
            Final sampled params from all chains. The params array shape is (chain_count, batch_size, ...).
        """
        (
            key_initial_sample,
            key_reward,
            key_loop,
        ) = jax.random.split(key, 3)
        batch = self.init_param(
            jax.random.split(key_initial_sample, (self.chain_count, batch_size)),
            self.std_fn(times[0]),
        )
        del key_initial_sample
        reward_with_unit_temperature = self.intermediate_reward_fn_vmap(jax.random.split(key_reward, self.chain_count), times[0], batch)
        del key_reward
        for time_step, (t0, t1) in enumerate(zip(times, times[1:])):
            std = self.std_fn(t0, t1=t1)
            (
                key_sample_noise,
                key_reward,
                key_pcn,
                key_loop,
            ) = jax.random.split(key_loop, 4)
            sample_noise = self.init_param(
                jax.random.split(key_sample_noise, (self.chain_count, batch_size)),
                std,
            )
            expected_next = self.expected_next_fn(t0, t1, batch)
            reward_with_unit_temperature = self.intermediate_reward_fn_vmap(
                jax.random.split(key_reward, self.chain_count), t1, expected_next + sample_noise
            )
            if pcn_should_update_fn(time_step, t0, t1, reward_with_unit_temperature):
                for pcn_update in range(pcn_update_count):
                    (
                        key_update_chains,
                        key_swap_chains,
                        key_pcn,
                    ) = jax.random.split(key_pcn, 3)
                    sample_noise, reward_with_unit_temperature = self.update_chain_jit(
                        jax.random.split(key_update_chains, self.chain_count),
                        expected_next,
                        sample_noise,
                        lambda key_reward, exp_next, noise: self.intermediate_reward_fn(
                            key_reward,
                            t1,
                            exp_next + noise,
                        ),
                        reward_with_unit_temperature,
                        self.inverse_temperature,
                        std,
                        *self.args_kernel_propose,
                    )
                    if pcn_should_replica_exchange_fn(time_step, t0, t1, pcn_update, reward_with_unit_temperature):
                        for k in range(self.chain_count - 1):
                            (
                                key_swap,
                                key_swap_chains,
                            ) = jax.random.split(key_swap_chains)
                            reward_with_unit_temperature_before_swap = reward_with_unit_temperature
                            sample_noise, reward_with_unit_temperature = self.swap_chain(key_swap, sample_noise, reward_with_unit_temperature_before_swap, k)
                            # use same key to ensure same swap
                            expected_next, reward_with_unit_temperature2 = self.swap_chain(key_swap, expected_next, reward_with_unit_temperature_before_swap, k)
                            # assert jnp.all(reward_with_unit_temperature == reward_with_unit_temperature2)
            batch = expected_next + sample_noise
        key_reward, _ = jax.random.split(key_pcn)
        return batch

    def update_chain(self, key, expected_next, param, reward_fn, reward_with_unit_temperature, inverse_temperature, std, *args_kernel_propose):
        (
            key_proposer,
            key_reward,
            key_accept,
        ) = jax.random.split(key, 3)
        new_param = self.kernel.propose(key_proposer, param, *args_kernel_propose, std)
        new_reward_with_unit_temperature = reward_fn(key_reward, expected_next, new_param)
        return self._apply_update(key_accept, inverse_temperature, param, new_param, reward_with_unit_temperature, new_reward_with_unit_temperature)

    @partial(jax.jit, static_argnames=['self'])
    def _apply_update(self, key, inverse_temperature, param, new_param, reward_with_unit_temperature, new_reward_with_unit_temperature):
        accept = self.kernel.get_acceptance(
            key,
            inverse_temperature * reward_with_unit_temperature,
            inverse_temperature * new_reward_with_unit_temperature,
        )
        updated_reward_with_unit_temperature = jnp.where(accept, new_reward_with_unit_temperature, reward_with_unit_temperature)
        accept = accept.reshape(*accept.shape, *[1 for _ in range(param.ndim - accept.ndim)])
        updated_param = jnp.where(accept, new_param, param)
        return updated_param, updated_reward_with_unit_temperature

    @partial(jax.jit, static_argnames=['self'])
    def swap_chain(self, key, param, reward_with_unit_temperature, k):
        swap_probability = self.kernel.parallel_tempering_swap_probability(
            reward_hot=reward_with_unit_temperature[k], inverse_temperature_hot=self.inverse_temperature[k],
            reward_cold=reward_with_unit_temperature[k + 1], inverse_temperature_cold=self.inverse_temperature[k + 1],
        )
        key_chain_swap, _ = jax.random.split(key)
        swap = jax.random.bernoulli(key_chain_swap, swap_probability).astype(jnp.bool)
        reward_temp_1_kp1 = jnp.where(swap, reward_with_unit_temperature[k], reward_with_unit_temperature[k + 1])
        reward_temp_1_k = jnp.where(swap, reward_with_unit_temperature[k + 1], reward_with_unit_temperature[k])
        reward_with_unit_temperature = reward_with_unit_temperature.at[k].set(reward_temp_1_k)
        reward_with_unit_temperature = reward_with_unit_temperature.at[k + 1].set(reward_temp_1_kp1)
        swap = swap.reshape(*swap.shape, *[1 for _ in range(param.ndim - 1 - swap.ndim)])
        param_kp1 = jnp.where(swap, param[k], param[k + 1])
        param_k = jnp.where(swap, param[k + 1], param[k])
        param = param.at[k].set(param_k)
        param = param.at[k + 1].set(param_kp1)
        return param, reward_with_unit_temperature

    def get_intermediate_reward_fn(self, intermediate_reward, reward_fn):
        if intermediate_reward is IntermediateReward.EXPECTED_SAMPLE:
            expected_sample_fn = self.kwargs_intermediate_reward.get('expected_sample_fn')
            if expected_sample_fn is None:
                raise ValueError(f'With intermediate_reward={intermediate_reward.value}, the keyword argument expected_sample_fn must be passed.')
            def intermediate_reward_fn(key, t0, batch):
                return reward_fn(expected_sample_fn(key, t0, batch))
        elif intermediate_reward is IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD:
            sub_ensemble_size = self.kwargs_intermediate_reward.get('sub_ensemble_size')
            pushforward_fn = self.kwargs_intermediate_reward.get('pushforward_fn')
            if pushforward_fn is None:
                raise ValueError(f'With intermediate_reward={intermediate_reward.value}, the keyword argument pushforward_fn must be passed.')
            if sub_ensemble_size is None:
                raise ValueError(f'With intermediate_reward={intermediate_reward.value}, the keyword argument sub_ensemble_size must be passed.')
            if sub_ensemble_size <= 0:
                raise ValueError(f'The keyword argument sub_ensemble_size must be greater than 0, not {sub_ensemble_size}.')
            def intermediate_reward_fn(key, t0, batch):
                key_sample_sub_ensemble = jax.random.split(key, sub_ensemble_size)
                batch = jnp.broadcast_to(batch[None], (sub_ensemble_size, *batch.shape))
                def get_sub_ensemble_member_reward(key, y0):
                    return reward_fn(pushforward_fn(key, t0, y0))
                sub_ensemble_member_reward = jax.vmap(get_sub_ensemble_member_reward)(key_sample_sub_ensemble, batch)
                return jnp.log(jnp.exp(sub_ensemble_member_reward).mean(axis=0))
        else:
            raise ValueError(f'Unknown intermediate reward: {intermediate_reward}')
        return intermediate_reward_fn
