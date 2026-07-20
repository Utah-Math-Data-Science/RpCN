import enum

import jax
import jax.random
import jax.scipy.special
import jax.numpy as jnp


class Potential(str, enum.Enum):
    DIFFERENCE = 'difference'
    MAX = 'max'
    SUM = 'sum'


class IntermediateReward(str, enum.Enum):
    EXPECTED_SAMPLE = 'expected_sample'
    SUB_ENSEMBLE_PUSHFORWARD = 'pushforward'


class ResampleFn:
    """
    Collection of resampling functions. Note that ``do_resample`` is intended
    to be used by other resampling functions to actually perform the resampling.
    Some of these functions also may take more arguments than will be passed by
    the ``FeynmanKac`` class, so use ``functools.partial``.
    """
    @staticmethod
    def do_resample(key, ensemble_size, probabilities):
        """
        Utility function to sample ensemble ancestor indices.

        Args
        ----
        key : jax.random.PRNGKey
            Pseudo-random number generator key.
        ensemble_size : int
            The number of ensemble members.
        probabilities : jnp.Array[ensemble_size]
            The probability of sampling each ensemble member.

        Returns
        -------
        jnp.Array[ensemble_size]
            The sampled ancestor indices.
        """
        return jax.random.choice(key, ensemble_size, shape=(ensemble_size,), p=probabilities)

    @staticmethod
    def adaptive_resample(
        key,
        time_step,
        ensemble_size,
        log_weights,
        *,
        zero_log_potential_when_skipping=False,
    ):
        r"""
        Adaptive resampling of the ensemble. Resamples when the effective
        sample size (ESS) is less than one half of the ensemble size :math:`K`.
        If ``zero_log_potential_when_skipping`` is true, a skipped resampling step
        also requests that the current potential be replaced by one.

        .. math::

            \mathrm{ESS} = 1 / \sum_{k = 1}^K \hat{G}_k^2

        where :math:`\hat{G}_k` = G_k / \sum_{k = 1}^K G_k` and :math:`G_k`
        is the potential of ensemble member :math:`k`.
        """
        probabilities = jax.nn.softmax(log_weights)
        effective_sample_size = 1 / jnp.square(probabilities).sum()
        should_resample = effective_sample_size < ensemble_size / 2
        ancestor_indices = jnp.where(
            should_resample,
            ResampleFn.do_resample(key, ensemble_size, probabilities),
            jnp.arange(ensemble_size),
        )
        use_zero_log_potential = jnp.logical_and(zero_log_potential_when_skipping, ~should_resample)
        return ancestor_indices, should_resample, use_zero_log_potential


class SelectFn:
    """Collection of functions for selecting one final particle per batch."""

    @staticmethod
    def _select(particles, indices):
        batch_indices = jnp.arange(particles.shape[1])
        return particles[indices, batch_indices]

    @staticmethod
    def highest_weight(particles, log_weights):
        """
        Select the particle with the highest final log-weight.

        Args
        ----
        particles : jnp.Array[ensemble_size, batch_size, ...]
            The final particles.
        log_weights : jnp.Array[ensemble_size, batch_size]
            The final unnormalized log-weights.

        Returns
        -------
        jnp.Array[batch_size, ...]
            The selected particles.
        """
        return SelectFn._select(particles, jnp.argmax(log_weights, axis=0))

    @staticmethod
    def weighted_sample(key, particles, log_weights):
        """
        Sample one particle per batch from the normalized final weights.

        Args
        ----
        key : jax.random.PRNGKey
            Pseudo-random number generator key.
        particles : jnp.Array[ensemble_size, batch_size, ...]
            The final particles.
        log_weights : jnp.Array[ensemble_size, batch_size]
            The final unnormalized log-weights.

        Returns
        -------
        jnp.Array[batch_size, ...]
            The sampled particles.
        """
        indices = jax.random.categorical(key, log_weights, axis=0)
        return SelectFn._select(particles, indices)

    @staticmethod
    def highest_reward(particles, reward_fn):
        """
        Select the particle with the highest terminal reward.

        Args
        ----
        particles : jnp.Array[ensemble_size, batch_size, ...]
            The final particles.
        reward_fn : Callable[param: jnp.Array[batch_size, ...]]
            Computes one terminal reward per batch member.

        Returns
        -------
        jnp.Array[batch_size, ...]
            The selected particles.
        """
        rewards = jax.vmap(reward_fn)(particles)
        return SelectFn._select(particles, jnp.argmax(rewards, axis=0))


class FeynmanKac:
    """
    Feynman-Kac Steering.

    Keyword Args
    ------------
    init_param_fn : Callable[key: jax.random.PRNGKey]
        Returns single initial param, not a batch.
    reward_fn : Callable[param: jnp.Array[batch_size, ...]]
        Computes the reward of a param at the end time, which is to be maximized.
    propose_fn : Callable[key: jax.random.PRNGKey, t0: scalar, t1: scalar, param: jnp.Array[batch_size, ...]]
        Returns a proposal param for ``t1`` given the current param at ``t0``.
    resample_fn : Callable[key: jax.random.PRNGKey, time_step: int, ensemble_size: int, log_weight: jnp.Array[ensemble_size]]
        Returns the ancestor index of each new ensemble member, whether
        resampling occurred, and whether to replace the current potential by
        one when resampling is skipped. The third return value must be false
        when resampling occurs.
    potential : Potential
        One of ``Potential`` or its string value. Specificies the particle
        potential calculation to use.
    intermediate_reward : IntermediateReward
        One of ``IntermediateReward`` or its string value. Specificies the
        intermediate reward calculation to use.
    inverse_temperature : scalar or one-dimensional jnp.Array
        The inverse temperature (i.e., tilt) of the target param distribution.
        Passing a temperature ladder runs one FK ensemble per temperature and
        enables replica exchange between adjacent temperatures.
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
        *,
        init_param_fn,
        reward_fn,
        propose_fn,
        resample_fn,
        potential,
        intermediate_reward,
        inverse_temperature,
        kwargs_intermediate_reward=None,
    ):
        try:
            potential = Potential(potential)
        except ValueError:
            raise ValueError(f'potential must be one of {[e.value for e in Potential]}, not {potential!r}.')
        try:
            intermediate_reward = IntermediateReward(intermediate_reward)
        except ValueError:
            raise ValueError(f'intermediate_reward must be one of {[e.value for e in IntermediateReward]}, not {intermediate_reward!r}.')
        inverse_temperature = jnp.asarray(inverse_temperature)
        if inverse_temperature.ndim > 1:
            raise ValueError('inverse_temperature must be a scalar or one-dimensional array.')
        if inverse_temperature.size == 0:
            raise ValueError('inverse_temperature must contain at least one value.')
        self.kwargs_intermediate_reward = kwargs_intermediate_reward or {}

        self.init_param_fn = jax.jit(jax.vmap(jax.vmap(jax.vmap(init_param_fn))))
        intermediate_reward_fn = self.get_intermediate_reward_fn(intermediate_reward, reward_fn)
        self.intermediate_reward_fn = jax.jit(jax.vmap(jax.vmap(
            intermediate_reward_fn,
            in_axes=(
                0,     # key
                None,  # t0
                0,     # param batch
            ),
        ), in_axes=(0, None, 0)))
        self.propose_fn = jax.jit(jax.vmap(jax.vmap(propose_fn, in_axes=(
            0,     # key
            None,  # t0
            None,  # t1
            0,     # param batch
        )), in_axes=(0, None, None, 0)))
        resample_fn = jax.vmap(
            resample_fn,
            in_axes=(0, None, None, 1),
            out_axes=(1, 0, 0),
        )
        self.resample_fn = jax.jit(jax.vmap(
            resample_fn,
            in_axes=(0, None, None, 0),
            out_axes=(0, 0, 0),
        ), static_argnums=2)
        self.log_potential_beta_one_fn = jax.jit(jax.vmap(jax.vmap(
            self.get_log_potential_fn(potential, 1.),
        )))
        self.potential = potential
        self.scalar_inverse_temperature = inverse_temperature.ndim == 0
        self.inverse_temperature = inverse_temperature.reshape(-1)
        self.temperature_count = self.inverse_temperature.size

    def run(
        self,
        key,
        times,
        ensemble_size,
        *,
        batch_size=1,
    ):
        """
        Args
        ----
        key : jax.random.PRNGKey
            Pseudo-random number generator key.
        times : jnp.Array
            One-dimensional array of times to filter over.
        ensemble_size : int
            The number of particles (i.e., chains).

        Keyword Args
        ------------
        batch_size : int, default 1
            The batch size of params to sample.

        Returns
        -------
        tuple[jnp.Array, jnp.Array]
            For a scalar inverse temperature, the final particles have shape
            ``(ensemble_size, batch_size, ...)`` and their unnormalized
            log-weights have shape ``(ensemble_size, batch_size)``. For a
            temperature ladder, both outputs have an additional leading
            ``temperature_count`` dimension.
        """
        (
            key_init_ensemble,
            key_particle_resampling,
            key_reward_fn,
            key_propose_next_intermediate,
            key_parallel_tempering,
        ) = jax.random.split(key, 5)
        batch = self.init_param_fn(jax.random.split(
            key_init_ensemble, (self.temperature_count, ensemble_size, batch_size),
        ))
        key_reward, key_reward_fn = jax.random.split(key_reward_fn)
        reward = self.intermediate_reward_fn(jax.random.split(
            key_reward, (self.temperature_count, ensemble_size),
        ), times[0], batch)
        inverse_temperature = self.inverse_temperature.astype(reward.dtype).reshape(-1, 1, 1)
        if self.potential in (Potential.DIFFERENCE, Potential.MAX):
            aggregated_reward = reward
        elif self.potential is Potential.SUM:
            aggregated_reward = jnp.zeros_like(reward)
        else:
            raise ValueError(f'Aggregated reward initialization not implemented for {self.potential!r}.')
        if self.potential is Potential.DIFFERENCE:
            log_potential_beta_one = jnp.zeros_like(reward)
        else:
            log_potential_beta_one, aggregated_reward = self.log_potential_beta_one_fn(reward, aggregated_reward)
        aggregated_log_potential_beta_one = jnp.zeros_like(log_potential_beta_one)
        log_weight_beta_one = jnp.zeros_like(log_potential_beta_one)
        batch_indices = jnp.arange(batch_size)
        temperature_indices = jnp.arange(self.temperature_count)[:, None, None]
        for time_step, (t0, t1) in enumerate(zip(times, times[1:])):
            key_resample, key_particle_resampling = jax.random.split(key_particle_resampling)
            ancestor_indices, did_resample, use_zero_log_potential = self.resample_fn(
                jax.random.split(key_resample, (self.temperature_count, batch_size)),
                time_step,
                ensemble_size,
                inverse_temperature * (log_weight_beta_one + log_potential_beta_one),
            )
            use_zero_log_potential = jnp.logical_and(use_zero_log_potential, ~did_resample)
            log_potential_beta_one = jnp.where(use_zero_log_potential[:, None], 0., log_potential_beta_one)
            aggregated_log_potential_beta_one = aggregated_log_potential_beta_one + log_potential_beta_one
            log_weight_beta_one = log_weight_beta_one + log_potential_beta_one
            batch = batch[temperature_indices, ancestor_indices, batch_indices]
            aggregated_reward = aggregated_reward[temperature_indices, ancestor_indices, batch_indices]
            aggregated_log_potential_beta_one = aggregated_log_potential_beta_one[
                temperature_indices, ancestor_indices, batch_indices,
            ]
            log_weight_beta_one = log_weight_beta_one[temperature_indices, ancestor_indices, batch_indices]
            log_weight_beta_one = jnp.where(did_resample[:, None], 0., log_weight_beta_one)
            key_propose, key_propose_next_intermediate = jax.random.split(key_propose_next_intermediate)
            batch = self.propose_fn(jax.random.split(
                key_propose, (self.temperature_count, ensemble_size),
            ), t0, t1, batch)
            key_reward, key_reward_fn = jax.random.split(key_reward_fn)
            reward = self.intermediate_reward_fn(jax.random.split(
                key_reward, (self.temperature_count, ensemble_size),
            ), t1, batch)
            for temperature in range(self.temperature_count - 1):
                key_permutation, key_swap, key_parallel_tempering = jax.random.split(key_parallel_tempering, 3)
                permutation = jax.vmap(
                    lambda permutation_key: jax.random.permutation(permutation_key, ensemble_size),
                )(jax.random.split(key_permutation, batch_size)).T
                batch = batch.at[temperature].set(batch[temperature][permutation, batch_indices])
                reward = reward.at[temperature].set(reward[temperature][permutation, batch_indices])
                aggregated_reward = aggregated_reward.at[temperature].set(
                    aggregated_reward[temperature][permutation, batch_indices],
                )
                aggregated_log_potential_beta_one = aggregated_log_potential_beta_one.at[temperature].set(
                    aggregated_log_potential_beta_one[temperature][permutation, batch_indices],
                )
                log_weight_beta_one = log_weight_beta_one.at[temperature].set(
                    log_weight_beta_one[temperature][permutation, batch_indices],
                )
                swap_probability = self.parallel_tempering_swap_probability(
                    reward_hot=reward[temperature],
                    inverse_temperature_hot=inverse_temperature[temperature],
                    reward_cold=reward[temperature + 1],
                    inverse_temperature_cold=inverse_temperature[temperature + 1],
                )
                swap = jax.random.bernoulli(key_swap, swap_probability)
                batch = self.swap_temperature(swap, batch, temperature)
                reward = self.swap_temperature(swap, reward, temperature)
                aggregated_reward = self.swap_temperature(swap, aggregated_reward, temperature)
                aggregated_log_potential_beta_one = self.swap_temperature(
                    swap, aggregated_log_potential_beta_one, temperature,
                )
                log_weight_beta_one = self.swap_temperature(swap, log_weight_beta_one, temperature)
            if t1 == times[-1]:
                log_potential_beta_one = reward
            else:
                log_potential_beta_one, aggregated_reward = self.log_potential_beta_one_fn(
                    reward, aggregated_reward,
                )
        log_potential_beta_one = log_potential_beta_one - aggregated_log_potential_beta_one
        log_weight = inverse_temperature * (log_weight_beta_one + log_potential_beta_one)
        if self.scalar_inverse_temperature:
            return batch[0], log_weight[0]
        return batch, log_weight

    @staticmethod
    def parallel_tempering_swap_probability(
        reward_hot, inverse_temperature_hot,
        reward_cold, inverse_temperature_cold,
    ):
        delta_reward = reward_cold - reward_hot
        delta_inverse_temperature = inverse_temperature_cold - inverse_temperature_hot
        return jnp.clip(jnp.exp(-delta_inverse_temperature * delta_reward), max=1)

    @staticmethod
    def swap_temperature(swap, value, temperature):
        swap = swap.reshape(*swap.shape, *[1 for _ in range(value.ndim - 1 - swap.ndim)])
        value_hot = jnp.where(swap, value[temperature + 1], value[temperature])
        value_cold = jnp.where(swap, value[temperature], value[temperature + 1])
        value = value.at[temperature].set(value_hot)
        value = value.at[temperature + 1].set(value_cold)
        return value

    def get_log_potential_fn(self, potential, inverse_temperature):
        if potential is Potential.DIFFERENCE:
            def log_potential_fn(new_reward, aggregated_reward):
                new_log_potential = inverse_temperature * (new_reward - aggregated_reward)
                new_aggregated_reward = new_reward
                return new_log_potential, new_aggregated_reward
        elif potential is Potential.MAX:
            def log_potential_fn(new_reward, aggregated_reward):
                new_aggregated_reward = jnp.maximum(new_reward, aggregated_reward)
                new_log_potential = inverse_temperature * new_aggregated_reward
                return new_log_potential, new_aggregated_reward
        elif potential is Potential.SUM:
            def log_potential_fn(new_reward, aggregated_reward):
                new_aggregated_reward = new_reward + aggregated_reward
                new_log_potential = inverse_temperature * new_aggregated_reward
                return new_log_potential, new_aggregated_reward
        else:
            raise ValueError(f'Unknown Feynman-Kac steering potential: {potential}')
        return log_potential_fn

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
                return jax.scipy.special.logsumexp(sub_ensemble_member_reward, axis=0) - jnp.log(sub_ensemble_size)
        else:
            raise ValueError(f'Unknown Feynman-Kac steering intermediate reward: {intermediate_reward}')
        return intermediate_reward_fn
