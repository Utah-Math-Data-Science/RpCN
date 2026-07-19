import enum
import math

import torch


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
    def do_resample(ensemble_size, probabilities):
        """
        Utility function to sample ensemble ancestor indices.

        Args
        ----
        ensemble_size : int
            The number of ensemble members.
        probabilities : torch.Tensor[ensemble_size]
            The probability of sampling each ensemble member.

        Returns
        -------
        torch.Tensor[ensemble_size]
            The sampled ancestor indices.
        """
        return torch.multinomial(probabilities, ensemble_size, replacement=True)

    @staticmethod
    def adaptive_resample(
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
        probabilities = torch.softmax(log_weights, dim=0)
        effective_sample_size = 1 / torch.square(probabilities).sum()
        should_resample = effective_sample_size < ensemble_size / 2
        ancestor_indices = torch.where(
            should_resample,
            ResampleFn.do_resample(ensemble_size, probabilities),
            torch.arange(ensemble_size, device=log_weights.device),
        )
        use_zero_log_potential = zero_log_potential_when_skipping & ~should_resample
        return ancestor_indices, should_resample, use_zero_log_potential


class SelectFn:
    """Collection of functions for selecting one final particle per batch."""

    @staticmethod
    def _select(particles, indices):
        batch_indices = torch.arange(particles.shape[1], device=particles.device)
        return particles[indices, batch_indices]

    @staticmethod
    def highest_weight(particles, log_weights):
        """
        Select the particle with the highest final log-weight.

        Args
        ----
        particles : torch.Tensor[ensemble_size, batch_size, ...]
            The final particles.
        log_weights : torch.Tensor[ensemble_size, batch_size]
            The final unnormalized log-weights.

        Returns
        -------
        torch.Tensor[batch_size, ...]
            The selected particles.
        """
        return SelectFn._select(particles, torch.argmax(log_weights, dim=0))

    @staticmethod
    def weighted_sample(particles, log_weights):
        """
        Sample one particle per batch from the normalized final weights.

        Args
        ----
        particles : torch.Tensor[ensemble_size, batch_size, ...]
            The final particles.
        log_weights : torch.Tensor[ensemble_size, batch_size]
            The final unnormalized log-weights.

        Returns
        -------
        torch.Tensor[batch_size, ...]
            The sampled particles.
        """
        probabilities = torch.softmax(log_weights, dim=0).transpose(0, 1)
        indices = torch.multinomial(probabilities, 1).squeeze(1)
        return SelectFn._select(particles, indices)

    @staticmethod
    def highest_reward(particles, reward_fn):
        """
        Select the particle with the highest terminal reward.

        Args
        ----
        particles : torch.Tensor[ensemble_size, batch_size, ...]
            The final particles.
        reward_fn : Callable[param: torch.Tensor[batch_size, ...]]
            Computes one terminal reward per batch member.

        Returns
        -------
        torch.Tensor[batch_size, ...]
            The selected particles.
        """
        rewards = torch.vmap(reward_fn, randomness='different')(particles)
        return SelectFn._select(particles, torch.argmax(rewards, dim=0))


class FeynmanKac:
    """
    Feynman-Kac Steering.

    Keyword Args
    ------------
    init_param_fn : Callable[]
        Returns single initial param, not a batch.
    reward_fn : Callable[param: torch.Tensor[batch_size, ...]]
        Computes the reward of a param at the end time, which is to be maximized.
    propose_fn : Callable[t0: scalar, t1: scalar, param: torch.Tensor[batch_size, ...]]
        Returns a proposal param for ``t1`` given the current param at ``t0``.
    resample_fn : Callable[time_step: int, ensemble_size: int, log_weight: torch.Tensor[ensemble_size]]
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
    inverse_temperature : scalar
        The inverse temperature (i.e., tilt) of the target param distribution.
    kwargs_intermediate_reward : dict
        Additional keyword arguments for building the intermediate reward function.

        - If ``intermediate_reward == IntermediateReward.EXPECTED_SAMPLE``,
            - expected_sample_fn : Callable[t0: scalar, param: torch.Tensor[batch_size, ...]]
                Returns the expected param at the end time given the param at ``t0``.
        - If ``intermediate_reward == IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD``
            - sub_ensemble_size : int
                The number of particles to sample starting from the param at ``t0``.
            - pushforward_fn : Callable[t0: scalar, param: torch.Tensor[batch_size, ...]]
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
        self.kwargs_intermediate_reward = kwargs_intermediate_reward or {}

        self.init_param_fn = torch.vmap(torch.vmap(lambda *_: init_param_fn(), randomness='different'), randomness='different')
        self.intermediate_reward_fn = torch.vmap(
            self.get_intermediate_reward_fn(intermediate_reward, reward_fn),
            in_dims=(
                None,  # t0
                0,     # param batch
            ),
            randomness='different',
        )
        self.propose_fn = torch.vmap(propose_fn, in_dims=(
            None,  # t0
            None,  # t1
            0,     # param batch
        ), randomness='different')
        self.resample_fn = torch.vmap(
            resample_fn,
            in_dims=(None, None, 1),
            out_dims=(1, 0, 0),
            randomness='different',
        )
        self.log_potential_fn = torch.vmap(self.get_log_potential_fn(potential, inverse_temperature))
        self.potential = potential
        self.inverse_temperature = inverse_temperature

    def run(
        self,
        times,
        ensemble_size,
        *,
        batch_size=1,
    ):
        """
        Args
        ----
        times : torch.Tensor
            One-dimensional array of times to filter over.
        ensemble_size : int
            The number of particles (i.e., chains).

        Keyword Args
        ------------
        batch_size : int, default 1
            The batch size of params to sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The final particles with shape ``(ensemble_size, batch_size, ...)``
            and their unnormalized log-weights with shape
            ``(ensemble_size, batch_size)``.
        """
        batch = self.init_param_fn(torch.zeros(1).expand(ensemble_size, batch_size))
        reward = self.intermediate_reward_fn(times[0], batch)
        if self.potential in (Potential.DIFFERENCE, Potential.MAX):
            aggregated_reward = reward
        elif self.potential is Potential.SUM:
            aggregated_reward = torch.zeros_like(reward)
        else:
            raise ValueError(f'Aggregated reward initialization not implemented for {self.potential!r}.')
        if self.potential is Potential.DIFFERENCE:
            log_potential = torch.zeros_like(reward)
        else:
            log_potential, aggregated_reward = self.log_potential_fn(reward, aggregated_reward)
        aggregated_log_potential = torch.zeros_like(log_potential)
        log_weight = torch.zeros_like(log_potential)
        batch_indices = torch.arange(batch_size, device=batch.device)
        for time_step, (t0, t1) in enumerate(zip(times, times[1:])):
            ancestor_indices, did_resample, use_zero_log_potential = self.resample_fn(
                time_step, ensemble_size, log_weight + log_potential,
            )
            use_zero_log_potential = use_zero_log_potential & ~did_resample
            log_potential = torch.where(use_zero_log_potential, 0., log_potential)
            aggregated_log_potential = aggregated_log_potential + log_potential
            log_weight = log_weight + log_potential
            batch = batch[ancestor_indices, batch_indices]
            aggregated_reward = aggregated_reward[ancestor_indices, batch_indices]
            aggregated_log_potential = aggregated_log_potential[ancestor_indices, batch_indices]
            log_weight = log_weight[ancestor_indices, batch_indices]
            log_weight = torch.where(did_resample, 0., log_weight)
            batch = self.propose_fn(t0, t1, batch)
            reward = self.intermediate_reward_fn(t1, batch)
            if t1 == times[-1]:
                log_potential = self.inverse_temperature * reward
            else:
                log_potential, aggregated_reward = self.log_potential_fn(reward, aggregated_reward)
        final_log_potential = log_potential - aggregated_log_potential
        final_log_weight = log_weight + final_log_potential
        return batch, final_log_weight

    def get_log_potential_fn(self, potential, inverse_temperature):
        if potential is Potential.DIFFERENCE:
            def log_potential_fn(new_reward, aggregated_reward):
                new_log_potential = inverse_temperature * (new_reward - aggregated_reward)
                new_aggregated_reward = new_reward
                return new_log_potential, new_aggregated_reward
        elif potential is Potential.MAX:
            def log_potential_fn(new_reward, aggregated_reward):
                new_aggregated_reward = torch.maximum(new_reward, aggregated_reward)
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
            def intermediate_reward_fn(t0, batch):
                return reward_fn(expected_sample_fn(t0, batch))
        elif intermediate_reward is IntermediateReward.SUB_ENSEMBLE_PUSHFORWARD:
            sub_ensemble_size = self.kwargs_intermediate_reward.get('sub_ensemble_size')
            pushforward_fn = self.kwargs_intermediate_reward.get('pushforward_fn')
            if pushforward_fn is None:
                raise ValueError(f'With intermediate_reward={intermediate_reward.value}, the keyword argument pushforward_fn must be passed.')
            if sub_ensemble_size is None:
                raise ValueError(f'With intermediate_reward={intermediate_reward.value}, the keyword argument sub_ensemble_size must be passed.')
            if sub_ensemble_size <= 0:
                raise ValueError(f'The keyword argument sub_ensemble_size must be greater than 0, not {sub_ensemble_size}.')
            def intermediate_reward_fn(t0, batch):
                batch = torch.broadcast_to(batch[None], (sub_ensemble_size, *batch.shape))
                def get_sub_ensemble_member_reward(y0):
                    return reward_fn(pushforward_fn(t0, y0))
                sub_ensemble_member_reward = torch.vmap(
                    get_sub_ensemble_member_reward, randomness='different',
                )(batch)
                return torch.logsumexp(sub_ensemble_member_reward, dim=0) - math.log(sub_ensemble_size)
        else:
            raise ValueError(f'Unknown Feynman-Kac steering intermediate reward: {intermediate_reward}')
        return intermediate_reward_fn
