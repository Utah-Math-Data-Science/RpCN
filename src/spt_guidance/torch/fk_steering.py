import enum

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
    the ``FeynmannKac`` class, so use ``functools.partial``.
    """
    @staticmethod
    def do_resample(ensemble, probabilities):
        """
        Utility function to perform the resampling of the ensemble.

        Args
        ----
        ensemble : torch.Tensor[batch_size, ...]
            The ensemble.
        probabilities : torch.Tensor[batch_size, 1]
            The probability of sampling each ensemble member.

        Returns
        -------
        torch.Tensor[batch_size, ...]
            The new ensemble members after resampling.
        """
        indices = torch.multinomial(probabilities, ensemble.shape[0], replacement=True)
        return ensemble[indices]

    @staticmethod
    def adaptive_resample(time_step, ensemble, potentials):
        r"""
        Adaptive resampling of the ensemble. Resamples when the effective
        sample size (ESS) is less than one half of the ensemble size :math:`K`.

        .. math::

            \mathrm{ESS} = 1 / \sum_{k = 1}^K \hat{G}_k^2

        where :math:`\hat{G}_k` = G_k / \sum_{k = 1}^K G_k` and :math:`G_k`
        is the potential of ensemble member :math:`k`.
        """
        probabilities = potentials / potentials.sum()
        effective_sample_size = 1 / torch.square(probabilities).sum()
        return torch.where(
            effective_sample_size < ensemble.shape[0] // 2,
            ResampleFn.do_resample(ensemble, probabilities),
            ensemble,
        )


class FeynmannKac:
    """
    Feynmann-Kac Steering.

    Keyword Args
    ------------
    init_param_fn : Callable[]
        Returns single initial param, not a batch.
    reward_fn : Callable[param: torch.Tensor[batch_size, ...]]
        Computes the reward of a param at the end time, which is to be maximized.
    propose_fn : Callable[t0: scalar, t1: scalar, param: torch.Tensor[batch_size, ...]]
        Returns a proposal param for ``t1`` given the current param at ``t0``.
    resample_fn : Callable[time_step: int, param: torch.Tensor[batch_size, ...], potential: torch.Tensor[batch_size, 1]]
        Returns a new param ensemble given the ``param`` ensemble at ``time_step`` and each ensemble member ``potential``.
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
        self.resample_fn = torch.vmap(resample_fn, in_dims=(None, 1, 1), out_dims=1, randomness='different')
        self.potential_fn = torch.vmap(self.get_potential_fn(potential, inverse_temperature))
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
        torch.Tensor
            Final sampled params from all chains. The params array shape is (chain_count, batch_size, ...).
        """
        batch = self.init_param_fn(torch.zeros(1).expand(ensemble_size, batch_size))
        reward = self.intermediate_reward_fn(times[0], batch)
        if self.potential in (Potential.DIFFERENCE, Potential.MAX):
            aggregated_reward = reward
        elif self.potential is Potential.SUM:
            aggregated_reward = torch.zeros(ensemble_size, batch_size, device=batch.device)
        else:
            raise ValueError(f'Aggregated reward initialization not implemented for {self.potential!r}.')
        potential, aggregated_reward = self.potential_fn(reward, aggregated_reward)
        aggregated_potential = 1.
        for time_step, (t0, t1) in enumerate(zip(times, times[1:])):
            aggregated_potential = aggregated_potential / potential
            batch = self.resample_fn(time_step, batch, potential)
            batch = self.propose_fn(t0, t1, batch)
            reward = self.intermediate_reward_fn(t0, batch)
            if t1 == times[-1] and self.potential in (Potential.MAX, Potential.SUM):
                potential = torch.exp(self.inverse_temperature * reward)
            else:
                potential, aggregated_reward = self.potential_fn(reward, aggregated_reward)
        if self.potential in (Potential.MAX, Potential.SUM):
            final_potential = potential / aggregated_potential
        elif self.potential is Potential.DIFFERENCE:
            final_potential = potential
        else:
            raise ValueError(f'Final potential computation not implemented for {self.potential!r}.')
        batch = torch.vmap(lambda i, b: b[i.unsqueeze(0)], in_dims=(0, 1))(torch.argmax(final_potential, dim=0), batch).squeeze(1)
        return batch

    def get_potential_fn(self, potential, inverse_temperature):
        if potential is Potential.DIFFERENCE:
            def potential_fn(new_reward, aggregated_reward):
                new_potential = torch.exp(inverse_temperature * (new_reward - aggregated_reward))
                new_aggregated_reward = new_reward
                return new_potential, new_aggregated_reward
        elif potential is Potential.MAX:
            def potential_fn(new_reward, aggregated_reward):
                new_aggregated_reward = torch.maximum(new_reward, aggregated_reward)
                new_potential = torch.exp(inverse_temperature * new_aggregated_reward)
                return new_potential, new_aggregated_reward
        elif potential is Potential.SUM:
            def potential_fn(new_reward, aggregated_reward):
                new_aggregated_reward = new_reward + aggregated_reward
                new_potential = torch.exp(inverse_temperature * new_aggregated_reward)
                return new_potential, new_aggregated_reward
        else:
            raise ValueError(f'Unknown Feynmann-Kac steering potential: {potential}')
        return potential_fn

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
                sub_ensemble_member_reward = torch.vmap(get_sub_ensemble_member_reward)(batch)
                return torch.log(torch.exp(sub_ensemble_member_reward).mean(dim=0))
        else:
            raise ValueError(f'Unknown Feynmann-Kac steering intermediate reward: {intermediate_reward}')
        return intermediate_reward_fn
