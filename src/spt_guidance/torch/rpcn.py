import enum

import torch

import spt_guidance.torch.spt as spt


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

    def propose(self, param, step_size, std, drift=None):
        r"""
        step_size : scalar, default None
            Kernel step size :math:`\beta`
        std : scalar
            The diagonal element of the covariance.
        """
        if drift is None:
            drift = torch.sqrt(1 - torch.square(step_size))
        return (
            drift * param
            + step_size * self._sample_noise(param.shape, std)
        )

    def init_param(self, std, dtype=None, device=None):
        return self._sample_noise(self.shape, std, dtype=dtype, device=device)

    def _sample_noise(self, shape, std, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return torch.randn(shape, dtype=dtype, device=device) * std

    @staticmethod
    def get_acceptance(reward, new_reward):
        delta_reward = new_reward - reward
        acceptance_probability = torch.clip(torch.exp(delta_reward), max=1)
        accept = torch.bernoulli(acceptance_probability).to(torch.bool)
        return accept

    @staticmethod
    def parallel_tempering_swap_probability(
        reward_hot, inverse_temperature_hot,
        reward_cold, inverse_temperature_cold,
    ):
        delta_reward = reward_cold - reward_hot
        delta_inverse_temperature = inverse_temperature_cold - inverse_temperature_hot
        swap_probability = torch.clip(torch.exp(-delta_inverse_temperature * delta_reward), max=1)
        return swap_probability


class ResidualMCMCSteering:
    r"""
    Residual MCMC

    Args
    ----
    kernel : Kernel
        Kernel that updates the params of each chain.
    *args_kernel_propose : torch.Tensor, default None
        One-dimensional arrays of matching the shape of ``inverse_temperature``
        to be passed to the prose function of the kernel.

    Keyword Args
    ------------
    inverse_temperature : torch.Tensor
        One-dimensional array of inverse temperatures :math:`\beta_k = 1/T_k`.
    reward_fn : Callable[param: torch.Tensor[batch_size, ...]]
        Computes the reward of a param at the end time, which is to be maximized.
    expected_next_fn : Callable[t0: scalar, t1: scalar, param: torch.Tensor[batch_size, ...]]
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
        self.init_param = torch.vmap(torch.vmap(kernel.init_param, randomness='different',), randomness='different')
        self.intermediate_reward_fn = self.get_intermediate_reward_fn(intermediate_reward, reward_fn)
        self.intermediate_reward_fn_vmap = torch.vmap(
            self.intermediate_reward_fn,
            in_dims=(
                None,  # t0
                0,     # param batch
            ),
            randomness='different',
        )
        self.expected_next_fn = torch.vmap(expected_next_fn, in_dims=(None, None, 0), randomness='different')
        self.std_fn = std_fn
        self.inverse_temperature = inverse_temperature
        self.chain_count = torch.numel(inverse_temperature)
        self.update_chain_jit = torch.vmap(
            self.update_chain,
            in_dims=(0, 0, None, 0, 0, None, *([0] * len(args_kernel_propose))),
            randomness='different',
        )

    def run(
        self,
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
        times : torch.Tensor
            One-dimensional array of times to filter over.

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
        torch.Tensor
            Final sampled params from all chains. The params array shape is (chain_count, batch_size, ...).
        """
        batch = self.init_param(self.std_fn(times[0]).expand(self.chain_count, batch_size))
        reward_with_unit_temperature = self.intermediate_reward_fn_vmap(times[0], batch)
        for time_step, (t0, t1) in enumerate(zip(times, times[1:])):
            std = self.std_fn(t0, t1=t1)
            sample_noise = self.init_param(std.expand(self.chain_count, batch_size))
            expected_next = self.expected_next_fn(t0, t1, batch)
            reward_with_unit_temperature = self.intermediate_reward_fn_vmap(t1, expected_next + sample_noise)
            if pcn_should_update_fn(time_step, t0, t1, reward_with_unit_temperature):
                for pcn_update in range(pcn_update_count):
                    sample_noise, reward_with_unit_temperature = self.update_chain_jit(
                        expected_next,
                        sample_noise,
                        lambda exp_next, noise: self.intermediate_reward_fn(
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
                            reward_with_unit_temperature_before_swap = reward_with_unit_temperature
                            swap_probability = self.kernel.parallel_tempering_swap_probability(
                                reward_hot=reward_with_unit_temperature[k], inverse_temperature_hot=self.inverse_temperature[k],
                                reward_cold=reward_with_unit_temperature[k + 1], inverse_temperature_cold=self.inverse_temperature[k + 1],
                            )
                            swap = torch.bernoulli(swap_probability).to(torch.bool)
                            sample_noise, reward_with_unit_temperature = self.swap_chain(swap, sample_noise, reward_with_unit_temperature_before_swap, k)
                            expected_next, _ = self.swap_chain(swap, expected_next, reward_with_unit_temperature_before_swap, k)
            batch = expected_next + sample_noise
        return batch

    def update_chain(self, expected_next, param, reward_fn, reward_with_unit_temperature, inverse_temperature, std, *args_kernel_propose):
        new_param = self.kernel.propose(param, *args_kernel_propose, std)
        new_reward_with_unit_temperature = reward_fn(expected_next, new_param)
        accept = self.kernel.get_acceptance(
            inverse_temperature * reward_with_unit_temperature,
            inverse_temperature * new_reward_with_unit_temperature,
        )
        updated_reward_with_unit_temperature = torch.where(accept, new_reward_with_unit_temperature, reward_with_unit_temperature)
        accept = accept.reshape(*accept.shape, *[1 for _ in range(param.ndim - accept.ndim)])
        updated_param = torch.where(accept, new_param, param)
        return updated_param, updated_reward_with_unit_temperature

    def swap_chain(self, swap, param, reward_with_unit_temperature, k):
        reward_temp_1_kp1 = torch.where(swap, reward_with_unit_temperature[k], reward_with_unit_temperature[k + 1])
        reward_temp_1_k = torch.where(swap, reward_with_unit_temperature[k + 1], reward_with_unit_temperature[k])
        reward_with_unit_temperature[k] = reward_temp_1_k
        reward_with_unit_temperature[k + 1] = reward_temp_1_kp1
        swap = swap.reshape(*swap.shape, *[1 for _ in range(param.ndim - 1 - swap.ndim)])
        param_kp1 = torch.where(swap, param[k], param[k + 1])
        param_k = torch.where(swap, param[k + 1], param[k])
        param[k] = param_k
        param[k + 1] = param_kp1
        return param, reward_with_unit_temperature

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
