from typing import NamedTuple

import torch


__all__ = [
    'CREPE',
    'CREPEResult',
    'PathLogRatio',
    'SaveSampleFn',
    'TargetLogRatio',
]


class CREPEResult(NamedTuple):
    """Samples, final replicas, and communication diagnostics from CREPE."""

    samples: torch.Tensor
    replicas: torch.Tensor
    acceptance_rate: torch.Tensor
    acceptance_count: torch.Tensor
    attempt_count: torch.Tensor


class SaveSampleFn:
    """Collection of policies for retaining target-replica samples."""

    @staticmethod
    def always(iteration, sample):
        """Retain the target sample after every iteration.

        Args
        ----
        iteration : int
            Zero-based CREPE iteration index.
        sample : torch.Tensor[batch_size, ...]
            Current target-replica sample.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @staticmethod
    def burn_in_and_thin(iteration, sample, *, burn_in, thin):
        """Retain samples after burn-in at intervals of ``thin`` iterations.

        Args
        ----
        iteration : int
            Zero-based CREPE iteration index.
        sample : torch.Tensor[batch_size, ...]
            Current target-replica sample. Its value does not affect this
            policy.

        Keyword Args
        ------------
        burn_in : int
            Nonnegative number of initial iterations to discard.
        thin : int
            Positive interval between retained samples after burn-in.

        Returns
        -------
        bool
            Whether to retain the current sample.
        """
        if burn_in < 0:
            raise ValueError('burn_in must be nonnegative.')
        if thin < 1:
            raise ValueError('thin must be greater than 0.')
        return iteration >= burn_in and (iteration - burn_in) % thin == 0


class PathLogRatio:
    """Utilities for computing path log Radon-Nikodym estimators."""

    @staticmethod
    def from_transition_log_probabilities(forward_log_probability, backward_log_probability):
        r"""Return :math:`\sum_k (\log B_k - \log F_k)` for each batch member.

        This is the log form of Equation 3 of He et al. (2026); Equations 22
        and 25 give the discrete proposal- and model-process forms.

        Args
        ----
        forward_log_probability : torch.Tensor[step_count, batch_size]
            Per-step forward transition log probabilities for paths represented
            in increasing-time orientation.
        backward_log_probability : torch.Tensor[step_count, batch_size]
            Per-step backward transition log probabilities evaluated along the
            same paths and in the same increasing-time orientation.

        Returns
        -------
        torch.Tensor[batch_size]
            Path log RNE for each batch member.
        """
        PathLogRatio._validate_transition_log_probabilities(forward_log_probability, backward_log_probability)
        return (backward_log_probability - forward_log_probability).sum(dim=0)

    @staticmethod
    def from_euler_maruyama(
        times,
        path,
        forward_drift,
        backward_drift,
        diffusion,
    ):
        r"""Return the exact log RNE of one Euler-Maruyama transition.

        ``path`` is represented from the less-noisy endpoint to the more-noisy
        endpoint. The forward kernel has mean
        ``path[0] + dt * forward_drift[0]`` and the backward kernel has mean
        ``path[1] - dt * backward_drift[1]``. This matches a backward SDE drift
        integrated over a negative time increment.

        Args
        ----
        times : torch.Tensor[2]
            Strictly increasing endpoint times.
        path : torch.Tensor[2, batch_size, ...]
            Less-noisy and more-noisy endpoint states.
        forward_drift : torch.Tensor[2, batch_size, ...]
            Forward drift at both endpoints. Only the less-noisy value is used.
        backward_drift : torch.Tensor[2, batch_size, ...]
            Backward drift at both endpoints. Only the more-noisy value is used.
        diffusion : torch.Tensor
            Diffusion coefficients at both endpoints, with the same broadcasting
            rules as :meth:`from_ito_integrals`.

        Returns
        -------
        torch.Tensor[batch_size]
            ``log backward transition - log forward transition``.
        """
        PathLogRatio._validate_sde_path(times, path, forward_drift, backward_drift)
        diffusion = PathLogRatio._broadcast_diffusion(diffusion, path)
        time_delta = times[1] - times[0]
        forward_variance = time_delta * diffusion[0].square()
        backward_variance = time_delta * diffusion[1].square()
        forward_residual = path[1] - (path[0] + time_delta * forward_drift[0])
        backward_residual = path[0] - (path[1] - time_delta * backward_drift[1])

        log_two_pi = torch.log(path.new_tensor(2 * torch.pi))
        forward_log_probability = -0.5 * (
            forward_residual.square() / forward_variance
            + log_two_pi
            + forward_variance.log()
        )
        backward_log_probability = -0.5 * (
            backward_residual.square() / backward_variance
            + log_two_pi
            + backward_variance.log()
        )
        event_dims = tuple(range(1, path[0].ndim))
        if event_dims:
            forward_log_probability = forward_log_probability.sum(dim=event_dims)
            backward_log_probability = backward_log_probability.sum(dim=event_dims)
        return backward_log_probability - forward_log_probability

    @staticmethod
    def from_ito_integrals(
        times,
        path,
        forward_drift,
        backward_drift,
        diffusion,
    ):
        r"""Estimate the SDE log RNE over one adjacent-replica interval.

        This discretizes Equations 15--17 of He et al. (2026). ``times`` has
        shape ``(2,)`` and is ordered from less noisy to more noisy. ``path``,
        ``forward_drift``, and ``backward_drift`` have shape
        ``(2, batch_size, ...)``. ``diffusion`` contains one scalar coefficient
        per time, optionally per batch member, and is broadcast across event
        dimensions.

        The forward Ito integral uses the less-noisy endpoint, the backward Ito
        integral uses the more-noisy endpoint, and the ordinary time integral
        uses a one-step trapezoidal rule.

        Args
        ----
        times : torch.Tensor[2]
            The less-noisy and more-noisy diffusion times, in that order.
        path : torch.Tensor[2, batch_size, ...]
            Path states at the less-noisy and more-noisy endpoints.
        forward_drift : torch.Tensor[2, batch_size, ...]
            Forward-process drift evaluated at both path endpoints.
        backward_drift : torch.Tensor[2, batch_size, ...]
            Backward-process drift evaluated at both path endpoints.
        diffusion : torch.Tensor
            Scalar diffusion coefficients at both endpoints. The leading axis
            has length two; an optional batch axis has length one or
            ``batch_size``, and any event axes must be singleton.

        Returns
        -------
        torch.Tensor[batch_size]
            Two-endpoint path log RNE for each batch member.
        """
        PathLogRatio._validate_sde_path(times, path, forward_drift, backward_drift)

        diffusion = PathLogRatio._broadcast_diffusion(diffusion, path)
        time_delta = times[1] - times[0]
        path_delta = path[1] - path[0]
        diffusion_squared = diffusion.square()

        backward_ito = backward_drift[1] * path_delta / diffusion_squared[1]
        forward_ito = forward_drift[0] * path_delta / diffusion_squared[0]
        drift_energy_less_noisy = (forward_drift[0].square() - backward_drift[0].square()) / diffusion_squared[0]
        drift_energy_more_noisy = (forward_drift[1].square() - backward_drift[1].square()) / diffusion_squared[1]

        event_dims = tuple(range(1, path_delta.ndim))
        if event_dims:
            backward_ito = backward_ito.sum(dim=event_dims)
            forward_ito = forward_ito.sum(dim=event_dims)
            drift_energy_less_noisy = drift_energy_less_noisy.sum(dim=event_dims)
            drift_energy_more_noisy = drift_energy_more_noisy.sum(dim=event_dims)

        drift_energy = time_delta * (drift_energy_less_noisy + drift_energy_more_noisy) / 4
        return backward_ito - forward_ito + drift_energy

    @staticmethod
    def from_reference_stabilized_transition_log_probabilities(
        forward_log_probability, backward_log_probability,
        reference_forward_log_probability, reference_backward_log_probability,
        reference_log_probability_less_noisy, reference_log_probability_more_noisy,
    ):
        """Return the Equation 34 reference-stabilized log RNE.

        The reference endpoint correction uses the identity in Equation 30 of
        He et al. (2026).

        Args
        ----
        forward_log_probability : torch.Tensor[step_count, batch_size]
            Per-step forward transition log probabilities for the process whose
            log RNE is being computed.
        backward_log_probability : torch.Tensor[step_count, batch_size]
            Per-step backward transition log probabilities for that process.
        reference_forward_log_probability : torch.Tensor[step_count, batch_size]
            Per-step forward transition log probabilities for the reference
            process, evaluated along the same paths.
        reference_backward_log_probability : torch.Tensor[step_count, batch_size]
            Per-step backward transition log probabilities for the reference
            process, evaluated along the same paths.
        reference_log_probability_less_noisy : torch.Tensor[batch_size]
            Reference marginal log probability of each less-noisy endpoint.
        reference_log_probability_more_noisy : torch.Tensor[batch_size]
            Reference marginal log probability of each more-noisy endpoint.

        Returns
        -------
        torch.Tensor[batch_size]
            Reference-stabilized path log RNE for each batch member.
        """
        PathLogRatio._validate_transition_log_probabilities(forward_log_probability, backward_log_probability)
        PathLogRatio._validate_transition_log_probabilities(reference_forward_log_probability, reference_backward_log_probability)
        if reference_forward_log_probability.shape != forward_log_probability.shape:
            raise ValueError('reference and proposal log probabilities must have the same shape.')
        expected_endpoint_shape = forward_log_probability.shape[1:]
        if reference_log_probability_less_noisy.shape != expected_endpoint_shape:
            raise ValueError('less-noisy reference log probability must have shape (batch_size,).')
        if reference_log_probability_more_noisy.shape != expected_endpoint_shape:
            raise ValueError('more-noisy reference log probability must have shape (batch_size,).')
        endpoint_logs = (
            reference_log_probability_less_noisy,
            reference_log_probability_more_noisy,
        )
        if not all(torch.is_floating_point(value) for value in endpoint_logs):
            raise ValueError('reference endpoint log probabilities must have floating-point dtypes.')

        transition_log_ratio = (
            backward_log_probability - reference_backward_log_probability
            + reference_forward_log_probability - forward_log_probability
        ).sum(dim=0)
        return (
            transition_log_ratio
            + reference_log_probability_less_noisy - reference_log_probability_more_noisy
        )

    @staticmethod
    def _validate_transition_log_probabilities(forward, backward):
        if forward.ndim != 2:
            raise ValueError('transition log probabilities must have shape (step_count, batch_size).')
        if forward.shape != backward.shape:
            raise ValueError('forward and backward log probabilities must have the same shape.')
        if forward.shape[0] < 1:
            raise ValueError('transition log probabilities must contain at least one step.')
        if not torch.is_floating_point(forward) or not torch.is_floating_point(backward):
            raise ValueError('transition log probabilities must have floating-point dtypes.')
        if forward.device != backward.device:
            raise ValueError('forward and backward log probabilities must be on the same device.')

    @staticmethod
    def _validate_sde_path(times, path, forward_drift, backward_drift):
        if times.ndim != 1 or times.shape[0] != 2:
            raise ValueError('times must have shape (2,).')
        if path.ndim < 2 or path.shape[0] != 2:
            raise ValueError('path must have shape (2, batch_size, ...).')
        if forward_drift.shape != path.shape:
            raise ValueError('forward_drift must have the shape of path.')
        if backward_drift.shape != path.shape:
            raise ValueError('backward_drift must have the shape of path.')
        if times.device != path.device:
            raise ValueError('times and path must be on the same device.')
        if forward_drift.device != path.device or backward_drift.device != path.device:
            raise ValueError('drifts and path must be on the same device.')
        if not torch.is_floating_point(times) or not torch.is_floating_point(path):
            raise ValueError('times and path must have floating-point dtypes.')
        if not torch.is_floating_point(forward_drift) or not torch.is_floating_point(backward_drift):
            raise ValueError('drifts must have floating-point dtypes.')

    @staticmethod
    def _broadcast_diffusion(diffusion, path):
        if diffusion.ndim < 1 or diffusion.shape[0] != 2:
            raise ValueError('diffusion must have a leading axis of length 2.')
        if not torch.is_floating_point(diffusion):
            raise ValueError('diffusion must have a floating-point dtype.')
        if diffusion.device != path.device:
            raise ValueError('diffusion and path must be on the same device.')
        if diffusion.ndim > path.ndim:
            raise ValueError('diffusion has too many dimensions for path.')
        if diffusion.ndim >= 2 and diffusion.shape[1] not in (1, path.shape[1]):
            raise ValueError('diffusion batch axis must have length 1 or batch_size.')
        if any(size != 1 for size in diffusion.shape[2:]):
            raise ValueError('diffusion must be scalar across event dimensions.')
        while diffusion.ndim < path.ndim:
            diffusion = diffusion.unsqueeze(-1)
        try:
            return torch.broadcast_to(diffusion, path.shape)
        except RuntimeError as error:
            raise ValueError('diffusion must be broadcastable to path.') from error


class TargetLogRatio:
    """Utilities for constructing diffusion target log-density ratios."""

    @staticmethod
    def base_diffusion(model_log_rne):
        """Return the Equation 4 target ratio for one base diffusion model.

        Args
        ----
        model_log_rne : torch.Tensor[batch_size]
            Model-process path log RNE in increasing-time orientation.

        Returns
        -------
        torch.Tensor[batch_size]
            Base diffusion target log-density ratio for each batch member.
        """
        return -model_log_rne

    @staticmethod
    def tempered(model_log_rne, inverse_temperature):
        """Return the Equation 45 target ratio for a tempered diffusion.

        Args
        ----
        model_log_rne : torch.Tensor[batch_size]
            Model-process path log RNE in increasing-time orientation.
        inverse_temperature : float or torch.Tensor
            Inverse temperature multiplying the model log-density ratio. A
            tensor must be broadcastable to ``model_log_rne``.

        Returns
        -------
        torch.Tensor[batch_size]
            Tempered target log-density ratio for each batch member.
        """
        return -inverse_temperature * model_log_rne

    @staticmethod
    def reward_tilt(model_log_rne, reward_less_noisy, reward_more_noisy):
        """Return the Equation 48 reward-tilted target ratio.

        Args
        ----
        model_log_rne : torch.Tensor[batch_size]
            Model-process path log RNE in increasing-time orientation.
        reward_less_noisy : torch.Tensor[batch_size]
            Reward evaluated at each less-noisy path endpoint.
        reward_more_noisy : torch.Tensor[batch_size]
            Reward evaluated at each more-noisy path endpoint.

        Returns
        -------
        torch.Tensor[batch_size]
            Reward-tilted target log-density ratio for each batch member.
        """
        if reward_less_noisy.shape != model_log_rne.shape:
            raise ValueError('less-noisy reward must have the shape of model_log_rne.')
        if reward_more_noisy.shape != model_log_rne.shape:
            raise ValueError('more-noisy reward must have the shape of model_log_rne.')
        return -model_log_rne + reward_more_noisy - reward_less_noisy

    @staticmethod
    def weighted_diffusions(model_log_rnes, weights):
        """Return a weighted product target generalizing Equations 45 and 50.

        Args
        ----
        model_log_rnes : torch.Tensor[model_count, batch_size]
            Path log RNE for every model and batch member, all represented in
            increasing-time orientation.
        weights : torch.Tensor[model_count]
            Exponent applied to each model density in the weighted product.

        Returns
        -------
        torch.Tensor[batch_size]
            Weighted-product target log-density ratio for each batch member.
        """
        if model_log_rnes.ndim != 2:
            raise ValueError('model_log_rnes must have shape (model_count, batch_size).')
        if weights.ndim != 1 or weights.shape[0] != model_log_rnes.shape[0]:
            raise ValueError('weights must have shape (model_count,).')
        if weights.device != model_log_rnes.device:
            raise ValueError('weights and model_log_rnes must be on the same device.')
        return -(weights[:, None] * model_log_rnes).sum(dim=0)


class CREPE:
    r"""Control with REPlica Exchange (CREPE).

    This implements the non-reversible replica-exchange algorithm from
    ``CREPE: Controlling diffusion with REPlica Exchange`` (He et al., 2026).
    Replica ``m`` targets the diffusion marginal :math:`\pi_{t_m}` for a
    strictly increasing ladder :math:`t_0 < \cdots < t_M`. Communication
    proposes a forward path from :math:`t_m` to :math:`t_{m+1}` and an
    independent backward path in the opposite direction. Smaller times are
    less noisy, so ``t0`` is the less-noisy endpoint and ``t1`` is the
    more-noisy endpoint of every path callback.

    Each path callback returns ``(proposed, log_target_ratio,
    log_proposal_rne)``. Both log ratios must describe the path in increasing
    time order, irrespective of the direction in which it was generated:

    .. math::

        \Delta_\pi(x) &= \log \pi_{t_1}(x_{t_1})
            - \log \pi_{t_0}(x_{t_0}), \\
        \log R^Q(x) &= \log B^Q(x_{t_0} \mid x_{t_1})
            - \log F^Q(x_{t_1} \mid x_{t_0}).

    The callback may accumulate these quantities over any internal path
    discretization. CREPE accepts both proposed endpoints with log ratio

    .. math::

        \Delta_\pi(x) - \Delta_\pi(x')
        + \log R^Q(x) - \log R^Q(x').

    Keyword Args
    ------------
    init_path_fn : Callable[times: torch.Tensor[replica_count]]
        Returns one initialized replica ladder with shape
        ``(replica_count, ...)``, not a batch. It will commonly run one reverse
        diffusion trajectory and retain its states at ``times``.
    forward_path_fn : Callable[t0: scalar, t1: scalar, param: torch.Tensor[batch_size, ...]]
        Generates a path from the less-noisy ``param`` at ``t0`` to the
        more-noisy time ``t1`` and returns its proposed endpoint, target
        log-density ratio, and proposal log RNE. The proposed endpoint has shape
        ``(batch_size, ...)`` and both log-ratio outputs have shape
        ``(batch_size,)``.
    backward_path_fn : Callable[t0: scalar, t1: scalar, param: torch.Tensor[batch_size, ...]]
        Generates a path backward from the more-noisy ``param`` at ``t1`` to
        the less-noisy time ``t0``. Returns the proposed endpoint and the same
        increasing-time log ratios as ``forward_path_fn``. The proposed endpoint has
        shape ``(batch_size, ...)`` and both log-ratio outputs have shape
        ``(batch_size,)``.
    local_exploration_fn : Callable[time: scalar, param: torch.Tensor[batch_size, ...]], optional
        Applies one local exploration transition at a replica level after each
        communication step. It must return a tensor with the shape of ``param``.
    refresh_reference_fn : Callable[time: scalar], optional
        Returns one exact sample, not a batch, from the reference distribution
        at the noisiest level. When provided, the last replica is refreshed
        after every iteration.

    Notes
    -----
    The path callbacks receive an entire target batch. CREPE uses
    :func:`torch.vmap` to evaluate all disjoint replica pairs in a checkerboard
    communication step, with independent randomness for separate pairs. Path
    callbacks are responsible for drawing independent randomness across their
    batch dimension when independent chains are required.
    """

    def __init__(
        self,
        *,
        init_path_fn,
        forward_path_fn,
        backward_path_fn,
        local_exploration_fn=None,
        refresh_reference_fn=None,
    ):
        self.init_path_fn = torch.vmap(init_path_fn, out_dims=1, randomness='different')
        self.forward_path_fn = torch.vmap(forward_path_fn, in_dims=(0, 0, 0), randomness='different')
        self.backward_path_fn = torch.vmap(backward_path_fn, in_dims=(0, 0, 0), randomness='different')
        self.local_exploration_fn = (
            None if local_exploration_fn is None
            else torch.vmap(local_exploration_fn, in_dims=(0, 0), randomness='different')
        )
        self.refresh_reference_fn = (
            None if refresh_reference_fn is None
            else torch.vmap(refresh_reference_fn, randomness='different')
        )

    def run(
        self,
        times,
        iteration_count,
        *,
        batch_size=1,
        should_save_sample_fn,
    ):
        """Run CREPE and return retained target samples and diagnostics.

        Args
        ----
        times : one-dimensional torch.Tensor
            Strictly increasing diffusion times, ordered from the least-noisy
            target replica to the most-noisy reference replica.
        iteration_count : int
            Number of checkerboard communication iterations. Iteration one
            updates edges ``(0, 1), (2, 3), ...`` and iteration two updates
            ``(1, 2), (3, 4), ...``.

        Keyword Args
        ------------
        batch_size : int, default 1
            Number of independent CREPE chains processed together.
        should_save_sample_fn : Callable[iteration: int, sample: torch.Tensor[batch_size, ...]]
            Returns whether to retain the current target-replica sample. The
            iteration index is zero-based and the callback runs after
            communication, local exploration, and reference refresh.

        Returns
        -------
        CREPEResult
            ``samples`` has shape ``(sample_count, batch_size, ...)`` and
            contains states from the target replica at ``times[0]``.
            ``replicas`` has shape ``(replica_count, batch_size, ...)``. The
            three diagnostic tensors have shape
            ``(replica_count - 1, batch_size)``.

        Notes
        -----
        When ``times[0]`` is greater than the terminal data time, ``samples``
        must be denoised to the terminal time by the caller.
        """
        times = torch.as_tensor(times)
        self._validate_run_args(times, iteration_count, batch_size)

        replicas = self._initialize(times, batch_size)
        replica_count = times.numel()
        acceptance_count = torch.zeros(replica_count - 1, batch_size, dtype=torch.long, device=replicas.device)
        attempt_count = torch.zeros_like(acceptance_count)

        checkerboard_edges = [
            torch.arange(parity + 1, replica_count, 2, device=times.device)
            for parity in range(2)
        ]

        samples = []
        for iteration in range(iteration_count):
            time_edges = checkerboard_edges[iteration % 2]
            if time_edges.numel() > 0:
                less_noisy = replicas[time_edges - 1]
                more_noisy = replicas[time_edges]
                t0 = times[time_edges - 1]
                t1 = times[time_edges]

                proposed_more_noisy, forward_target, forward_rne = self.forward_path_fn(t0, t1, less_noisy)
                proposed_less_noisy, backward_target, backward_rne = self.backward_path_fn(t0, t1, more_noisy)
                self._validate_path_outputs('forward_path_fn', proposed_more_noisy, forward_target, forward_rne, more_noisy)
                self._validate_path_outputs('backward_path_fn', proposed_less_noisy, backward_target, backward_rne, less_noisy)

                log_acceptance = self.log_acceptance_ratio(forward_target, forward_rne, backward_target, backward_rne)
                accept = torch.rand_like(log_acceptance).log() < log_acceptance.clamp(max=0)
                mask = accept.reshape(*accept.shape, *([1] * (replicas.ndim - accept.ndim)))
                replicas = replicas.index_copy(0, time_edges - 1, torch.where(mask, proposed_less_noisy, less_noisy))
                replicas = replicas.index_copy(0, time_edges, torch.where(mask, proposed_more_noisy, more_noisy))
                acceptance_count[time_edges - 1] += accept.to(torch.long)
                attempt_count[time_edges - 1] += 1

            if self.local_exploration_fn is not None:
                exploration_times = times
                if self.refresh_reference_fn is not None:
                    exploration_times = times[:-1]
                exploration_replicas = replicas[:exploration_times.numel()]
                explored = self.local_exploration_fn(exploration_times, exploration_replicas)
                if explored.shape != exploration_replicas.shape:
                    raise ValueError('local_exploration_fn must return a tensor with the shape of its param input.')
                if explored.device != replicas.device:
                    raise ValueError('local_exploration_fn must return a tensor on the param device.')
                if self.refresh_reference_fn is None:
                    replicas = explored
                else:
                    replicas = torch.cat((explored, replicas[-1:]), dim=0)

            if self.refresh_reference_fn is not None:
                refreshed = self._refresh_reference(times[-1], batch_size)
                if refreshed.shape != replicas[-1].shape:
                    raise ValueError('refresh_reference_fn must return one sample with the replica event shape.')
                if refreshed.device != replicas.device:
                    raise ValueError('refresh_reference_fn must return a tensor on the replica device.')
                replicas = replicas.index_copy(0, torch.tensor([replica_count - 1], device=replicas.device), refreshed[None])

            sample = replicas[0]
            if should_save_sample_fn(iteration, sample):
                samples.append(sample.clone())

        acceptance_rate = acceptance_count / attempt_count.clamp(min=1)
        samples = (
            torch.stack(samples)
            if samples
            else replicas.new_empty(0, *replicas.shape[1:])
        )
        return CREPEResult(
            samples=samples,
            replicas=replicas,
            acceptance_rate=acceptance_rate,
            acceptance_count=acceptance_count,
            attempt_count=attempt_count,
        )

    @staticmethod
    def log_acceptance_ratio(forward_target, forward_rne, backward_target, backward_rne):
        """Return the Equation 13 CREPE path-space Metropolis-Hastings log ratio.

        Args
        ----
        forward_target : torch.Tensor
            Target log-density ratio for each forward-generated path.
        forward_rne : torch.Tensor
            Proposal log RNE for each forward-generated path.
        backward_target : torch.Tensor
            Target log-density ratio for each backward-generated path, reported
            in increasing-time orientation.
        backward_rne : torch.Tensor
            Proposal log RNE for each backward-generated path, reported in
            increasing-time orientation.

        Returns
        -------
        torch.Tensor
            Path-space Metropolis-Hastings log ratio with the input shape.
        """
        return forward_target - backward_target + forward_rne - backward_rne

    def _initialize(self, times, batch_size):
        batched_times = times.expand(batch_size, -1)
        replicas = self.init_path_fn(batched_times)
        if replicas.ndim < 2 or replicas.shape[:2] != (times.numel(), batch_size):
            raise ValueError('init_path_fn must return one ladder with shape (replica_count, ...).')
        return replicas

    def _refresh_reference(self, time, batch_size):
        return self.refresh_reference_fn(time.expand(batch_size))

    @staticmethod
    def _validate_run_args(times, iteration_count, batch_size):
        if times.ndim != 1 or times.numel() < 2:
            raise ValueError('times must be a one-dimensional tensor with at least two entries.')
        if not torch.is_floating_point(times):
            raise ValueError('times must have a floating-point dtype.')
        if not bool(torch.isfinite(times).all()):
            raise ValueError('times must contain only finite values.')
        if not bool((times[1:] > times[:-1]).all()):
            raise ValueError('times must be strictly increasing.')
        if iteration_count < 1:
            raise ValueError('iteration_count must be greater than 0.')
        if batch_size < 1:
            raise ValueError('batch_size must be greater than 0.')

    @staticmethod
    def _validate_path_outputs(name, proposed, target_ratio, proposal_rne, expected):
        if proposed.shape != expected.shape:
            raise ValueError(f'{name} proposed endpoint must have the shape of the destination replicas.')
        if proposed.device != expected.device:
            raise ValueError(f'{name} proposed endpoint must be on the replica device.')
        expected_log_shape = expected.shape[:2]
        if target_ratio.shape != expected_log_shape:
            raise ValueError(f'{name} target log ratio must have shape (pair_count, batch_size).')
        if proposal_rne.shape != expected_log_shape:
            raise ValueError(f'{name} proposal log RNE must have shape (pair_count, batch_size).')
        if not torch.is_floating_point(target_ratio) or not torch.is_floating_point(proposal_rne):
            raise ValueError(f'{name} log ratios must have floating-point dtypes.')
        if target_ratio.device != expected.device or proposal_rne.device != expected.device:
            raise ValueError(f'{name} log ratios must be on the replica device.')
        if not bool(torch.isfinite(target_ratio).all()) or not bool(torch.isfinite(proposal_rne).all()):
            raise ValueError(f'{name} log ratios must contain only finite values.')
