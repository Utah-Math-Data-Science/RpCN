"""CREPE (§16): replicas at levels u_0 < ... < u_M = 1, path exchanges between adjacent levels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .evaluate import evaluate_rows, start_record
from .exchange import EvenOdd, replica_exchange
from .moves import MoveContext
from .ops import device_of, namespace, put_levels, stack_levels, take_levels, tree_map
from .potentials import FKDiff, Relative, ReverseRatio, Sum, TiltEnds, Zero
from .records import Feature, PartialRecord
from .resample import ESSBelow, Never
from .smc import SMCResult, smc


@dataclass(frozen=True)
class CREPEConfig:
    """Configuration of CREPE: replicas at noise levels u_0 < ... < u_M = 1, exchanging paths between neighbours.

    Level 0 (u_0, usually 0) is the data end and holds the tilted target; level M is the prior. If u_0 > 0, level 0
    samples ``p_{u_0} exp(lambda(tau(u_0)) r_hat)`` and `crepe_finish` carries its samples to u = 0 (`crepe_sample`
    does both in one call). A path exchange between levels m and m+1 runs `steps_per_gap` fine steps across the gap:
    up (towards noise) with `proposal_noising` as the proposal and `model_noising` as the target, and down (towards
    data) with `proposal_denoising` as the proposal and `model_denoising` as the target.

    Parameters
    ----------
    u_levels : tuple of float
        Increasing schedule coordinates of the levels; ``u_levels[-1]`` must be 1.
    steps_per_gap : int
        Number of fine steps per gap between adjacent levels.
    model_noising : Kernel
        Forward (noising) kernel of the reference process, e.g. ``Noising(schedule)``.
    model_denoising : Kernel
        Backward (denoising) kernel of the model, e.g. ``EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD)``.
    proposal_noising : Kernel
        Forward proposal kernel. Passing the same object as `model_noising` makes the forward terms cancel statically.
    proposal_denoising : Kernel
        Backward proposal kernel, e.g. ``Guided(model_denoising, tilt(lam))``.
    lam : Callable[[array], array]
        Tilt strength by time, e.g. ``level_lambda(schedule.time(u_levels), beta, rho)``. Should be 0 at the top
        level, so that refreshing it with a prior draw is exact.
    pairs : PairSchedule, optional
        Which adjacent pairs exchange in each sweep. Default `EvenOdd()`.
    local_move : Move, optional
        A move applied to the replicas at `local_levels` after each exchange sweep, e.g.
        ``LangevinMove(Guided(Langevin(h), tilt(lam, 0.5)))``. None for no local move.
    local_levels : tuple of int, optional
        Static subset of levels 0..M-1 the local move applies to.
    out_needs : frozenset of Feature, optional
        Features every replica carries for output (e.g. for `emit`). Default ``{Feature.X0}``.
    """

    u_levels: tuple                      # increasing; u_levels[-1] = 1
    steps_per_gap: int                   # fine steps per gap
    model_noising: Any
    model_denoising: Any
    proposal_noising: Any
    proposal_denoising: Any
    lam: Callable                        # e.g. level_lambda(time(u_levels), beta, rho)
    pairs: Any = EvenOdd()
    local_move: Any = None               # e.g. LangevinMove(Guided(Langevin(h), tilt(lam, 0.5)))
    local_levels: tuple = ()             # static subset of 0..M-1
    out_needs: frozenset = frozenset({Feature.X0})

    @property
    def n_gaps(self):
        """Index of the top level: the number of levels minus one."""
        return len(self.u_levels) - 1

    @property
    def replica_needs(self):
        """Features every replica carries.

        These are the features the four kernels and the local move require, plus REWARD and `out_needs`.
        """
        kernels = (self.model_noising, self.model_denoising, self.proposal_noising, self.proposal_denoising)
        requirements = [kernel.requires for kernel in kernels]
        requirements += [self.local_move.requires] if self.local_move else []
        return frozenset().union(*requirements) | {Feature.REWARD} | frozenset(self.out_needs)


def fine_grid(cfg, gap, *, xp, dtype, device=None):
    """Return the fine grid across gap `gap`, ascending.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    gap : int
        Gap index in 0..M-1, between levels ``gap`` and ``gap + 1``.
    xp : namespace
        Array namespace to create the grid with.
    dtype : dtype
        Grid dtype.
    device : device, optional
        Grid device.

    Returns
    -------
    array
        Schedule coordinates, shape [K+1]; the endpoints are exactly ``u_levels[gap]`` and ``u_levels[gap + 1]``.
    """
    u_lo, u_hi = cfg.u_levels[gap], cfg.u_levels[gap + 1]
    points = [u_lo, *(u_lo + (u_hi - u_lo) * j / cfg.steps_per_gap for j in range(1, cfg.steps_per_gap)), u_hi]
    return xp.asarray(points, dtype=dtype, device=device)


def per_row_grid(cfg, gaps, n_instances, ascending, *, xp, dtype, device=None):
    """Per-row fine grids for several gaps at once, as `smc` takes them.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    gaps : sequence of int
        The P gap indices.
    n_instances : int
        Number of instances.
    ascending : bool
        True for grids running up (towards noise), False for down.
    xp : namespace
        Array namespace.
    dtype : dtype
        Grid dtype.
    device : device, optional
        Grid device.

    Returns
    -------
    array
        Schedule coordinates, shape [I, P, K+1].
    """
    grids = xp.stack([fine_grid(cfg, gap, xp=xp, dtype=dtype, device=device) for gap in gaps])
    grids = grids if ascending else xp.flip(grids, axis=-1)
    return xp.broadcast_to(grids[None, ...], (n_instances, *grids.shape))


@dataclass(frozen=True)
class PathExchange:
    """CREPE's exchange: two one-directional `smc` calls per round, up from the lower level and down from the upper.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    schedule : Schedule
        The noise schedule.
    """

    cfg: CREPEConfig
    evaluate: Callable
    schedule: Any

    def propose(self, ladder, lower, upper, key, backend):   # pair (lower[p], upper[p] = lower[p] + 1) is gap lower[p]
        """Propose swapped replicas for the pairs (lower[p], upper[p]).

        The lower replica's path is pushed up across the gap and the upper replica's path down; each run is
        weighted by the endpoint tilts and the reverse-kernel ratio.

        Parameters
        ----------
        ladder : Record
            Replicas, shape [I, M+1, ...].
        lower : sequence of int
            Lower levels of the P pairs.
        upper : sequence of int
            Upper levels, ``upper[p] = lower[p] + 1``.
        key : backend key
            Random key.
        backend : Backend
            Randomness and control flow.

        Returns
        -------
        new_lower : Record
            Candidates for the lower levels (the end of the downward path), shape [I, P, ...].
        new_upper : Record
            Candidates for the upper levels (the end of the upward path), shape [I, P, ...].
        log_accept_ratio : array
            Log acceptance ratio, shape [I, P].
        """
        cfg = self.cfg
        key_up, key_down = backend.split(key, 2)
        xp = namespace(ladder.t)
        n_instances, dtype, device = ladder.t.shape[0], ladder.t.dtype, device_of(ladder.t)

        def path_potential(target_reverse, proposal_reverse):
            return Sum(TiltEnds(cfg.lam), ReverseRatio(target_reverse, proposal_reverse))

        def run_path(start, grid, target, proposal, potential, key):
            return smc(start, grid, backend=backend, evaluate=self.evaluate, schedule=self.schedule, target=target,
                       proposal=proposal, potential=potential, policy=Never(), out_needs=cfg.replica_needs,
                       key=key)

        def gap_grids(ascending):
            return per_row_grid(cfg, lower, n_instances, ascending, xp=xp, dtype=dtype, device=device)

        upward = run_path(take_levels(ladder, lower), gap_grids(True), cfg.model_noising, cfg.proposal_noising,
                          path_potential(cfg.model_denoising, cfg.proposal_denoising), key_up)
        downward = run_path(take_levels(ladder, upper), gap_grids(False), cfg.model_denoising, cfg.proposal_denoising,
                            path_potential(cfg.model_noising, cfg.proposal_noising), key_down)
        return downward.final, upward.final, upward.log_weights + downward.log_weights   # lower <- z_0, upper <- y_K


def top_replica(cfg, n_instances, event_shape, *, backend, evaluate, schedule, dtype, device, key):
    """Draw a fresh top-level replica from the prior at u = 1, evaluated with ``cfg.replica_needs``.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    n_instances : int
        Number of instances.
    event_shape : tuple of int
        Event shape.
    backend : Backend
        Randomness and control flow; ``backend.xp`` creates the arrays.
    evaluate : Callable[[array, array, frozenset], Record]
        The model, e.g. from `make_evaluate_torch` or `make_evaluate_jax`.
    schedule : Schedule
        The noise schedule; its prior gives the top level.
    dtype : dtype
        Run dtype.
    device : device
        Run device.
    key : backend key
        Random key.

    Returns
    -------
    Record
        The replica, shape [I, 1, *D].
    """
    xp = backend.xp
    prior = schedule.prior((n_instances, 1), event_shape, xp=xp, dtype=dtype, device=device)
    t_top = schedule.time(xp.ones((n_instances, 1), dtype=dtype, device=device))
    return start_record(prior, t_top, cfg.replica_needs, evaluate, backend, key)[0]


def crepe_init(cfg, n_instances, event_shape, *, backend, evaluate, schedule, dtype, device=None, key):
    """Build the initial replicas: a prior draw at the top, carried down with the model's kernel `model_denoising`.

    Runs M chained, unweighted `smc` calls, one per gap from M-1 down to 0.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    n_instances : int
        Number of independent chains.
    event_shape : tuple of int
        Event shape.
    backend : Backend
        Randomness and control flow; ``backend.xp`` creates the arrays.
    evaluate : Callable[[array, array, frozenset], Record]
        The model, e.g. from `make_evaluate_torch` or `make_evaluate_jax`.
    schedule : Schedule
        The noise schedule; its prior gives the top level.
    dtype : dtype
        Run dtype.
    device : device, optional
        Run device.
    key : backend key
        Random key.

    Returns
    -------
    Record
        Replicas, shape [I, M+1, *D]; level m has t = ``schedule.time(u_levels[m])`` and carries
        ``cfg.replica_needs``.
    """
    xp = backend.xp
    key_top, _unused_key, *gap_keys = backend.split(key, cfg.n_gaps + 2)
    top = top_replica(cfg, n_instances, event_shape, backend=backend, evaluate=evaluate, schedule=schedule,
                      dtype=dtype, device=device, key=key_top)
    levels, current = [None] * cfg.n_gaps + [top], top
    for level in range(cfg.n_gaps, 0, -1):                           # static loop
        result = smc(current, xp.flip(fine_grid(cfg, level - 1, xp=xp, dtype=dtype, device=device), axis=0),
                     backend=backend, evaluate=evaluate, schedule=schedule, target=cfg.model_denoising,
                     proposal=cfg.model_denoising, potential=Zero(), policy=Never(), out_needs=cfg.replica_needs,
                     key=gap_keys[level - 1])
        levels[level - 1] = current = result.final
    return stack_levels(levels)                                      # [I, M+1, ...]


def crepe(cfg, n_instances, event_shape, n_blocks, emit, *, backend, evaluate, schedule, dtype, device=None, key,
          initial_ladder=None, thin=1, burn_in=0):
    """Run CREPE: replica exchange across noise levels, emitting a sample from level 0 after every `thin`-th sweep.

    Each sweep proposes path exchanges between adjacent levels (per `cfg.pairs`), accepts or rejects them per
    chain, applies the local move (if any), and refreshes the top replica with a fresh prior draw when the next
    sweep uses it. Samples are correlated MCMC output: `burn_in` blocks run first and emit nothing.

    A block is ``L = lcm(cfg.pairs.period, thin)`` sweeps, so the pair-schedule parity lines up across blocks.
    The `burn_in` blocks and the `n_blocks` emitting blocks (``L // thin`` emits each) are two scans, because
    whether a block emits is static. Thinning suits `crepe_finish`, where every kept
    sample costs `n_tail` evaluations and correlated particles overstate its effective sample size: set `thin`
    to about twice the integrated autocorrelation time (in sweeps) of a cheap emitted scalar such as the reward.
    For plain averages over CREPE's own samples, thinning only discards information.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    n_instances : int
        Number of independent chains.
    event_shape : tuple of int
        Event shape.
    n_blocks : int
        Number of emitting blocks, after the burn-in; each runs ``L = lcm(cfg.pairs.period, thin)`` sweeps.
    emit : Callable[[Record], Any]
        Maps the level-0 replicas (a Record of shape [I, 1, ...]) to any pytree of arrays, e.g. ``lambda r: r.x0``,
        ``lambda r: (r.x0, r.reward)``, or ``lambda r: r.project(finish_needs(cfg))`` for `crepe_finish`.
        Features it reads must be in ``cfg.out_needs`` (or otherwise in ``cfg.replica_needs``).
    backend : Backend
        Randomness and control flow; ``backend.xp`` creates the arrays.
    evaluate : Callable[[array, array, frozenset], Record]
        The model, e.g. from `make_evaluate_torch` or `make_evaluate_jax`.
    schedule : Schedule
        The noise schedule; its prior gives the top level.
    dtype : dtype
        Run dtype.
    device : device, optional
        Run device.
    key : backend key
        Root random key. Pass a fresh key when continuing from `initial_ladder`.
    initial_ladder : Record, optional
        Replicas from an earlier call (its returned `ladder`) to continue from, skipping `crepe_init`. Must carry
        ``cfg.replica_needs``; extras are dropped. Level times are the caller's responsibility.
    thin : int, optional
        Static, >= 1. Emit after every `thin`-th sweep only (sweeps ``thin, 2 thin, ...``); the sweeps in between
        still run. Default 1: emit after every sweep.
    burn_in : int, optional
        Static, >= 0. Blocks (L sweeps each) run before sampling, with nothing emitted or stored. They cost the
        same evaluations as any other block. A run continued from `initial_ladder` usually needs 0.

    Returns
    -------
    samples : pytree
        Stacked `emit` outputs: every leaf has shape [n_blocks, L // thin, I, 1, ...].
    ladder : Record
        The final replicas, shape [I, M+1, *D].

    Raises
    ------
    PartialRecord
        If `initial_ladder` lacks features in ``cfg.replica_needs``.
    ValueError
        If `thin` is not an int >= 1 or `burn_in` not an int >= 0.
    """
    if not isinstance(thin, int) or thin < 1:
        raise ValueError("crepe needs a static int thin >= 1")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("crepe needs a static int burn_in >= 0")
    xp = backend.xp
    key_init, key_run = backend.split(key, 2)                        # same split whether or not initial_ladder is given
    if initial_ladder is None:                                       # static
        ladder = crepe_init(cfg, n_instances, event_shape, backend=backend, evaluate=evaluate, schedule=schedule,
                            dtype=dtype, device=device, key=key_init)
    else:
        if not initial_ladder.provides(cfg.replica_needs):
            raise PartialRecord(cfg.replica_needs)
        ladder = initial_ladder.project(cfg.replica_needs)           # level times are the caller's responsibility
    evaluate_point = lambda point: evaluate_rows(evaluate, point, cfg.replica_needs, backend)
    path_exchange = PathExchange(cfg, evaluate, schedule)

    def top_used_in_sweep(sweep):                                    # static
        return any(cfg.n_gaps in pair
                   for pairs_round in cfg.pairs.rounds(cfg.n_gaps + 1, sweep) for pair in pairs_round)

    sweeps_per_block = math.lcm(cfg.pairs.period, thin)              # static: sweeps per scan body

    def block(ladder, key, emitting):                                # emitting: static
        emitted = []
        for sweep, sweep_key in enumerate(backend.split(key, sweeps_per_block)):
            key_exchange, key_local, key_refresh = backend.split(sweep_key, 3)
            ladder = replica_exchange(ladder, cfg.pairs, path_exchange, sweep=sweep, key=key_exchange,
                                      backend=backend)
            if cfg.local_move is not None:                           # static
                local_ladder = take_levels(ladder, cfg.local_levels)
                ctx = cfg.local_move(MoveContext(local_ladder, Zero().init(local_ladder), None, None),
                                     evaluate_point=evaluate_point, potential=Zero(), is_last=False, move_count=sweep,
                                     key=key_local, backend=backend)
                ladder = put_levels(ladder, cfg.local_levels, ctx.record)
            if top_used_in_sweep(sweep + 1):                         # refresh only if used next sweep
                top = top_replica(cfg, n_instances, event_shape, backend=backend, evaluate=evaluate,
                                  schedule=schedule, dtype=dtype, device=device, key=key_refresh)
                ladder = put_levels(ladder, [cfg.n_gaps], top)
            if emitting and (sweep + 1) % thin == 0:                 # static
                emitted.append(emit(take_levels(ladder, [0])))       # e.g. lambda r: r.x0
        return ladder, (tree_map(lambda *a: xp.stack(a), *emitted) if emitting else None)   # any pytree

    key_burn_in, key_sample = backend.split(key_run, 2)
    if burn_in > 0:                                                  # static
        ladder, _ = backend.scan(lambda ladder, key: block(ladder, key, False), ladder,
                                 backend.split(key_burn_in, burn_in))
    ladder, samples = backend.scan(lambda ladder, key: block(ladder, key, True), ladder,
                                   backend.split(key_sample, n_blocks))
    return samples, ladder


# ---- finishing from u_0 > 0 ------------------------------------------------------------------

def finish_needs(cfg):
    """Features `crepe_finish` reads from its start: use ``emit = lambda r: r.project(finish_needs(cfg))``.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.

    Returns
    -------
    frozenset of Feature
        What ``cfg.model_denoising`` and ``cfg.proposal_denoising`` require, plus REWARD.
    """
    return cfg.model_denoising.requires | cfg.proposal_denoising.requires | {Feature.REWARD}


def pool_samples(samples):
    """Pool `crepe`'s samples per chain: each chain becomes an instance, its samples that instance's particles.

    Parameters
    ----------
    samples : pytree
        `crepe`'s samples; every leaf has shape [n_blocks, E, I, 1, ...], with E emits per block. Burn-in is
        `crepe`'s job (its `burn_in`).

    Returns
    -------
    pytree
        Every leaf has shape [I, n_blocks * E, ...]: one instance per chain, its particles that chain's emitted
        samples in order (chains are never mixed); the level axis (size 1) is dropped.
    """
    def pool(a):
        xp = namespace(a)
        n_blocks, n_emits, n_instances = a.shape[0], a.shape[1], a.shape[2]
        flat = xp.reshape(a, (n_blocks * n_emits, n_instances) + tuple(a.shape[4:]))   # level axis (size 1) dropped
        return xp.moveaxis(flat, 0, 1)

    return tree_map(pool, samples)


def tail_grid(cfg, n_tail, *, xp, dtype, device=None):
    """The finish's grid: from exactly ``u_levels[0]`` down to exactly 0 in `n_tail` equal steps.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    n_tail : int
        Number of steps.
    xp : namespace
        Array namespace to create the grid with.
    dtype : dtype
        Grid dtype.
    device : device, optional
        Grid device.

    Returns
    -------
    array
        Schedule coordinates, shape [n_tail + 1], decreasing.
    """
    u0 = cfg.u_levels[0]
    return xp.asarray([u0 * (1 - j / n_tail) for j in range(n_tail)] + [0.0], dtype=dtype, device=device)


def crepe_finish(start, cfg, n_tail, *, exact, backend, evaluate, schedule, policy=None,
                 out_needs=frozenset({Feature.X0}), key) -> SMCResult:
    """Carry level-0 samples from ``u_levels[0] > 0`` down to u = 0 with one `smc` run over `n_tail` steps.

    Most callers use `crepe_sample`, which composes `crepe`, `pool_samples` and this function.

    Parameters
    ----------
    start : Record
        Level-0 samples with leading axes [I, N] at t = ``schedule.time(u_levels[0])``, e.g.
        ``pool_samples(samples)`` with ``emit = lambda r: r.project(finish_needs(cfg))``.
    cfg : CREPEConfig
        The configuration; ``u_levels[0]`` must be > 0.
    n_tail : int
        Number of steps from ``u_levels[0]`` to 0.
    exact : bool
        Static. True: weighted run with target ``cfg.model_denoising``, proposal ``cfg.proposal_denoising`` and
        potential ``Relative(FKDiff(cfg.lam))``, exact for ``p_0 exp(lambda(tau(0)) r_hat)``. False: unweighted
        guided denoising with ``cfg.proposal_denoising`` (``log_weights`` identically 0); cheaper and approximate.
    backend : Backend
        Randomness and control flow.
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    schedule : Schedule
        The noise schedule.
    policy : policy, optional
        Resampling policy for ``exact=True`` (default ``ESSBelow(0.5)``, within each instance). Ignored for
        ``exact=False``, which never resamples.
    out_needs : frozenset of Feature, optional
        Features the final particles carry. Default ``{Feature.X0}``.
    key : backend key
        Random key.

    Returns
    -------
    SMCResult
        The finished particles, [I, N, ...], and their log-weights.

    Raises
    ------
    ValueError
        If ``cfg.u_levels[0] <= 0``.
    """
    if cfg.u_levels[0] <= 0:
        raise ValueError("crepe_finish needs u_levels[0] > 0")
    xp = namespace(start.t)
    grid = tail_grid(cfg, n_tail, xp=xp, dtype=start.t.dtype, device=device_of(start.t))
    if exact:                                                        # static: weighted, exact target
        target, proposal = cfg.model_denoising, cfg.proposal_denoising
        potential = Relative(FKDiff(cfg.lam))
        policy = ESSBelow(0.5) if policy is None else policy
    else:                                                            # unweighted guided denoising
        target = proposal = cfg.proposal_denoising
        potential, policy = Zero(), Never()
    return smc(start, grid, backend=backend, evaluate=evaluate, schedule=schedule, target=target, proposal=proposal,
               potential=potential, policy=policy, out_needs=out_needs, key=key)



def crepe_sample(cfg, n_instances, event_shape, n_blocks, *, n_tail=None, exact=None, backend, evaluate, schedule,
                 dtype, device=None, key, initial_ladder=None, thin=1, burn_in=0, policy=None):
    """Run CREPE end to end: the chains, the pooling, and, when ``u_levels[0] > 0``, the finish to u = 0.

    Composes `crepe` (with an `emit` it chooses, so the finish never re-evaluates), `pool_samples` and
    `crepe_finish`. The result is an `SMCResult` either way: with ``u_levels[0] == 0`` there is no finish, `final` is
    the pooled level-0 samples and `log_weights` is exactly 0. Estimate with the self-normalised `log_weights` average.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    n_instances : int
        Number of independent chains.
    event_shape : tuple of int
        Event shape.
    n_blocks : int
        Number of emitting blocks, >= 1; see `crepe`.
    n_tail : int, optional
        Finish steps from ``u_levels[0]`` to 0. Required if ``u_levels[0] > 0``; ignored otherwise.
    exact : bool, optional
        Static. Weighted (True) or unweighted (False) finish; see `crepe_finish`. Required if ``u_levels[0] > 0``;
        ignored otherwise.
    backend : Backend
        Randomness and control flow; ``backend.xp`` creates the arrays.
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    schedule : Schedule
        The noise schedule.
    dtype : dtype
        Run dtype.
    device : device, optional
        Run device.
    key : backend key
        Random key; split in two, for `crepe` and `crepe_finish`.
    initial_ladder : Record, optional
        Replicas to continue from (a previous call's `ladder`); pass a fresh `key` and usually ``burn_in=0``.
    thin : int, optional
        See `crepe`.
    burn_in : int, optional
        See `crepe`.
    policy : policy, optional
        The finish's resampling policy; see `crepe_finish`.

    Returns
    -------
    result : SMCResult
        ``final``: [I, N, ...] at u = 0, carrying exactly ``cfg.out_needs``, with ``N = n_blocks * L // thin``
        (``L = lcm(cfg.pairs.period, thin)``); ``log_weights``: [I, N]. Each chain is one instance.
    ladder : Record
        `crepe`'s final replicas, shape [I, M+1, *D].

    Raises
    ------
    ValueError
        If ``n_blocks < 1``, or ``u_levels[0] > 0`` and `n_tail` or `exact` is missing.
    """
    xp = backend.xp
    if n_blocks < 1:
        raise ValueError("crepe_sample needs n_blocks >= 1")
    needs_finish = cfg.u_levels[0] > 0                               # static
    if needs_finish and (n_tail is None or exact is None):
        raise ValueError("u_levels[0] > 0: crepe_sample needs n_tail and exact")
    emit_needs = finish_needs(cfg) if needs_finish else cfg.out_needs   # both subsets of cfg.replica_needs
    key_chain, key_tail = backend.split(key, 2)
    samples, ladder = crepe(cfg, n_instances, event_shape, n_blocks, lambda r: r.project(emit_needs), backend=backend,
                            evaluate=evaluate, schedule=schedule, dtype=dtype, device=device, key=key_chain,
                            initial_ladder=initial_ladder, thin=thin, burn_in=burn_in)
    start = pool_samples(samples)                                    # [I, N, ...], one instance per chain
    if not needs_finish:                                             # static: level 0 is already u = 0
        return SMCResult(final=start, log_weights=xp.zeros_like(start.t)), ladder
    finished = crepe_finish(start, cfg, n_tail, exact=exact, backend=backend, evaluate=evaluate, schedule=schedule,
                            policy=policy, out_needs=cfg.out_needs, key=key_tail)
    return SMCResult(final=finished.final.project(cfg.out_needs), log_weights=finished.log_weights), ladder
