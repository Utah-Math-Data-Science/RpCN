"""CREPE (§16): replicas at levels u_0 < ... < u_M = 1, path exchanges between adjacent levels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .evaluate import evaluate_rows, start_record
from .exchange import EvenOdd, replica_exchange
from .moves import MoveContext
from .ops import device, namespace, put_levels, stack_levels, take_levels, tree_map
from .potentials import FKDiff, Relative, ReverseRatio, Sum, TiltEnds, Zero
from .records import Feature, PartialRecord
from .resample import ESSBelow, Never
from .smc import SMCResult, smc


@dataclass(frozen=True)
class CREPEConfig:
    """Configuration of CREPE: replicas at noise levels u_0 < ... < u_M = 1, exchanging paths between neighbours.

    Level 0 (u_0, usually 0) is the data end and holds the tilted target; level M is the prior. If u_0 > 0, level 0
    samples ``p_{u_0} exp(lambda(tau(u_0)) r_hat)`` and `crepe_finish` carries its samples to u = 0. A path exchange
    between levels m and m+1 runs `K` fine steps across the gap: up (towards noise) with `F_Q` as the proposal
    and `F_P` as the target, and down (towards data) with `B_Q` as the proposal and `B_P` as the target.

    Parameters
    ----------
    u_levels : tuple of float
        Increasing schedule coordinates of the levels; ``u_levels[-1]`` must be 1.
    K : int
        Number of fine steps per gap between adjacent levels.
    F_P : Kernel
        Forward (noising) kernel of the reference process, e.g. ``Noising(schedule)``.
    B_P : Kernel
        Backward (denoising) kernel of the model, e.g. ``EulerMaruyama(ReverseSDE(schedule), Direction.BACKWARD)``.
    F_Q : Kernel
        Forward proposal kernel. Passing the same object as `F_P` makes the forward terms cancel statically.
    B_Q : Kernel
        Backward proposal kernel, e.g. ``Guided(B_P, tilt(lam))``.
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
    K: int                               # fine steps per gap
    F_P: Any
    B_P: Any
    F_Q: Any
    B_Q: Any
    lam: Callable                        # e.g. level_lambda(time(u_levels), beta, rho)
    pairs: Any = EvenOdd()
    local_move: Any = None               # e.g. LangevinMove(Guided(Langevin(h), tilt(lam, 0.5)))
    local_levels: tuple = ()             # static subset of 0..M-1
    out_needs: frozenset = frozenset({Feature.X0})

    @property
    def M(self):
        """Index of the top level: the number of levels minus one."""
        return len(self.u_levels) - 1

    @property
    def replica_needs(self):
        """Features every replica carries.

        These are the features the four kernels and the local move require, plus REWARD and `out_needs`.
        """
        reqs = [k.requires for k in (self.F_P, self.B_P, self.F_Q, self.B_Q)]
        reqs += [self.local_move.requires] if self.local_move else []
        return frozenset().union(*reqs) | {Feature.REWARD} | frozenset(self.out_needs)


def fine_grid(cfg, m, *, xp, dtype, device=None):
    """Return the fine grid across gap m, ascending.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    m : int
        Gap index in 0..M-1, between levels m and m+1.
    xp : namespace
        Array namespace to create the grid with.
    dtype : dtype
        Grid dtype.
    device : device, optional
        Grid device.

    Returns
    -------
    array
        Schedule coordinates, shape [K+1]; the endpoints are exactly ``u_levels[m]`` and ``u_levels[m+1]``.
    """
    u0, u1 = cfg.u_levels[m], cfg.u_levels[m + 1]
    pts = [u0, *(u0 + (u1 - u0) * j / cfg.K for j in range(1, cfg.K)), u1]
    return xp.asarray(pts, dtype=dtype, device=device)


def per_row_grid(cfg, gaps, I, ascending, *, xp, dtype, device=None):
    """Per-row fine grids for several gaps at once, as `smc` takes them.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    gaps : sequence of int
        The P gap indices.
    I : int
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
    g = xp.stack([fine_grid(cfg, m, xp=xp, dtype=dtype, device=device) for m in gaps])
    g = g if ascending else xp.flip(g, axis=-1)
    return xp.broadcast_to(g[None, ...], (I, *g.shape))


@dataclass(frozen=True)
class PathExchange:
    """CREPE's exchange: two one-directional `smc` calls per round, up from level lo and down from level hi.

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

    def propose(self, R, lo, hi, key, bk):           # pair (lo[p], hi[p] = lo[p] + 1) is gap lo[p]
        """Propose swapped replicas for the pairs (lo[p], hi[p]).

        The lower replica's path is pushed up across the gap and the upper replica's path down; each run is
        weighted by the endpoint tilts and the reverse-kernel ratio.

        Parameters
        ----------
        R : Record
            Replicas, shape [I, M+1, ...].
        lo : sequence of int
            Lower levels of the P pairs.
        hi : sequence of int
            Upper levels, ``hi[p] = lo[p] + 1``.
        key : backend key
            Random key.
        bk : Backend
            Randomness and control flow.

        Returns
        -------
        new_lo : Record
            Candidates for the lower levels (the end of the downward path), shape [I, P, ...].
        new_hi : Record
            Candidates for the upper levels (the end of the upward path), shape [I, P, ...].
        log_a : array
            Log acceptance ratio, shape [I, P].
        """
        c = self.cfg
        k_up, k_dn = bk.split(key, 2)
        xp = namespace(R.t)
        I, dtype, dev = R.t.shape[0], R.t.dtype, device(R.t)

        def pot(P_rev, Q_rev):
            return Sum(TiltEnds(c.lam), ReverseRatio(P_rev, Q_rev))

        def run(start, grid, tgt, prop, p, k):
            return smc(start, grid, bk=bk, evaluate=self.evaluate, schedule=self.schedule, target=tgt,
                       proposal=prop, potential=p, policy=Never(), final_needs=c.replica_needs, key=k)

        def grid(asc):
            return per_row_grid(c, lo, I, asc, xp=xp, dtype=dtype, device=dev)

        up = run(take_levels(R, lo), grid(True), c.F_P, c.F_Q, pot(c.B_P, c.B_Q), k_up)
        dn = run(take_levels(R, hi), grid(False), c.B_P, c.B_Q, pot(c.F_P, c.F_Q), k_dn)
        return dn.final, up.final, up.log_w + dn.log_w           # lower <- z_0, upper <- y_K


def top_replica(cfg, I, D, *, bk, evaluate, schedule, dtype, device, key):
    """Draw a fresh top-level replica from the prior at u = 1, evaluated with ``cfg.replica_needs``.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    I : int
        Number of instances.
    D : tuple of int
        Event shape.
    bk : Backend
        Randomness and control flow; ``bk.xp`` creates the arrays.
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
    xp = bk.xp
    prior = schedule.prior((I, 1), D, xp=xp, dtype=dtype, device=device)
    t1 = schedule.time(xp.ones((I, 1), dtype=dtype, device=device))
    return start_record(prior, t1, cfg.replica_needs, evaluate, bk, key)[0]


def crepe_init(cfg, I, D, *, bk, evaluate, schedule, dtype, device=None, key):
    """Build the initial replicas: a prior draw at the top, carried down with the model's kernel `B_P`.

    Runs M chained, unweighted `smc` calls, one per gap from M-1 down to 0.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    I : int
        Number of independent chains.
    D : tuple of int
        Event shape.
    bk : Backend
        Randomness and control flow; ``bk.xp`` creates the arrays.
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
    xp = bk.xp
    k0, k1, *ks = bk.split(key, cfg.M + 2)
    top = top_replica(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule, dtype=dtype, device=device, key=k0)
    levels, cur = [None] * cfg.M + [top], top
    for m in range(cfg.M, 0, -1):                                    # static loop
        res = smc(cur, xp.flip(fine_grid(cfg, m - 1, xp=xp, dtype=dtype, device=device), axis=0),
                  bk=bk, evaluate=evaluate, schedule=schedule, target=cfg.B_P, proposal=cfg.B_P,
                  potential=Zero(), policy=Never(), final_needs=cfg.replica_needs, key=ks[m - 1])
        levels[m - 1] = cur = res.final
    return stack_levels(levels)                                      # [I, M+1, ...]


def crepe(cfg, I, D, n_blocks, emit, *, bk, evaluate, schedule, dtype, device=None, key, R0=None, thin=1,
          burn_in=0):
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
    I : int
        Number of independent chains.
    D : tuple of int
        Event shape.
    n_blocks : int
        Number of emitting blocks, after the burn-in; each runs ``L = lcm(cfg.pairs.period, thin)`` sweeps.
    emit : Callable[[Record], Any]
        Maps the level-0 replicas (a Record of shape [I, 1, ...]) to any pytree of arrays, e.g. ``lambda r: r.x0``,
        ``lambda r: (r.x0, r.reward)``, or ``lambda r: r.project(finish_needs(cfg))`` for `crepe_finish`.
        Features it reads must be in ``cfg.out_needs`` (or otherwise in ``cfg.replica_needs``).
    bk : Backend
        Randomness and control flow; ``bk.xp`` creates the arrays.
    evaluate : Callable[[array, array, frozenset], Record]
        The model, e.g. from `make_evaluate_torch` or `make_evaluate_jax`.
    schedule : Schedule
        The noise schedule; its prior gives the top level.
    dtype : dtype
        Run dtype.
    device : device, optional
        Run device.
    key : backend key
        Root random key. Pass a fresh key when continuing from `R0`.
    R0 : Record, optional
        Replicas from an earlier call (its returned `R`) to continue from, skipping `crepe_init`. Must carry
        ``cfg.replica_needs``; extras are dropped. Level times are the caller's responsibility.
    thin : int, optional
        Static, >= 1. Emit after every `thin`-th sweep only (sweeps ``thin, 2 thin, ...``); the sweeps in between
        still run. Default 1: emit after every sweep.
    burn_in : int, optional
        Static, >= 0. Blocks (L sweeps each) run before sampling, with nothing emitted or stored. They cost the
        same evaluations as any other block. A run continued from `R0` usually needs 0.

    Returns
    -------
    samples : pytree
        Stacked `emit` outputs: every leaf has shape [n_blocks, L // thin, I, 1, ...].
    R : Record
        The final replicas, shape [I, M+1, *D].

    Raises
    ------
    PartialRecord
        If `R0` lacks features in ``cfg.replica_needs``.
    ValueError
        If `thin` is not an int >= 1 or `burn_in` not an int >= 0.
    """
    if not isinstance(thin, int) or thin < 1:
        raise ValueError("crepe needs a static int thin >= 1")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("crepe needs a static int burn_in >= 0")
    xp = bk.xp
    k_init, k_run = bk.split(key, 2)                                 # same split whether or not R0 is given
    if R0 is None:                                                   # static
        R = crepe_init(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule, dtype=dtype, device=device,
                       key=k_init)
    else:
        if not R0.provides(cfg.replica_needs):
            raise PartialRecord(cfg.replica_needs)
        R = R0.project(cfg.replica_needs)                            # level times are the caller's responsibility
    ev = lambda p: evaluate_rows(evaluate, p, cfg.replica_needs, bk)
    xchg = PathExchange(cfg, evaluate, schedule)

    def top_used(n):                                                 # static
        return any(cfg.M in p for rnd in cfg.pairs.rounds(cfg.M + 1, n) for p in rnd)

    L = math.lcm(cfg.pairs.period, thin)                             # static: sweeps per scan body

    def block(R, key, emitting):                                     # emitting: static
        outs = []
        for n, kk in enumerate(bk.split(key, L)):
            k_x, k_l, k_r = bk.split(kk, 3)
            R = replica_exchange(R, cfg.pairs, xchg, sweep=n, key=k_x, bk=bk)
            if cfg.local_move is not None:                           # static
                sub = take_levels(R, cfg.local_levels)
                ctx = cfg.local_move(MoveContext(sub, Zero().init(sub), None, None), ev=ev, potential=Zero(),
                                     last=False, count=n, key=k_l, bk=bk)
                R = put_levels(R, cfg.local_levels, ctx.rec)
            if top_used(n + 1):                                      # refresh only if used next sweep
                top = top_replica(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule,
                                  dtype=dtype, device=device, key=k_r)
                R = put_levels(R, [cfg.M], top)
            if emitting and (n + 1) % thin == 0:                     # static
                outs.append(emit(take_levels(R, [0])))               # e.g. lambda r: r.x0
        return R, (tree_map(lambda *a: xp.stack(a), *outs) if emitting else None)   # any pytree

    k_burn, k_samp = bk.split(k_run, 2)
    if burn_in > 0:                                                  # static
        R, _ = bk.scan(lambda R, k: block(R, k, False), R, bk.split(k_burn, burn_in))
    R, samples = bk.scan(lambda R, k: block(R, k, True), R, bk.split(k_samp, n_blocks))
    return samples, R


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
        What ``cfg.B_P`` and ``cfg.B_Q`` require, plus REWARD.
    """
    return cfg.B_P.requires | cfg.B_Q.requires | {Feature.REWARD}


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
        nb, E, I = a.shape[0], a.shape[1], a.shape[2]
        return xp.moveaxis(xp.reshape(a, (nb * E, I) + tuple(a.shape[4:])), 0, 1)   # level axis (size 1) dropped

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


def crepe_finish(start, cfg, n_tail, *, exact, bk, evaluate, schedule, policy=None,
                 final_needs=frozenset({Feature.X0}), key) -> SMCResult:
    """Carry level-0 samples from ``u_levels[0] > 0`` down to u = 0 with one `smc` run over `n_tail` steps.

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
        Static. True: weighted run with target ``cfg.B_P``, proposal ``cfg.B_Q`` and potential
        ``Relative(FKDiff(cfg.lam))``, exact for ``p_0 exp(lambda(tau(0)) r_hat)``. False: unweighted guided
        denoising with ``cfg.B_Q`` (``log_w`` identically 0); cheaper and approximate.
    bk : Backend
        Randomness and control flow.
    evaluate : Callable[[array, array, frozenset], Record]
        The model.
    schedule : Schedule
        The noise schedule.
    policy : policy, optional
        Resampling policy for ``exact=True`` (default ``ESSBelow(0.5)``, within each instance). Ignored for
        ``exact=False``, which never resamples.
    final_needs : frozenset of Feature, optional
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
    grid = tail_grid(cfg, n_tail, xp=xp, dtype=start.t.dtype, device=device(start.t))
    if exact:                                                        # static: weighted, exact target
        target, proposal = cfg.B_P, cfg.B_Q
        potential = Relative(FKDiff(cfg.lam))
        policy = ESSBelow(0.5) if policy is None else policy
    else:                                                            # unweighted guided denoising
        target = proposal = cfg.B_Q
        potential, policy = Zero(), Never()
    return smc(start, grid, bk=bk, evaluate=evaluate, schedule=schedule, target=target, proposal=proposal,
               potential=potential, policy=policy, final_needs=final_needs, key=key)

