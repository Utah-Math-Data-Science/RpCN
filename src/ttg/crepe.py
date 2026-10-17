"""CREPE (§16): replicas at levels u_0 < ... < u_M = 1, path exchanges between adjacent levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .evaluate import evaluate_rows, start_record
from .exchange import EvenOdd, replica_exchange
from .moves import MoveContext
from .ops import device, namespace, put_levels, stack_levels, take_levels
from .potentials import ReverseRatio, Sum, TiltEnds, Zero
from .records import Feature
from .resample import Never
from .smc import smc


@dataclass(frozen=True)
class CREPEConfig:
    """Configuration of CREPE: replicas at noise levels u_0 < ... < u_M = 1, exchanging paths between neighbours.

    Level 0 (u_0, usually 0) is the data end and holds the tilted target; level M is the prior. A path exchange
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


def crepe(cfg, I, D, n_blocks, emit, *, bk, evaluate, schedule, dtype, device=None, key):
    """Run CREPE: replica exchange across noise levels, emitting a sample from level 0 after every sweep.

    Each sweep proposes path exchanges between adjacent levels (per `cfg.pairs`), accepts or rejects them per
    chain, applies the local move (if any), and refreshes the top replica with a fresh prior draw when the next
    sweep uses it. Samples are correlated MCMC output; discard a burn-in.

    Parameters
    ----------
    cfg : CREPEConfig
        The configuration.
    I : int
        Number of independent chains.
    D : tuple of int
        Event shape.
    n_blocks : int
        Number of scan iterations; each runs ``cfg.pairs.period`` sweeps.
    emit : Callable[[Record], array]
        Maps the level-0 replicas (a Record of shape [I, 1, ...]) to the output, e.g. ``lambda r: r.x0``.
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
        Root random key.

    Returns
    -------
    samples : array
        Stacked `emit` outputs, shape [n_blocks, period, I, 1, ...].
    R : Record
        The final replicas, shape [I, M+1, *D].
    """
    xp = bk.xp
    k_init, k_run = bk.split(key, 2)
    R = crepe_init(cfg, I, D, bk=bk, evaluate=evaluate, schedule=schedule, dtype=dtype, device=device, key=k_init)
    ev = lambda p: evaluate_rows(evaluate, p, cfg.replica_needs, bk)
    xchg = PathExchange(cfg, evaluate, schedule)

    def top_used(n):                                                 # static
        return any(cfg.M in p for rnd in cfg.pairs.rounds(cfg.M + 1, n) for p in rnd)

    def body(R, key):
        outs = []
        for n, kk in zip(range(cfg.pairs.period), bk.split(key, cfg.pairs.period)):
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
            outs.append(emit(take_levels(R, [0])))                   # e.g. lambda r: r.x0
        return R, xp.stack(outs)

    R, samples = bk.scan(body, R, bk.split(k_run, n_blocks))
    return samples, R

