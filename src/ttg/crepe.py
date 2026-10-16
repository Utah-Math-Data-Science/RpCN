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
        return len(self.u_levels) - 1

    @property
    def replica_needs(self):
        reqs = [k.requires for k in (self.F_P, self.B_P, self.F_Q, self.B_Q)]
        reqs += [self.local_move.requires] if self.local_move else []
        return frozenset().union(*reqs) | {Feature.REWARD} | frozenset(self.out_needs)


def fine_grid(cfg, m, *, xp, dtype, device=None):
    """Gap m: [K+1] ascending; endpoints exactly u_m, u_{m+1}."""
    u0, u1 = cfg.u_levels[m], cfg.u_levels[m + 1]
    pts = [u0, *(u0 + (u1 - u0) * j / cfg.K for j in range(1, cfg.K)), u1]
    return xp.asarray(pts, dtype=dtype, device=device)


def per_row_grid(cfg, gaps, I, ascending, *, xp, dtype, device=None):
    """-> [I, P, K+1]."""
    g = xp.stack([fine_grid(cfg, m, xp=xp, dtype=dtype, device=device) for m in gaps])
    g = g if ascending else xp.flip(g, axis=-1)
    return xp.broadcast_to(g[None, ...], (I, *g.shape))


@dataclass(frozen=True)
class PathExchange:
    """Two one-directional `smc` calls per round: up from level lo, down from level hi."""

    cfg: CREPEConfig
    evaluate: Callable
    schedule: Any

    def propose(self, R, lo, hi, key, bk):           # pair (lo[p], hi[p] = lo[p] + 1) is gap lo[p]
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
    """A fresh level-M replica: a prior draw, evaluated with replica_needs."""
    xp = bk.xp
    prior = schedule.prior((I, 1), D, xp=xp, dtype=dtype, device=device)
    t1 = schedule.time(xp.ones((I, 1), dtype=dtype, device=device))
    return start_record(prior, t1, cfg.replica_needs, evaluate, bk, key)[0]


def crepe_init(cfg, I, D, *, bk, evaluate, schedule, dtype, device=None, key):
    """M chained `smc` calls with the model's own kernel. -> replicas [I, M+1, ...]."""
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
    """-> (samples [n_blocks, period, I, 1, ...] from emit(level-0 replicas), final replicas)."""
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

