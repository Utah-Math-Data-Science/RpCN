Test-Time Sequential Steering of Diffusion Models via Preconditioned Crank-Nicolson
===================================================================================

Code repository for `Test-Time Sequential Steering of Diffusion Models via Preconditioned Crank-Nicolson <https://openreview.net/forum?id=N7zXHlKXnl>`_.
Implementations for Residual preconditioned Crank-Nicolson (RpCN) and other test-time guidance methods are provided as an installable Python package (see `Usage`_).

The code for replicating the paper's experiments are on different branches of this Git repository:

   * `toy <https://github.com/Utah-Math-Data-Science/RpCN/tree/toy>`_: Code for **Section 5.1: Synthetic Expermiment**.
   * `image_gen <https://github.com/Utah-Math-Data-Science/RpCN/tree/image_gen>`_: Code for **Section 5.2: Conditional Image Generation**.
   * `dynamical-systems <https://github.com/Utah-Math-Data-Science/RpCN/tree/dynamical-systems>`_: Code for **Section 5.3: Dynamical Systems**.
   * `inverse_problems <https://github.com/Utah-Math-Data-Science/RpCN/tree/inverse_problems>`_: Code for **Section 5.4: Inverse Problems**.

Library Installation
====================

Clone the repository and install using ``pip`` or ``uv``:

.. code:: bash

   pip install .

.. warning::

   The dependency versions for PyTorch and Jax have not been specified to maximize compatibility potential without verification of compatibility.
   We hope it is compatible with the versions of PyTorch or Jax that your project is using.

Usage
====

Test-time guidance for diffusion models: FK steering, DAS, CREPE and RpCN in one codebase, written against
the Array API standard (2025.12). It runs on JAX, PyTorch (through `array-api-compat`) and `array_api_strict`.
The specification is `../impl.md`.

Examples
--------

Every method calls your model through one ``evaluate`` function. Build it from a noise-prediction network
``eps_net(x [B, *D], t [B]) -> eps`` and a reward ``reward(x0 [B, *D]) -> [B]``, both PyTorch callables
(``make_evaluate_jax`` and ``JaxBackend`` are the JAX equivalents). The builders take
``reward(x0, x, t) -> [B]``, so a reward may also read the noisy state and time; ``on_x0`` adapts a reward of
the estimate alone:

.. code:: python

   from ttg import (CREPEConfig, DDIMStep, Direction, ESSBelow, EulerMaruyama, EvenOdd, Every,
                    FKDiff, Guided, Langevin, LangevinMove, Never, Noising, PCN, PosteriorStep,
                    SequentialSweep, ReverseSDE, VPLinear, constant, crepe, crepe_sample,
                    das_lambda, ladder, level_lambda, make_evaluate_torch, on_x0, smc, tilt, x0_from_eps)
   from ttg.backend import TorchBackend

   bk = TorchBackend()
   xp = bk.xp                                        # array_api_compat.torch
   sch = VPLinear()
   D = (3, 32, 32)                                   # event shape of one sample
   I, N = 4, 16                                      # independent instances x particles
   evaluate = make_evaluate_torch(eps_net, lambda x, t, eps, s: (x0_from_eps(x, t, eps, s), eps), sch,
                                  on_x0(reward))

   u = xp.linspace(1.0, 0.0, 100, dtype=xp.float32)  # schedule coordinate, noise (1) to data (0)
   P = PosteriorStep(sch)
   start = sch.prior((I, N), D, xp=xp, dtype=xp.float32)
   key = bk.key(0)

**FK steering**

.. code:: python

   fk = smc(start, u, bk=bk, evaluate=evaluate, schedule=sch, target=P, proposal=P,
            potential=FKDiff(constant(10.0)), policy=Every(5), key=key)
   x0, log_w = fk.final.x0, fk.log_w                 # [I, N, *D] samples, [I, N] log-weights

**DAS**

.. code:: python

   lam = das_lambda(sch.time(u), gamma=0.008, alpha_kl=0.01)
   das = smc(start, u, bk=bk, evaluate=evaluate, schedule=sch, target=P, proposal=Guided(P, tilt(lam)),
             potential=FKDiff(lam), policy=ESSBelow(0.5), key=key)

**RpCN** — the particle axis is a temperature ladder :math:`0 = \beta_0 < \dots < \beta_K = \beta`;
there is no resampling.

.. code:: python

   K, beta = 3, 10.0
   betas = xp.linspace(0.0, beta, K + 1, dtype=xp.float32)
   P = DDIMStep(sch, eta=1.0)
   rpcn = smc(sch.prior((I, K + 1), D, xp=xp, dtype=xp.float32), u, bk=bk, evaluate=evaluate, schedule=sch,
              target=P, proposal=P, potential=FKDiff(ladder(betas)), policy=Never(),
              move=PCN(n_steps=20, step=constant(0.95), pairs=SequentialSweep()),
              key=key)
   coldest = rpcn.final.x0[:, K]                     # [I, *D]: the beta_K = beta replica

**CREPE** — replica exchange across noise levels :math:`0 = u_0 < \dots < u_M = 1`.

.. code:: python

   u_levels = (0.0, 0.25, 0.5, 0.75, 1.0)
   lam = level_lambda(sch.time(xp.asarray(u_levels, dtype=xp.float32)), beta=10.0, rho=1.0)
   B_P = EulerMaruyama(ReverseSDE(sch), Direction.BACKWARD)
   cfg = CREPEConfig(u_levels=u_levels, K=10, F_P=Noising(sch), F_Q=Noising(sch), B_P=B_P,
                     B_Q=Guided(B_P, tilt(lam)), lam=lam,
                     local_move=LangevinMove(Guided(Langevin(h=lambda t: 0.01 * sch.std(t) ** 2), tilt(lam, 0.5))),
                     local_levels=(0, 1, 2, 3))
   samples, replicas = crepe(cfg, I, D, 100, lambda r: r.x0, bk=bk, evaluate=evaluate, schedule=sch,
                             dtype=xp.float32, key=key)   # samples: [100, period, I, 1, *D]
   more, replicas = crepe(cfg, I, D, 100, lambda r: r.x0, bk=bk, evaluate=evaluate, schedule=sch,
                          dtype=xp.float32, key=bk.key(1), R0=replicas)   # continue: no re-initialisation

``crepe_sample`` runs CREPE end to end and returns an ``SMCResult`` with one instance per chain. With the lowest
level above the data end (:math:`u_0 > 0`), it also finishes the level-0 samples the rest of the way to
:math:`u = 0`, weighted (``exact=True``) or as plain guided denoising (``exact=False``):

.. code:: python

   cfg = CREPEConfig(u_levels=(0.25, 0.5, 0.75, 1.0), ...)       # as above, with lam on these levels
   out, replicas = crepe_sample(cfg, I, D, 80, n_tail=10, exact=False, bk=bk, evaluate=evaluate, schedule=sch,
                                dtype=xp.float32, key=key, thin=4, burn_in=20)   # 80 samples, 1 per 4 sweeps
   x0, log_w = out.final.x0, out.log_w                           # [I, 80, *D], [I, 80]

``crepe``, ``pool_samples`` and ``crepe_finish`` remain available for custom ``emit`` functions. ``thin`` keeps
every ``thin``-th sweep's sample, so the particles the finish pays ``n_tail`` evaluations for are closer to
independent; set it to about twice the autocorrelation time of a cheap emitted scalar such
as the reward. ``burn_in`` blocks run first and store nothing; a run continued with ``R0`` usually needs none.

The results are ``torch.Tensor`` objects. On JAX, the same calls can be wrapped in ``jax.jit``.
