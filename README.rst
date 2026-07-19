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
=====

We provide implementations of the guidance algorithms in PyTorch and Jax.

Residual preconditioned Crank-Nicolson
--------------------------------------

Example usage of Residual preconditioned Crank-Nicolson (RpCN).

.. code:: python

   import spt_guidance.torch.rpcn as rpcn
   # import spt_guidance.jax.rpcn as rpcn  # for Jax

   kernel = rpcn.KernelPreconditionedCrankNicolson(
      data_dims  # shape of one data sample, not a batch
   )
   rpcn = rpcn.ResidualMCMCSteering(
       kernel,
       torch.linspace(torch.pi / 2 - 1e-3, 0.05, num_chains, device=device).sin(),  # pCN kernel step size
       reward_fn=reward_fn,  # function be be maximized (reward function)
       expected_next_fn=expected_next_fn,  # function that computes the next state's mean given the current state
       std_fn=std_fn,  # function that returns the standard deviation of each dimension of the current or next state (see docstring)
       intermediate_reward=rpcn.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
       inverse_temperature=torch.linspace(0., 5., num_chains, device=device),  # inverse temperatures
       kwargs_intermediate_reward=dict(
           expected_sample_fn=expected_sample_fn,  # function to return the expected final samples of the current particles
       ),
   )
   final_samples = rpcn.run(
      # key,  # PRNG key if using Jax
      times,  # sampling time interval discretization
      pcn_should_update_fn=pcn_should_update_fn,  # function to indicate whether the pCN updates should run
      pcn_should_replica_exchange_fn=pcn_should_replica_exchange_fn,  # function to indicate whether to do replica exchange
      pcn_update_count=pcn_update_count,  # number of pcn steps to take per update
      batch_size=batch_size,  # number of target samples to generate at once
   )

Source Parallel Tempering
-------------------------

Example usage of Source Parallel Tempering.

.. code:: python

   import spt_guidance.torch.spt as spt
   # import spt_guidance.jax.spt as spt   # for Jax

   kernel = spt.KernelPreconditionedCrankNicolson(
      data_dims  # shape of one data sample, not a batch
   )
   ptmcmc = spt.PTMCMC(
       kernel,
       potential_fn,  # function to be minimized (negative reward function)
       torch.linspace(0., 5., num_chains, device=device),  # inverse temperatures
       torch.linspace(torch.pi / 2 - 1e-3, 0.05, num_chains, device=device).sin(),  # pCN kernel step size
   )
   last_chain = ptmcmc.run(
      # key,  # PRNG key if using Jax
      100,  # number of iterations (tmax) to run SPT for
      batch_size=batch_size,  # number of source samples to generate at once
   )[-1]  # take the last (coldest) chain
   final_samples = transport(last_chain)  # map source samples to the target distribution

Feynman-Kac Steering
--------------------

Example usage of Feynman-Kac Steering.

.. code:: python

   import spt_guidance.torch.fk_steering as fk_steering
   # import spt_guidance.jax.fk_steering as fk_steering   # for Jax

   fk = fk_steering.FeynmanKac(
       init_param_fn=partial(torch.randn, data_dims, device=device),  # data_dims is shape of one data sample, not a batch
       reward_fn=reward_fn,  # function be be maximized (reward function)
       propose_fn=propose_fn,  # function that updates the current particles to the next time step
       resample_fn=fk_steering.ResampleFn.adaptive_resample,  # function to resample the particles based on their potential
       potential=fk_steering.Potential.MAX,  # how the potential of each particle is computed
       intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
       inverse_temperature=5.,  # the tilt of the target distribution
       kwargs_intermediate_reward=dict(
           expected_sample_fn=lambda t0, z: propose_fn(t0, 1., z),  # function to return the expected final samples of the current particles
       ),
   )
   final_particles, final_log_weights = fk.run(
       # key,  # PRNG key if using Jax
       times,  # sampling time interval discretization
       4,  # number of FK Steering chains to use
       batch_size=batch_size,  # number of target samples to generate at once
   )
   final_samples = fk_steering.SelectFn.highest_reward(final_particles, reward_fn)
