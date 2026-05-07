Installation
============

Clone the repository and install using ``pip`` or ``uv``:

.. code:: bash

   pip install .

.. warning::

   The dependency versions for package have been lowered to maximize compatibility potential without verification of compatibility.
   We hope it is compatible with the versions of PyTorch or Jax that your project is using.

Usage
=====

We provide implementations of the guidance algorithms in PyTorch and Jax.

Source Parallel Tempering
-------------------------

Example usage of Source Parallel Tempering.
Note that this implementation assumes that the source distribution is at ``time = 0`` and the target distribution is at ``time = 1``.

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

Feynmann-Kac Steering
---------------------

Example usage of Feynmann-Kac Steering.
Note that this implementation assumes that the source distribution is at ``time = 0`` and the target distribution is at ``time = 1``.

.. code:: python

   import spt_guidance.torch.fk_steering as fk_steering
   # import spt_guidance.jax.fk_steering as fk_steering   # for Jax

   fk = fk_steering.FeynmannKac(
       init_param_fn=partial(torch.randn, data_dims, device=device),  # data_dims is shape of one data sample, not a batch
       reward_fn=reward_fn,  # function be be maximized (reward function)
       propose_fn=propose_fn,  # function that updates the current particles to the next time step
       resample_fn=fk_steering.ResampleFn.adaptive_resample,  # function to resample the particles based on their potential
       potential=fk_steering.Potential.MAX,  # how the potential of each particle is computed
       intermediate_reward=fk_steering.IntermediateReward.EXPECTED_SAMPLE,  # how the intermediate reward of each particle is computed
       inverse_temperature=5.,  # the tilt of the target distribution
       kwargs_intermediate_reward=dict(
           expected_sample_fn=lambda t0, z: propose_fn(t0, 1., z),  # function to return the expected final samples (at time=1) of the current particles
       ),
   )
   final_samples = fk.run(
      # key,  # PRNG key if using Jax
      times,  # sampling time interval discretization
      4,  # number of FK Steering chains to use
      batch_size=batch_size,  # number of target samples to generate at once
   )

Residual preconditioned Crank-Nicolson Steering
-----------------------------------------------

Example usage of Residual preconditioned Crank-Nicolson Steering (RpCN-Steering).
Note that this implementation assumes that the source distribution is at ``time = 0`` and the target distribution is at ``time = 1``.

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
           expected_sample_fn=expected_sample_fn,  # function to return the expected final samples (at time=1) of the current particles
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

Examples
========
We provide some examples of using the library in the ``examples`` directory.

Mean flow steering
------------------
This example contains code to train a mean flow model on 2D toy datasets and then guide it.
To train a mean flow model on the ``Circle`` toy dataset, run

   .. code:: bash

      python examples/guide_mean_flow/training_mean_flow/train_mean_flow_model.py --distribution Circle

Next, to apply source parallel tempering to the model, run

   .. code:: bash

      python examples/guide_mean_flow/steering_mean_flow/steer_model.py --guidance SPT --distribution Circle --save 0 --plot 1

and then view the results in ``examples/guide_mean_flow/steering_mean_flow/results/imgs``.

Installation for library development
====================================

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. Install the dependencies in ``pyproject.toml`` dependency-groups.
   For example,

   .. code:: bash

      uv pip install torch==2.9.1

#. Activate the Python virtual environment:

   .. code:: bash

      source .venv/bin/activate
