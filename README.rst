Residual preconditioned Crank-Nicolson for Dynamical Systems
===============================================

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. Install ``cuDNN``:

   .. code:: bash

      apt-get install -y cudnn9-cuda-12

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Change ``Conf.out_dir`` in ``conf/conf.py`` to the directory where the outputs of all experiments should be stored.

#. Create a ``runs`` directory inside ``Conf.out_dir``.

Supplementary Documentation
===========================

* `Hydra <https://hydra.cc/docs/1.3/intro/>`_: Command-line inferface configuration library for configuring the experiments in this project.
* `Hydra ORM <https://github.com/reepoi/hydra-orm>`_: Library for saving experiment configurations to an `SQLite <https://sqlite.org/>`_ database.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/2.6.0/>`_: Library for handling model training and evaluation loops.

Training the models
===================

Example commands to train the generative models:

.. code:: bash

   # Lorenz '63
   python src/userfm/main.py +experiment=TrainLorenz63 model/probability_path=VarianceExploding
   # FitzHugh-Nagumo
   python src/userfm/main.py +experiment=TrainFitzHughNagumo model/probability_path=VarianceExploding

The trained model will be saved in ``{Conf.out_dir}/runs/{Conf.alt_id}`` where ``Conf.alt_id`` is a randomly generated string.
The ``Conf.alt_id`` uniquely identifies the model training experiment that was run, and it is needed to evaluate the model.

Evaluating the models
=====================

Below are example experiments of generating samples using a trained model and a steering algorithm.
The command override ``model.conf={Conf.alt_id}`` specifies that trained model to use.
By default, the experiment logs the number of trajectories per batch generated that are event trajectories.
To log the constraint value of each trajectory generated, add the override ``log_constraint_values=true``.
Logs are stored in ``{Conf.out_dir}/runs/{Conf.alt_id}/metrics.csv``.

No Steering
-----------

.. code:: bash

   python src/userfm/main.py +experiment=SteerLorenz63 steering=NoSteering model.conf=o53qnpz8
   python src/userfm/main.py +experiment=SteerFitzHughNagumo steering=NoSteering model.conf=aezlb9vb

Residual preconditioned Crank-Nicolson
--------------------------------------

.. code:: bash

   python src/userfm/main.py +experiment=SteerLorenz63 steering=RpCN steering.chain_count=8 steering.pcn_update_count=8 steering/intermediate_reward=FeynmannKacIntermediateRewardExpectedSample steering.tilt=8.29 model.conf=o53qnpz8
   python src/userfm/main.py +experiment=SteerFitzHughNagumo steering=RpCN steering.chain_count=8 steering.pcn_update_count=8 steering/intermediate_reward=FeynmannKacIntermediateRewardExpectedSample steering.tilt=10.27 model.conf=aezlb9vb

Feynmann-Kac
------------

.. code:: bash

   python src/userfm/main.py +experiment=SteerLorenz63 steering=FeynmannKac steering.ensemble_size=8 steering.potential=DIFFERENCE steering/intermediate_reward=FeynmannKacIntermediateRewardExpectedSample steering.tilt=8.29 model.conf=o53qnpz8
   python src/userfm/main.py +experiment=SteerFitzHughNagumo steering=FeynmannKac steering.ensemble_size=8 steering.potential=DIFFERENCE steering/intermediate_reward=FeynmannKacIntermediateRewardExpectedSample steering.tilt=10.27 model.conf=aezlb9vb
