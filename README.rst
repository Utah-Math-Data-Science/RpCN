Residual preconditioned Crank-Nicolson Toy Example
==================================================

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

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

Evaluating the models
=====================

Below are example experiments of generating samples using a steering algorithm.
By default, the experiment logs samples to ``{Conf.out_dir}/runs/{Conf.alt_id}/samples.csv``.

Metropolis-Hastings Reference
-----------------------------

.. code:: bash

   python src/toy/main.py +experiment=GMM steering=MH

Residual preconditioned Crank-Nicolson
--------------------------------------

.. code:: bash

   python src/toy/main.py +experiment=GMM steering=RpCN steering.pcn_update_count=5 steering.n_chains=4

Feynmann-Kac
------------

.. code:: bash

   python src/toy/main.py +experiment=GMM steering=FK steering.potential=DIFFERENCE steering.n_chains=20
