Residual preconditioned Crank-Nicolson Synthetic Example
========================================================

Installation
============

#. Install ``uv``:

   .. code:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh

#. Install Python dependencies using ``uv``:

   .. code:: bash

      uv sync

#. Change ``SAVE_DIR`` in ``ttg_1d.py`` and ``rpcn_sweep_1d.py`` to the directory where figures should be saved.
   It defaults to ``~/Overleaf/RpCN/imgs/toy_example``.

Supplementary Documentation
===========================

* `marimo <https://docs.marimo.io/>`_: Reactive Python notebook library used for the experiments in this project.
* `PyTorch <https://docs.pytorch.org/docs/stable/index.html>`_: Library used for the denoiser and sampling.

Running the experiments
=======================

The experiments are marimo notebooks.
Run them from the repository root, since ``style.mplstyle`` is loaded by a relative path.
Both notebooks use ``device='cuda'`` by default.
Experiment parameters (e.g., the RNG seed, number of GMM modes, reward tilt, and NFE budget) are set with the notebook's UI controls.

Steering Comparison
-------------------

Compares Feynman-Kac, DAS, CREPE, and Residual preconditioned Crank-Nicolson steering on a 1D Gaussian mixture model at a matched NFE budget.
The histograms are saved to ``{SAVE_DIR}/NiceHistogram``.

.. code:: bash

   uv run marimo edit ttg_1d.py

Residual preconditioned Crank-Nicolson Ablation
-----------------------------------------------

Sweeps the number of replicas and pCN steps of Residual preconditioned Crank-Nicolson.
The heatmaps are saved to ``{SAVE_DIR}/Ablation``.

.. code:: bash

   uv run marimo edit rpcn_sweep_1d.py
