#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 4)

env_parallel --delay 0 --eta -j $((5 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/toy/main.py +experiment=GMM steering=FK steering.potential={potential} steering.n_chains={n_chains}' ::: potential MAX ::: n_chains 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
