#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 4)

env_parallel --delay 0 --eta -j $((20 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/toy/main.py +experiment=GMM steering=RpCN steering.pcn_update_count={pcn_update_count} steering.n_chains={n_chains}' ::: pcn_update_count 1 2 4 8 16 32 ::: n_chains 1 2 4 8
