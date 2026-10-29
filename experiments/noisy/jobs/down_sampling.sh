#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
source ~/.bashrc
cd "$repo_root"

job_slot_to_gpu=(-1 1 2 3 4 5 6 7)

CUDA_VISIBLE_DEVICES=${MEASUREMENT_GPU:-0} python -m experiments.noisy.make_measurements \
    --operators down_sampling --noise_sigma 0.05

env_parallel \
    --delay 3 \
    --eta \
    -j $((${#job_slot_to_gpu[@]} - 1)) \
    --header : \
    --colsep , \
    CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' \
    python -m experiments.noisy.rpcn \
        --save \
        --plot \
        --seed 31 \
        --img_id {id} \
        --operator down_sampling \
        --n_chains 3 \
        --resampling_frequency 50 \
        --pcn_update_count 3 \
        --operator_seed 42 \
        --measurement_seed 42 \
        --noise_sigma 0.05 \
    :::: "$repo_root/ids.csv"
