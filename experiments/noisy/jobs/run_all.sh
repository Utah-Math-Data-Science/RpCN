#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
source ~/.bashrc
cd "$repo_root"

job_slot_to_gpu=(-1 1 4 6 7)

CUDA_VISIBLE_DEVICES=${MEASUREMENT_GPU:-0} python -m experiments.noisy.make_measurements \
    --noise_sigma 0.05 \
    --operator_seed 42 \
    --measurement_seed 42

env_parallel \
    --delay 3 \
    --eta \
    -j $((${#job_slot_to_gpu[@]} - 1)) \
    --header : \
    --colsep , \
    CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' \
    "$repo_root/experiments/noisy/jobs/run_one.sh" \
        {method} {op} {id} {seed} 0.05 \
    ::: seed 42 \
    ::: op gaussian_blur motion_blur down_sampling inpainting_random \
    ::: method das rpcn \
    :::: "$repo_root/ids.csv"
