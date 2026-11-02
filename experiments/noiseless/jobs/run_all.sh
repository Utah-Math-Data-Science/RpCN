#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
source ~/.bashrc
cd "$repo_root"

job_slot_to_gpu=(-1 0 5 6 7)

env_parallel \
    --delay 1 \
    --eta \
    -j $((${#job_slot_to_gpu[@]} - 1)) \
    --header : \
    --colsep , \
    CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' \
    python -m experiments.noiseless.{method} \
        --save \
        --plot \
        --seed {seed} \
        --operator_seed 42 \
        --img_id {id} \
        --operator {op} \
    ::: seed 42 \
    ::: op gaussian_blur motion_blur down_sampling inpainting_random \
    ::: method rpcn das \
    :::: "$repo_root/ids.csv"
