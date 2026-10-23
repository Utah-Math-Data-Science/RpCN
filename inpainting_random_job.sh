#!/usr/bin/env bash

source ~/.bashrc

job_slot_to_gpu=(-1 1 2 3 4 5 6 7)

env_parallel \
    --delay 3 \
    --eta \
    -j $((${#job_slot_to_gpu[@]} - 1)) \
    --header : \
    --colsep , \
    CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' \
    python RpCN_sample.py \
        --save \
        --plot \
        --seed 31 \
        --img_id {id} \
        --operator inpainting_random \
        --n_chains 3 \
        --resampling_frequency 50 \
        --pcn_update_count 3 \
        --guidance_schedule late \
        --guidance_scaling std \
        --guidance_scale 0.0125 \
        --grad_clip_norm 2.5 \
    :::: ids.csv
