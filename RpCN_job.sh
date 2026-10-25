source ~/.bashrc

job_slot_to_gpu=(-1 0 1 2 3 4 7)

env_parallel \
    --delay 3 \
    --eta \
    -j $((${#job_slot_to_gpu[@]} - 1)) \
    --header : \
    --colsep , \
    CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' \
    python RpCN_sample_noiseless.py \
        --save \
        --plot \
        --seed {seed} \
        --operator_seed 42 \
        --img_id {id} \
        --operator {op} \
    ::: seed 42 \
    ::: op gaussian_blur motion_blur down_sampling inpainting_random\
    :::: ids.csv
