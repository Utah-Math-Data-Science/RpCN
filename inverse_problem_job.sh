source ~/.bashrc

job_slot_to_gpu=(-1 2 3 4 5 7)

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
        --seed {seed} \
        --img_id {id} \
        --operator {op} \
    ::: seed 31 \
    ::: op gaussian_blur motion_blur down_sampling\
    :::: ids.csv