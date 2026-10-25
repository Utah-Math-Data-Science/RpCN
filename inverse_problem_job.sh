source ~/.bashrc

job_slot_to_gpu=(-1 1 2 3 4 5 6 7)

env_parallel \
    --delay 3 \
    --eta \
    -j $((${#job_slot_to_gpu[@]} - 1)) \
    --header : \
    --colsep , \
    CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' \
    python {method}.py \
        --save \
        --plot \
        --seed {seed} \
        --operator_seed 42 \
        --img_id {id} \
        --operator {op} \
    ::: seed 42 \
    ::: op inpainting_random\
    ::: method RpCN_sample_noiseless DAS_sample_noiseless\
    :::: ids.csv \
