source ~/.bashrc
job_slot_to_gpu=(-1 2 3 5 6 7)

env_parallel --delay 3 --eta -j $((${#job_slot_to_gpu[@]} - 1)) --header : --colsep , CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' python RpCN_sample.py --seed {seed} --img_id {id} --operator {op} ::: seed 31 ::: op  inpainting_random :::: ids.csv