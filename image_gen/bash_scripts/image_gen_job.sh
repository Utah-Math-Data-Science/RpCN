source ~/.bashrc
job_slot_to_gpu=(-1  4 5 6 7)

env_parallel --delay 3 --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
 CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}'' python packaged.py \
    --img_id {id} \
    --model v1.5 \
    --resample_freq 20 \
    --pCN_ct 10' \
  :::: generated/prompt_ids.csv