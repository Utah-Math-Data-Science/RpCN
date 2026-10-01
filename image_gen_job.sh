source ~/.bashrc
job_slot_to_gpu=(-1 1 2 3 4 5 6 7)

env_parallel --delay 3 --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
 CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}'' python RpCN_sample.py \
    --img_id {id} \
    --model v2.1 \
    --resample_freq 20 \
    --pCN_ct 10' \
  :::: prompts/prompt_ids.csv