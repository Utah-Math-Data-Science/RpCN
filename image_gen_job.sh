source ~/.bashrc
job_slot_to_gpu=(-1 2 3 4 5 6 7)

env_parallel --delay 3 --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
 CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}'' python RpCN_sample.py \
    --img_id {id} \
    --model {model}\
    --resample_freq 20 \
    --pCN_ct 10' \
  :::: generated/prompt_ids.csv ::: model v1.4 v1.5 v2.1

