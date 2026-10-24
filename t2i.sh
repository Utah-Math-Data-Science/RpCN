source ~/.bashrc
job_slot_to_gpu=(-1 0 1 2 3 4 5 6 7)

env_parallel --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
 CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' python {script}.py \
    --img_id {id} \
    --model {model}\
  :::: generated/prompt_ids.csv ::: model v1.4 v1.5 v2.1 ::: script RpCN_sample_new DAS_sample_new CREPE_sample_new

