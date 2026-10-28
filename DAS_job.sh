source ~/.bashrc
job_slot_to_gpu=(-1 2 3 6 7)

env_parallel --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
 CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' python DAS_sample_new.py \
    --batch_size 4\
    --gradient_batch_size 1 \
    --alpha_kl 0.01 \
    --img_id {id} \
    --model {model}\
  :::: missing_ids.csv ::: model v1.4 v1.5 v2.1

