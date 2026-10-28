source ~/.bashrc
job_slot_to_gpu=(-1 0 3 5 6 7)

generate() {
  local id=$1
  local model=$2
  local config

  case "$model" in
    v1.4)
      config=(--move_steps 0,20,40,60,80)
      ;;
    v1.5)
      config=(--pcn_step_ladder 1,0.85,0.5,0.1 --move_steps 0,20,40,60,80)
      ;;
    v2.1)
      config=(--pcn_step_ladder 1,0.85,0.5,0.3 --move_steps 0,20,40,60,80)
      ;;
  esac

  python RpCN_sample_new.py \
    --img_id "$id" \
    --model "$model" \
    --resample_freq 20 \
    --pCN_ct 10 \
    --output_label RpCN \
    "${config[@]}"
}

env_parallel --delay 3 --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
  CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' generate {id} {model} \
  :::: generated/prompt_ids.csv ::: model v1.4 v1.5 v2.1
