#!/usr/bin/env bash

# shellcheck source=/dev/null
source ~/.bashrc
job_slot_to_gpu=(-1 3 4 5 6 7)

generate() {
  local id=$1
  local model=$2
  local sampler=$3
  local config=()

  case "$sampler" in
    RpCN)
      case "$model" in
        v1.4)
          config=(--move_steps "0,20,40,60,80")
          ;;
        v1.5)
          config=(--pcn_step_ladder "1,0.85,0.5,0.1" --move_steps "0,20,40,60,80")
          ;;
        v2.1)
          config=(--pcn_step_ladder "1,0.85,0.5,0.3" --move_steps "0,20,40,60,80")
          ;;
      esac

      rpcn-sample \
        --img_id "$id" \
        --model "$model" \
        --resample_freq 20 \
        --pCN_ct 10 \
        --output_label RpCN \
        "${config[@]}"
      ;;
    CREPE)
      crepe-sample \
        --img_id "$id" \
        --model "$model"
      ;;
    DAS_0.1)
      das-sample \
        --img_id "$id" \
        --model "$model" \
        --alpha_kl 0.1
      ;;
    DAS_0.01)
      das-sample \
        --img_id "$id" \
        --model "$model" \
        --alpha_kl 0.01
      ;;
    *)
      printf 'Unknown sampler: %s\n' "$sampler" >&2
      return 2
      ;;
  esac
}

prompt_file=$(python -c 'from importlib.resources import files; print(files("rpcn.data").joinpath("prompt_ids.csv"))')

# GNU Parallel expands the quoted job slot and replacement fields.
# shellcheck disable=SC2016,SC1083
env_parallel --delay 3 --eta \
  -j $((${#job_slot_to_gpu[@]} - 1)) \
  --colsep , --header : \
  CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' generate {id} {model} {sampler} \
  :::: "$prompt_file" \
  ::: model v1.4 v1.5 v2.1 \
  ::: sampler RpCN CREPE DAS_0.1 DAS_0.01
