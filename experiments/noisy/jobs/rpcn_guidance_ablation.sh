#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ids_file="$repo_root/ids.csv"
output_root="$repo_root/generated/ablations/noisy_rpcn"
operator="down_sampling"
limit=12
seed=31
noise_sigma=0.05
gpus_csv="0"
python_bin="python"
execute=1
step_sizes=(0.1 0.3 1.0)
start_points=(0.25 0.5 0.75)

usage() {
    cat <<'EOF'
Usage: experiments/noisy/jobs/rpcn_guidance_ablation.sh [options]

Options:
  --operator NAME       Forward operator (default: down_sampling)
  --gpus IDS            Comma-separated GPU IDs (default: 0)
  --ids PATH            CSV containing an id column
  --limit N             Number of image IDs (default: 12)
  --seed N              Sampling seed (default: 31)
  --noise-sigma VALUE   Measurement noise in model space (default: 0.05)
  --output-root PATH    Ablation output root
  --python PATH         Python executable
  --dry-run             Print commands without executing
  -h, --help            Show this help
EOF
}

while (($#)); do
    case "$1" in
        --operator) operator=$2; shift 2 ;;
        --gpus) gpus_csv=$2; shift 2 ;;
        --ids) ids_file=$2; shift 2 ;;
        --limit) limit=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --noise-sigma) noise_sigma=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --python) python_bin=$2; shift 2 ;;
        --dry-run) execute=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

mapfile -t image_ids < <(
    while IFS=, read -r image_id _; do
        [[ $image_id == "id" || -z $image_id ]] && continue
        printf '%s\n' "$image_id"
    done < "$ids_file"
)
image_ids=("${image_ids[@]:0:limit}")
IFS=',' read -r -a gpus <<< "$gpus_csv"

commands=()
for start in "${start_points[@]}"; do
    for step_size in "${step_sizes[@]}"; do
        config_root="$output_root/start_${start}_step_${step_size}"
        for image_id in "${image_ids[@]}"; do
            printf -v command '%q ' \
                "$python_bin" -m experiments.noisy.rpcn \
                --save --img_id "$image_id" --seed "$seed" \
                --operator "$operator" --noise_sigma "$noise_sigma" \
                --operator_seed 42 --measurement_seed 42 \
                --guidance_start "$start" --guidance_step_size "$step_size" \
                --output_root "$config_root"
            commands+=("$command")
        done
    done
done

printf 'Planned %d noisy RpCN runs.\n' "${#commands[@]}"
if ((execute == 0)); then
    printf '%s\n' "${commands[@]}"
    exit 0
fi

cd "$repo_root"
CUDA_VISIBLE_DEVICES=${MEASUREMENT_GPU:-${gpus[0]}} \
    "$python_bin" -m experiments.noisy.make_measurements \
    --operators "$operator" --noise_sigma "$noise_sigma" \
    --operator_seed 42 --measurement_seed 42

run_queue() {
    local gpu=$1 worker=$2 count=$3 index
    for ((index = worker; index < ${#commands[@]}; index += count)); do
        CUDA_VISIBLE_DEVICES="$gpu" bash -lc "${commands[index]}"
    done
}

pids=()
for ((worker = 0; worker < ${#gpus[@]}; worker++)); do
    run_queue "${gpus[worker]}" "$worker" "${#gpus[@]}" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
