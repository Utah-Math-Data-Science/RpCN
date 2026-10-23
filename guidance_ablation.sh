#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

operator="down_sampling"
ids_file="$script_dir/ids.csv"
output_root="$script_dir/generated/ablations"
seed=31
limit=12
gpus_csv="0"
python_bin="python"
scaling="variance"
clip_norm=2.5
execute=1
rerun=0
schedules=(early middle late constant)
strengths=(0.25 0.5 1.25 2.5 5.0)

usage() {
    cat <<'EOF'
Usage: ./guidance_ablation.sh [options]

Options:
  --operator NAME       Forward operator (default: down_sampling)
  --gpus IDS            Comma-separated GPU IDs (default: 0)
  --strengths "VALUES"  Space-separated guidance scales
  --scaling MODE        std or variance (default: variance)
  --clip-norm VALUE     Gradient clipping norm (default: 2.5)
  --ids PATH            CSV containing an id column (default: ids.csv)
  --limit N             Number of image IDs to run (default: 12)
  --seed N              Base seed (default: 31)
  --output-root PATH    Ablation output root
  --python PATH         Python executable (default: python)
  --dry-run             Print the planned commands without executing them
  --execute             Execute runs (default; retained for compatibility)
  --rerun               Replace tensors that already exist
  -h, --help            Show this help
EOF
}

while (($#)); do
    case "$1" in
        --operator) operator=$2; shift 2 ;;
        --gpus) gpus_csv=$2; shift 2 ;;
        --strengths) read -r -a strengths <<< "$2"; shift 2 ;;
        --scaling) scaling=$2; shift 2 ;;
        --clip-norm) clip_norm=$2; shift 2 ;;
        --ids) ids_file=$2; shift 2 ;;
        --limit) limit=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --python) python_bin=$2; shift 2 ;;
        --dry-run) execute=0; shift ;;
        --execute) execute=1; shift ;;
        --rerun) rerun=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ $scaling != "std" && $scaling != "variance" ]]; then
    printf '%s\n' "--scaling must be std or variance" >&2
    exit 2
fi
if ((limit < 1)); then
    printf '%s\n' "--limit must be at least 1" >&2
    exit 2
fi
if ((${#strengths[@]} == 0)); then
    printf '%s\n' "--strengths must contain at least one value" >&2
    exit 2
fi

IFS=',' read -r -a gpus <<< "$gpus_csv"
if ((${#gpus[@]} == 0)); then
    printf '%s\n' "--gpus must contain at least one GPU ID" >&2
    exit 2
fi

mapfile -t image_ids < <(
    while IFS=, read -r image_id _; do
        [[ $image_id == "id" || -z $image_id ]] && continue
        printf '%s\n' "$image_id"
    done < "$ids_file"
)
image_ids=("${image_ids[@]:0:limit}")
if ((${#image_ids[@]} == 0)); then
    printf 'No image IDs found in %s\n' "$ids_file" >&2
    exit 2
fi

job_schedules=()
job_strengths=()
job_image_ids=()
for schedule in "${schedules[@]}"; do
    for strength in "${strengths[@]}"; do
        for image_id in "${image_ids[@]}"; do
            job_schedules+=("$schedule")
            job_strengths+=("$strength")
            job_image_ids+=("$image_id")
        done
    done
done

printf 'Planned %d runs: %d schedules x %d strengths x %d images for %s.\n' \
    "${#job_schedules[@]}" "${#schedules[@]}" "${#strengths[@]}" \
    "${#image_ids[@]}" "$operator"

build_command() {
    local schedule=$1
    local strength=$2
    local image_id=$3
    local tensor_dir=$4
    command=(
        "$python_bin" "$script_dir/RpCN_sample.py"
        --save
        --img_id "$image_id"
        --seed "$seed"
        --operator "$operator"
        --guidance_schedule "$schedule"
        --guidance_scaling "$scaling"
        --guidance_scale "$strength"
        --grad_clip_norm "$clip_norm"
        --output_dir "$tensor_dir"
    )
}

if ((execute == 0)); then
    for ((index = 0; index < ${#job_schedules[@]}; index++)); do
        config_name="${job_schedules[index]}_${scaling}_scale_${job_strengths[index]}"
        tensor_dir="$output_root/$operator/$config_name/tensors"
        build_command "${job_schedules[index]}" "${job_strengths[index]}" \
            "${job_image_ids[index]}" "$tensor_dir"
        printf '%q ' "${command[@]}"
        printf '\n'
    done
    exit 0
fi

run_gpu_queue() {
    local gpu=$1
    local worker_index=$2
    local worker_count=$3
    local index schedule strength image_id effective_seed config_name tensor_dir output

    for ((index = worker_index; index < ${#job_schedules[@]}; index += worker_count)); do
        schedule=${job_schedules[index]}
        strength=${job_strengths[index]}
        image_id=${job_image_ids[index]}
        effective_seed=$((seed + 2435 * image_id))
        config_name="${schedule}_${scaling}_scale_${strength}"
        tensor_dir="$output_root/$operator/$config_name/tensors"
        output="$tensor_dir/${image_id}_${operator}_${effective_seed}.pt"

        if [[ -f $output && $rerun -eq 0 ]]; then
            printf 'skip %s\n' "$output"
            continue
        fi

        mkdir -p "$tensor_dir"
        build_command "$schedule" "$strength" "$image_id" "$tensor_dir"
        printf 'gpu=%s schedule=%s scale=%s image=%s\n' \
            "$gpu" "$schedule" "$strength" "$image_id"
        CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
    done
}

pids=()
for ((worker_index = 0; worker_index < ${#gpus[@]}; worker_index++)); do
    run_gpu_queue "${gpus[worker_index]}" "$worker_index" "${#gpus[@]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
