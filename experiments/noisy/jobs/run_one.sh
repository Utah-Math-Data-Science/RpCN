#!/usr/bin/env bash
set -uo pipefail

if (($# != 5)); then
    echo "Usage: run_one.sh METHOD OPERATOR IMAGE_ID SEED NOISE_SIGMA" >&2
    exit 2
fi

method=$1
operator=$2
image_id=$3
seed=$4
noise_sigma=$5
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=${PYTHON_BIN:-python}
max_attempts=${MAX_ATTEMPTS:-3}

case "$method" in
    rpcn) method_dir=RpCN ;;
    das) method_dir=DAS ;;
    crepe) method_dir=CREPE ;;
    *) echo "Unknown method: $method" >&2; exit 2 ;;
esac

output="$repo_root/generated/inverse_problems/noisy/sigma_${noise_sigma}/${method_dir}/${operator}/sample_seed_${seed}/${image_id}.pt"
if [[ -s $output ]]; then
    printf 'skip existing %s\n' "$output"
    exit 0
fi

cd "$repo_root"
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    printf 'method=%s operator=%s image=%s attempt=%d/%d gpu=%s\n' \
        "$method" "$operator" "$image_id" "$attempt" "$max_attempts" \
        "${CUDA_VISIBLE_DEVICES:-unset}"
    if "$python_bin" -m "experiments.noisy.${method}" \
        --save \
        --plot \
        --seed "$seed" \
        --operator_seed 42 \
        --measurement_seed 42 \
        --noise_sigma "$noise_sigma" \
        --img_id "$image_id" \
        --operator "$operator"; then
        exit 0
    fi
    if [[ -s $output ]]; then
        printf 'sample was saved despite a later command failure: %s\n' "$output"
        exit 0
    fi
    if ((attempt < max_attempts)); then
        sleep $((15 * attempt))
    fi
done

printf 'failed after %d attempts: %s %s %s\n' \
    "$max_attempts" "$method" "$operator" "$image_id" >&2
exit 1
