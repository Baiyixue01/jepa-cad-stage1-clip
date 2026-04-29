#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
export NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-8}"
export PREFETCH_FACTOR_OVERRIDE="${PREFETCH_FACTOR_OVERRIDE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Comma-separated physical GPU ids. Example: GPU_IDS=0,1,2,3,4,5
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS:-0,1,2,3,4,5,6,7}"

mapfile -t EXPERIMENT_NAMES < <(
python - <<'PY'
from jepa_stage1_clip_ablation import config

excluded = {config.EXP_D["name"]}
for name in config.EXPERIMENTS:
    if name not in excluded:
        print(name)
PY
)

if [[ "${#EXPERIMENT_NAMES[@]}" -eq 0 ]]; then
  echo "No experiments to run after excluding EXP_D." >&2
  exit 1
fi

if [[ "${#GPU_IDS[@]}" -lt "${#EXPERIMENT_NAMES[@]}" ]]; then
  echo "Not enough GPUs: experiments=${#EXPERIMENT_NAMES[@]} gpu_ids=${#GPU_IDS[@]}" >&2
  echo "Set GPU_IDS with enough ids, for example: GPU_IDS=0,1,2,3,4,5" >&2
  exit 1
fi

echo "ROOT_DIR=${ROOT_DIR}"
echo "GPU_IDS=${GPU_IDS[*]}"
echo "BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE:-<use config>}"
echo "NUM_WORKERS_OVERRIDE=${NUM_WORKERS_OVERRIDE}"
echo "PREFETCH_FACTOR_OVERRIDE=${PREFETCH_FACTOR_OVERRIDE}"
echo "Launching one experiment per GPU:"

pids=()
for idx in "${!EXPERIMENT_NAMES[@]}"; do
  exp_name="${EXPERIMENT_NAMES[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"

  log_dir="$(EXP_NAME="$exp_name" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

print(config.EXPERIMENTS[os.environ["EXP_NAME"]]["log_dir"])
PY
)"
  mkdir -p "$log_dir"
  stdout_log="${log_dir}/stdout.log"

  echo "  GPU ${gpu_id}: ${exp_name}"
  echo "    log: ${stdout_log}"

  CUDA_VISIBLE_DEVICES="$gpu_id" EXP_NAME="$exp_name" python - <<'PY' > "$stdout_log" 2>&1 &
import os

from jepa_stage1_clip_ablation import config
from jepa_stage1_clip_ablation.train import train_experiment

exp = dict(config.EXPERIMENTS[os.environ["EXP_NAME"]])

if os.environ.get("BATCH_SIZE_OVERRIDE"):
    exp["batch_size"] = int(os.environ["BATCH_SIZE_OVERRIDE"])
if os.environ.get("NUM_WORKERS_OVERRIDE"):
    config.NUM_WORKERS = int(os.environ["NUM_WORKERS_OVERRIDE"])
if os.environ.get("PREFETCH_FACTOR_OVERRIDE"):
    config.PREFETCH_FACTOR = int(os.environ["PREFETCH_FACTOR_OVERRIDE"])

print(
    "[launch] "
    f"experiment={exp['name']} "
    f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"batch_size={exp['batch_size']} "
    f"num_workers={config.NUM_WORKERS} "
    f"prefetch_factor={config.PREFETCH_FACTOR}",
    flush=True,
)
train_experiment(exp)
PY
  pids+=("$!")
done

echo "All non-D experiments launched."
echo "PIDs: ${pids[*]}"
echo "Monitor GPU usage with: nvidia-smi"
echo "Monitor logs with:"
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
  log_dir="$(EXP_NAME="$exp_name" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

print(config.EXPERIMENTS[os.environ["EXP_NAME"]]["log_dir"])
PY
)"
  echo "  tail -f ${log_dir}/stdout.log"
done

wait "${pids[@]}"
echo "All non-D experiments finished."
