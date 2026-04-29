#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Run one experiment at a time on one visible GPU. Increase BATCH_SIZE_OVERRIDE
# on large-memory GPUs to raise memory use and throughput.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-128}"
export NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-8}"
export PREFETCH_FACTOR_OVERRIDE="${PREFETCH_FACTOR_OVERRIDE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE}"
echo "NUM_WORKERS_OVERRIDE=${NUM_WORKERS_OVERRIDE}"
echo "PREFETCH_FACTOR_OVERRIDE=${PREFETCH_FACTOR_OVERRIDE}"

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

echo "Experiments to run sequentially:"
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
  echo "  ${exp_name}"
done

for exp_name in "${EXPERIMENT_NAMES[@]}"; do
  log_dir="$(EXP_NAME="$exp_name" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

print(config.EXPERIMENTS[os.environ["EXP_NAME"]]["log_dir"])
PY
)"
  mkdir -p "$log_dir"
  stdout_log="${log_dir}/stdout.log"

  echo "============================================================"
  echo "Starting ${exp_name}"
  echo "Log: ${stdout_log}"
  echo "============================================================"

  EXP_NAME="$exp_name" python - <<'PY' 2>&1 | tee "$stdout_log"
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
    f"batch_size={exp['batch_size']} "
    f"num_workers={config.NUM_WORKERS} "
    f"prefetch_factor={config.PREFETCH_FACTOR}"
)
train_experiment(exp)
PY

  echo "Finished ${exp_name}"
done

echo "All non-D experiments finished."
