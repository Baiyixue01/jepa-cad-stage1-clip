#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EXPERIMENT_VARS=(EXP_A EXP_B EXP_FIRST)
GPU_IDS=(0 1 2)

if [[ "${#EXPERIMENT_VARS[@]}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "Experiment count and GPU count do not match." >&2
  exit 1
fi

python - <<'PY'
from jepa_stage1_clip_ablation import config

required = ["EXP_A", "EXP_B", "EXP_FIRST"]
missing = [name for name in required if not hasattr(config, name)]
if missing:
    raise SystemExit(f"Missing experiment config(s): {', '.join(missing)}")

for name in required:
    exp = getattr(config, name)
    if not exp.get("name"):
        raise SystemExit(f"{name} has no experiment name")
    if not exp.get("output_dir"):
        raise SystemExit(f"{name} has no output_dir")
PY

for idx in "${!EXPERIMENT_VARS[@]}"; do
  exp_var="${EXPERIMENT_VARS[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"

  exp_name="$(EXP_VAR="$exp_var" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

exp = getattr(config, os.environ["EXP_VAR"])
print(exp["name"])
PY
)"

  log_dir="$(EXP_VAR="$exp_var" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

exp = getattr(config, os.environ["EXP_VAR"])
print(exp["log_dir"])
PY
)"

  mkdir -p "$log_dir"
  stdout_log="$log_dir/stdout.log"

  echo "Starting ${exp_var} (${exp_name}) on GPU ${gpu_id}"
  echo "Log: ${stdout_log}"

  CUDA_VISIBLE_DEVICES="$gpu_id" EXP_VAR="$exp_var" nohup python - <<'PY' > "$stdout_log" 2>&1 &
import os

from jepa_stage1_clip_ablation import config
from jepa_stage1_clip_ablation.train import train_experiment

exp = getattr(config, os.environ["EXP_VAR"])
train_experiment(exp)
PY

  echo "${exp_var} pid=$!"
done

echo "All experiments launched."
echo "Monitor GPU usage with: nvidia-smi"
echo "Monitor logs with:"
for exp_var in "${EXPERIMENT_VARS[@]}"; do
  log_dir="$(EXP_VAR="$exp_var" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

exp = getattr(config, os.environ["EXP_VAR"])
print(exp["log_dir"])
PY
)"
  echo "  tail -f ${log_dir}/stdout.log"
done
