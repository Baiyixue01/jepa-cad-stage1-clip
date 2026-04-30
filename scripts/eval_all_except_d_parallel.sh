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
    if name not in excluded and config.EXPERIMENTS[name]["best_checkpoint"].exists():
        print(name)
PY
)

if [[ "${#EXPERIMENT_NAMES[@]}" -eq 0 ]]; then
  echo "No finished non-D experiments with best.pt found." >&2
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
echo "Evaluating one experiment per GPU:"

pids=()
for idx in "${!EXPERIMENT_NAMES[@]}"; do
  exp_name="${EXPERIMENT_NAMES[$idx]}"
  gpu_id="${GPU_IDS[$idx]}"

  eval_dir="$(EXP_NAME="$exp_name" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

print(config.EXPERIMENTS[os.environ["EXP_NAME"]]["eval_dir"])
PY
)"
  mkdir -p "$eval_dir"
  stdout_log="${eval_dir}/eval_stdout.log"

  echo "  GPU ${gpu_id}: ${exp_name}"
  echo "    log: ${stdout_log}"

  CUDA_VISIBLE_DEVICES="$gpu_id" EXP_NAME="$exp_name" python - <<'PY' > "$stdout_log" 2>&1 &
import os

from jepa_stage1_clip_ablation import config
from jepa_stage1_clip_ablation.eval_retrieval import eval_experiment

exp = dict(config.EXPERIMENTS[os.environ["EXP_NAME"]])

if os.environ.get("BATCH_SIZE_OVERRIDE"):
    exp["batch_size"] = int(os.environ["BATCH_SIZE_OVERRIDE"])
if os.environ.get("NUM_WORKERS_OVERRIDE"):
    config.NUM_WORKERS = int(os.environ["NUM_WORKERS_OVERRIDE"])
if os.environ.get("PREFETCH_FACTOR_OVERRIDE"):
    config.PREFETCH_FACTOR = int(os.environ["PREFETCH_FACTOR_OVERRIDE"])

print(
    "[eval_launch] "
    f"experiment={exp['name']} "
    f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"batch_size={exp['batch_size']} "
    f"num_workers={config.NUM_WORKERS} "
    f"prefetch_factor={config.PREFETCH_FACTOR}",
    flush=True,
)
metrics, best_epoch = eval_experiment(exp)
print(f"[eval_done] experiment={exp['name']} best_epoch={best_epoch} metrics={metrics}", flush=True)
PY
  pids+=("$!")
done

echo "All eval jobs launched."
echo "PIDs: ${pids[*]}"
echo "Monitor logs with:"
for exp_name in "${EXPERIMENT_NAMES[@]}"; do
  eval_dir="$(EXP_NAME="$exp_name" python - <<'PY'
import os
from jepa_stage1_clip_ablation import config

print(config.EXPERIMENTS[os.environ["EXP_NAME"]]["eval_dir"])
PY
)"
  echo "  tail -f ${eval_dir}/eval_stdout.log"
done

wait "${pids[@]}"

echo "Parallel eval finished. Writing comparison summary."
python - <<'PY'
import json

from jepa_stage1_clip_ablation import config
from jepa_stage1_clip_ablation.utils import write_csv

rows = []
for name, exp in config.EXPERIMENTS.items():
    if name == config.EXP_D["name"]:
        continue
    if not exp["eval_summary"].exists():
        print(f"skip summary for {name}: missing {exp['eval_summary']}")
        continue
    with exp["eval_summary"].open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows.append({
        "experiment_name": exp["name"],
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "top10": metrics["top10"],
        "mrr": metrics["mrr"],
        "mean_rank": metrics["mean_rank"],
        "median_rank": metrics["median_rank"],
        "best_epoch": "",
        "best_checkpoint": str(exp["best_checkpoint"]),
    })

write_csv(
    config.COMPARE_CSV,
    rows,
    [
        "experiment_name",
        "top1",
        "top5",
        "top10",
        "mrr",
        "mean_rank",
        "median_rank",
        "best_epoch",
        "best_checkpoint",
    ],
)
print(f"wrote {config.COMPARE_CSV}")
PY

echo "All non-D eval jobs finished."
