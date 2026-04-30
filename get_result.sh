#!/usr/bin/env bash
set -euo pipefail

# =====================
# Basic config
# =====================

REMOTE_HOST="jiangB200"

# 远端项目根目录
REMOTE_PROJECT_ROOT="/home/baiyixue/project/jepa-cad-stage1-clip"

# 远端结果目录
REMOTE_OUTPUT_DIR="${REMOTE_PROJECT_ROOT}/jepa_stage1_clip_ablation/outputs"

# 本地项目路径：默认取当前脚本所在目录的上一级
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 本地保存远端结果的位置
LOCAL_RESULT_DIR="${LOCAL_ROOT}/jepa-cad-stage1-clip/remote_results"
LOCAL_OUTPUT_DIR="${LOCAL_RESULT_DIR}/outputs"

mkdir -p "${LOCAL_OUTPUT_DIR}"

echo "Remote host: ${REMOTE_HOST}"
echo "Remote output dir: ${REMOTE_OUTPUT_DIR}"
echo "Local output dir: ${LOCAL_OUTPUT_DIR}"
echo

# =====================
# Check remote path
# =====================

echo "Checking remote output directory..."

if ! ssh "${REMOTE_HOST}" "test -d '${REMOTE_OUTPUT_DIR}'"; then
  echo "[ERROR] Remote output directory does not exist:"
  echo "  ${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}"
  echo
  echo "You can check manually with:"
  echo "  ssh ${REMOTE_HOST} 'ls -lh ${REMOTE_PROJECT_ROOT}/jepa_stage1_clip_ablation/'"
  exit 1
fi

echo "Remote output directory exists."
echo

# =====================
# Pull outputs without model weights
# =====================

echo "Pulling outputs directory without model weights..."

rsync -avhP \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='checkpoint*' \
  --exclude='checkpoints/' \
  "${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}/" \
  "${LOCAL_OUTPUT_DIR}/"

echo
echo "Done."
echo "Local results saved to:"
echo "  ${LOCAL_OUTPUT_DIR}"
echo
echo "Examples:"
echo "  ls -lh ${LOCAL_OUTPUT_DIR}"
echo "  tail -f ${LOCAL_OUTPUT_DIR}/exp_first_registers_giant_transformer_strong/logs/stdout.log"
echo "  tail -f ${LOCAL_OUTPUT_DIR}/exp_e_dinov2_registers_giant_fusion_transformer/logs/stdout.log"