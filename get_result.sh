#!/usr/bin/env bash
set -euo pipefail

# =====================
# Basic config
# =====================

REMOTE_HOST="jiangB200"

# 远端项目路径：按你的真实远端路径修改
REMOTE_ROOT="/home/byx/project/jepa-cad-stage1-clip"

# 本地项目路径：默认取当前脚本所在目录的上一级
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 本地保存远端结果的位置
LOCAL_RESULT_DIR="${LOCAL_ROOT}/remote_results"

mkdir -p "${LOCAL_RESULT_DIR}"

echo "Remote host: ${REMOTE_HOST}"
echo "Remote root: ${REMOTE_ROOT}"
echo "Local result dir: ${LOCAL_RESULT_DIR}"
echo

# =====================
# Pull lightweight training info
# =====================

echo "Pulling logs..."
rsync -avhP \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/"*/logs \
  "${LOCAL_RESULT_DIR}/outputs/" || true

echo
echo "Pulling eval results..."
rsync -avhP \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/"*/eval \
  "${LOCAL_RESULT_DIR}/outputs/" || true

echo
echo "Pulling comparison csv..."
rsync -avhP \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/compare_experiments.csv" \
  "${LOCAL_RESULT_DIR}/outputs/" || true

echo
echo "Pulling manifests..."
rsync -avhP \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/manifests" \
  "${LOCAL_RESULT_DIR}/outputs/" || true

echo
echo "Done."
echo "Local results saved to:"
echo "  ${LOCAL_RESULT_DIR}"
echo
echo "Example:"
echo "  tail -f ${LOCAL_RESULT_DIR}/outputs/exp_first_registers_giant_transformer_strong/logs/stdout.log"