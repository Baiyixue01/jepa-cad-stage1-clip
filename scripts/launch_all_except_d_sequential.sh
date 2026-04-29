#!/usr/bin/env bash
set -euo pipefail

echo "This script has been replaced by scripts/launch_all_except_d_parallel.sh"
echo "Use:"
echo "  GPU_IDS=0,1,2,3,4,5 bash scripts/launch_all_except_d_parallel.sh"
exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_all_except_d_parallel.sh"
