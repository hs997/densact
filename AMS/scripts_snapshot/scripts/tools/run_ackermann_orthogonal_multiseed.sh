#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
HOST_PYTHON="${HOST_PYTHON:-/home/cnc/Downloads/ENTER/bin/python3}"
SIM="${SIM:-/home/cnc/SSD_1T/xzw/isaac-sim}"
REFERENCE="${REFERENCE:-${ROOT}/logs/rsl_rl/platoon_happo/ams_bridge_refined_20260922/stages/hard_b_r2}"
BC_CHECKPOINT="${BC_CHECKPOINT:-${ROOT}/logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT}/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed}"
LOG="${LOG:-${OUTPUT_BASE}.log}"

mkdir -p "$(dirname "${OUTPUT_BASE}")"
exec >> "${LOG}" 2>&1
echo "[ORTHO_MULTI] start $(date '+%F %T %Z')"
echo "[ORTHO_MULTI] output_base=${OUTPUT_BASE}"

for seed_index in 0 1 2; do
  output="${OUTPUT_BASE}_s${seed_index}"
  echo "[ORTHO_MULTI] seed_index=${seed_index} output=${output}"
  "${HOST_PYTHON}" "${ROOT}/scripts/tools/run_ackermann_verified_followup.py" \
    --output "${output}" \
    --reference "${REFERENCE}" \
    --bc-checkpoint "${BC_CHECKPOINT}" \
    --methods happo_base_shared happo_meta happo_ams_only happo_meta_ams \
    --rounds 1 --seed-index "${seed_index}" --mode matrix
done

echo "[ORTHO_MULTI] complete $(date '+%F %T %Z')"
