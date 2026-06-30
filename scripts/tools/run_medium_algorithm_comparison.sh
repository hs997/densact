#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/cnc/SSD_1T/xzw/isaac-sim}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
LOG_ROOT="${ROOT}/logs/rsl_rl/platoon_happo"

TAG="${TAG:-medium_compare_$(date +%Y%m%d_%H%M%S)}"
PIPELINE_LOG="${PIPELINE_LOG:-${ROOT}/train_${TAG}.log}"
RESULT_ROOT="${RESULT_ROOT:-${LOG_ROOT}/${TAG}_package}"
RESULT_TAR="${RESULT_TAR:-${RESULT_ROOT}.tar.gz}"

NUM_ENVS="${NUM_ENVS:-64}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1200}"
EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEED="${SEED:-42}"
SKIP_EVAL="${SKIP_EVAL:-0}"
KIT_ARGS="${KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false}"

if [[ "${MEDIUM_COMPARE_LOG_ACTIVE:-0}" != "1" ]]; then
  export MEDIUM_COMPARE_LOG_ACTIVE=1
  mkdir -p "$(dirname "${PIPELINE_LOG}")"
  exec >> "${PIPELINE_LOG}" 2>&1
fi

cd "${ROOT}" || exit 1
trap 'status=$?; echo "[MEDIUM_COMPARE] exit status=${status} at line=${LINENO}"' EXIT

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
unset VIRTUAL_ENV PYTHONHOME

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/source/isaaclab:${ROOT}/source/isaaclab_rl:${ROOT}/source/isaaclab_tasks:${ROOT}/source/my_exts:${ISAAC_SIM_ROOT}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"

COMMON_OVERRIDES=(
  "env.algorithm.enable_attack=true"
  "env.algorithm.attack_level=medium"
  "env.algorithm.attack_mode=profile"
  "env.algorithm.attack_curriculum_warmup_updates=0"
  "env.algorithm.max_fdi_pos=2.0"
  "env.algorithm.max_fdi_acc=0.50"
  "env.algorithm.max_dos_rate=0.10"
  "env.algorithm.enable_shield=true"
  "env.algorithm.happo_action_clip=0.4"
  "env.commands.base_velocity.ranges.lin_vel_x=[0.30,0.45]"
  "env.safety_shield.d_drop=1.45"
  "env.safety_shield.catchup_action=-0.355"
  "env.safety_shield.lateral_turn_gain=0.32"
  "env.safety_shield.centerline_turn_gain=0.28"
  "env.safety_shield.pair3_lateral_gain_scale=0.85"
  "env.safety_shield.pair4_lateral_gain_scale=0.90"
  "env.actions.joint_vel_1.scale=12.5"
  "env.actions.joint_vel_2.scale=12.5"
  "env.actions.joint_vel_3.scale=12.5"
  "env.actions.joint_vel_4.scale=12.5"
  "env.actions.joint_vel_5.scale=12.5"
)

latest_run_for_name() {
  local run_name="$1"
  find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

select_checkpoints() {
  local run_dir="$1"
  python3 - "${run_dir}" "${EVAL_EVERY}" "${MAX_ITERATIONS}" <<'PY'
from pathlib import Path
import re
import sys

run = Path(sys.argv[1])
every = int(sys.argv[2])
max_iterations = int(sys.argv[3])
items = []
for path in run.glob("model_*.pt"):
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match:
        iteration = int(match.group(1))
        if iteration % every == 0:
            items.append((iteration, path))
items.sort(key=lambda item: item[0])
final = run / "model_final.pt"
if final.exists():
    items.append((max_iterations, final))
print(",".join(str(path.resolve()) for _, path in items))
PY
}

run_train() {
  local label="$1"
  shift
  local run_name="${TAG}_${label}"
  echo
  echo "[MEDIUM_COMPARE] train label=${label} run_name=${run_name} max_iterations=${MAX_ITERATIONS}"
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${NUM_ENVS}" \
    --headless \
    --seed "${SEED}" \
    --run_name "${run_name}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
  local run_dir
  run_dir="$(latest_run_for_name "${run_name}")"
  if [[ -z "${run_dir}" || ! -d "${run_dir}" ]]; then
    echo "[ERROR] could not locate run dir for ${run_name}" >&2
    exit 2
  fi
  mkdir -p "${RESULT_ROOT}/training"
  echo "${label}=${run_dir}" >> "${RESULT_ROOT}/training/runs.txt"
  echo "[MEDIUM_COMPARE] completed ${label}: ${run_dir}"
}

run_eval() {
  local label="$1"
  local run_dir="$2"
  shift 2
  local checkpoints
  checkpoints="$(select_checkpoints "${run_dir}")"
  if [[ -z "${checkpoints}" ]]; then
    echo "[ERROR] no checkpoints selected for ${label}: ${run_dir}" >&2
    exit 3
  fi
  local output_dir="${RESULT_ROOT}/evaluation/${label}"
  echo
  echo "[MEDIUM_COMPARE] eval label=${label} checkpoints=${checkpoints}"
  "${ISAAC_PYTHON}" scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${EVAL_NUM_ENVS}" \
    --seed "${SEED}" \
    --headless \
    --eval_checkpoints "${checkpoints}" \
    --eval_steps "${EVAL_STEPS}" \
    --warmup_steps 0 \
    --output_dir "${output_dir}" \
    --enable_attack_eval \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

copy_artifacts() {
  local label="$1"
  local run_dir="$2"
  local dst="${RESULT_ROOT}/training/${label}"
  mkdir -p "${dst}"
  cp -a "${run_dir}/platoon_metrics.csv" "${dst}/" 2>/dev/null || true
  cp -a "${run_dir}/params" "${dst}/" 2>/dev/null || true
  cp -a "${run_dir}/model_final.pt" "${dst}/" 2>/dev/null || true
  cp -a "${run_dir}/model_best.pt" "${dst}/" 2>/dev/null || true
}

echo "[MEDIUM_COMPARE] tag=${TAG}"
echo "[MEDIUM_COMPARE] log=${PIPELINE_LOG}"
echo "[MEDIUM_COMPARE] result_root=${RESULT_ROOT}"
echo "[MEDIUM_COMPARE] iterations=${MAX_ITERATIONS} eval_every=${EVAL_EVERY} eval_steps=${EVAL_STEPS}"

rm -rf "${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}/training" "${RESULT_ROOT}/evaluation"
{
  echo "tag=${TAG}"
  echo "medium_attack=max_fdi_pos=2.0 max_fdi_acc=0.50 max_dos_rate=0.10"
  echo "max_iterations=${MAX_ITERATIONS}"
  echo "eval_every=${EVAL_EVERY}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "seed=${SEED}"
  echo "local_backup=/home/cnc/SSD_1T/xzw/IsaacLab-main/backups/pre_medium_compare_code_assets_20260701_022215.tar"
  echo "git_backup_commit=79b8290"
  echo "git_current_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} > "${RESULT_ROOT}/manifest.txt"

run_train "mappo" \
  "env.algorithm.algorithm=mappo" \
  "env.algorithm.enable_teacher=false" \
  "env.algorithm.happo_use_factor=false"

run_train "happo_no_meta" \
  "env.algorithm.algorithm=happo" \
  "env.algorithm.enable_teacher=false" \
  "env.algorithm.happo_use_factor=true"

run_train "happo_meta" \
  "env.algorithm.algorithm=happo" \
  "env.algorithm.enable_teacher=true" \
  "env.algorithm.happo_use_factor=true"

while IFS='=' read -r label run_dir; do
  copy_artifacts "${label}" "${run_dir}"
done < "${RESULT_ROOT}/training/runs.txt"

if [[ "${SKIP_EVAL}" != "1" ]]; then
  while IFS='=' read -r label run_dir; do
    case "${label}" in
      mappo)
        run_eval "${label}" "${run_dir}" \
          "env.algorithm.algorithm=mappo" \
          "env.algorithm.enable_teacher=false" \
          "env.algorithm.happo_use_factor=false"
        ;;
      happo_no_meta)
        run_eval "${label}" "${run_dir}" \
          "env.algorithm.algorithm=happo" \
          "env.algorithm.enable_teacher=false" \
          "env.algorithm.happo_use_factor=true"
        ;;
      happo_meta)
        run_eval "${label}" "${run_dir}" \
          "env.algorithm.algorithm=happo" \
          "env.algorithm.enable_teacher=true" \
          "env.algorithm.happo_use_factor=true"
        ;;
    esac
  done < "${RESULT_ROOT}/training/runs.txt"

  python3 scripts/tools/plot_medium_algorithm_comparison.py --result-root "${RESULT_ROOT}"
fi

cp -a "${PIPELINE_LOG}" "${RESULT_ROOT}/pipeline.log" 2>/dev/null || true
cp -a debug_notes.md "${RESULT_ROOT}/debug_notes.md" 2>/dev/null || true
git diff > "${RESULT_ROOT}/git_diff_after_medium_compare.diff" || true
tar -czf "${RESULT_TAR}" -C "$(dirname "${RESULT_ROOT}")" "$(basename "${RESULT_ROOT}")"
echo "[MEDIUM_COMPARE] result root: ${RESULT_ROOT}"
echo "[MEDIUM_COMPARE] result tar:  ${RESULT_TAR}"
