#!/usr/bin/env bash
set -euo pipefail

# Evaluate one fixed spatial-path checkpoint over the complete off-to-hard-b ladder.
# Each level produces aggregate evaluation CSV, env-0 trajectory CSV, a real
# renderer video, a true top-down trajectory plot, and a reward/attack plot.
ROOT="${ROOT:-/home/cnc/SSD_1T/xzw/IsaacLab-main}"
source "${ROOT}/scripts/tools/experiment_env.sh"
export ISAACLAB_EXPERIMENT_ROOT="${ROOT}"
SIM="${ISAAC_SIM_ROOT:-/home/cnc/SSD_1T/xzw/isaac-sim}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the selected spatial-path .pt file}"
TAG="${TAG:-spatial_attack_ladder_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/logs/rsl_rl/platoon_happo/${TAG}}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
EVAL_STEPS="${EVAL_STEPS:-3500}"
VIDEO_STEPS="${VIDEO_STEPS:-3500}"
SEED="${SEED:-42}"
ALGORITHM_LABEL="${ALGORITHM_LABEL:-happo_meta_ams}"
PLOT_PYTHON="${PLOT_PYTHON:-/home/cnc/Downloads/ENTER/bin/python3}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"
DIAGNOSTIC_CLEAN_ONLY="${DIAGNOSTIC_CLEAN_ONLY:-0}"
ATTACK_SEED="${ATTACK_SEED:-3407}"
KIT_ARGS="${KIT_ARGS:---/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false}"

cd "${ROOT}"
CHECKPOINT="$(realpath "${CHECKPOINT}")"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi
if pgrep -af 'scripts/reinforcement_learning/rsl_rl/(train|play|eval_happo_platoon)\.py' | grep -v "$$"; then
  echo "[ERROR] another Isaac training/evaluation process is active; refusing concurrent launch." >&2
  exit 3
fi

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL
unset _CONDA_EXE _CONDA_ROOT _CE_CONDA _CE_M
unset VIRTUAL_ENV PYTHONHOME
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/source/isaaclab:${ROOT}/source/isaaclab_rl:${ROOT}/source/isaaclab_tasks:${ROOT}/source/my_exts:${SIM}/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"

COMMON_OVERRIDES=(
  "env.episode_length_s=70.0"
  "env.commands.base_velocity.resampling_time_range=[70.0,70.0]"
  "env.commands.base_velocity.ranges.lin_vel_x=[0.30,0.45]"
  "env.commands.base_velocity.lane_change_enabled=true"
  "env.commands.base_velocity.lane_change_spatial=true"
  "env.commands.base_velocity.lane_change_start_x=-7.75"
  "env.commands.base_velocity.lane_change_length_m=6.0"
  "env.commands.base_velocity.lane_start_y=-0.065"
  "env.commands.base_velocity.lane_end_y=-1.160"
  "env.commands.base_velocity.enforce_road_boundaries=true"
  "env.commands.base_velocity.road_near_edge_y=0.47"
  "env.commands.base_velocity.road_far_edge_y=-1.72"
  "env.commands.base_velocity.road_edge_line_width_m=0.026"
  "env.commands.base_velocity.vehicle_length_m=0.240"
  "env.commands.base_velocity.vehicle_width_m=0.200"
  "env.commands.base_velocity.road_boundary_margin_m=0.05"
  "env.commands.base_velocity.road_boundary_termination_exceedance_m=0.10"
  "env.scene.robot.init_state.pos=[-8.75,-0.065,0.20]"
  "env.scene.robot_2.init_state.pos=[-10.25,-0.065,0.20]"
  "env.scene.robot_3.init_state.pos=[-11.75,-0.065,0.20]"
  "env.scene.robot_4.init_state.pos=[-13.25,-0.065,0.20]"
  "env.scene.robot_5.init_state.pos=[-14.75,-0.065,0.20]"
  "env.algorithm.enable_shield=true"
  "env.algorithm.happo_action_clip=0.4"
  "env.safety_shield.d_crit=0.50"
  "env.safety_shield.d_drop=1.50"
  "env.safety_shield.catchup_action=0.40"
  "env.safety_shield.lateral_tol=0.035"
  "env.safety_shield.lateral_turn_gain=0.30"
  "env.safety_shield.lateral_turn_clip=0.08"
  "env.safety_shield.lateral_velocity_gain=0.10"
  "env.safety_shield.centerline_turn_gain=0.25"
  "env.safety_shield.centerline_turn_clip=0.08"
  "env.safety_shield.boundary_turn_gain=0.80"
  "env.safety_shield.boundary_turn_clip=0.18"
  "env.safety_shield.first_follower_centerline_gain=0.0"
  "env.safety_shield.first_follower_centerline_clip=0.0"
  "env.safety_shield.forward_bias_gain=0.0"
  "env.safety_shield.forward_bias_clip=0.0"
  "env.safety_shield.forward_bias_min_command=0.30"
  "env.safety_shield.forward_bias_speed_margin=-0.02"
  "env.safety_shield.forward_bias_min_gap=1.45"
  "env.safety_shield.forward_bias_leader_gain_scale=2.5"
  "env.safety_shield.forward_bias_leader_clip_scale=2.0"
)

COMMON_OVERRIDES+=(
  "env.safety_shield.expert_path_tracking_enabled=false"
  "env.safety_shield.expert_blend_ratio=0.0"
  "env.safety_shield.expert_blend_end_ratio=0.0"
  "env.safety_shield.expert_blend_hold_steps=0"
  "env.safety_shield.expert_blend_decay_steps=0"
  "env.algorithm.action_fdi_longitudinal_only=true"
)
if [[ -n "${COMMON_EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra_items <<< "${COMMON_EXTRA_OVERRIDES}"
  COMMON_OVERRIDES+=("${extra_items[@]}")
fi

case "${ALGORITHM_LABEL}" in
  mappo)
    ALGO_OVERRIDES=("env.algorithm.algorithm=mappo" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=false" "env.algorithm.happo_share_actor=false" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=false") ;;
  happo_no_meta)
    ALGO_OVERRIDES=("env.algorithm.algorithm=happo" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=false" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=false") ;;
  happo_base_shared)
    ALGO_OVERRIDES=("env.algorithm.algorithm=happo" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=false") ;;
  happo_meta)
    ALGO_OVERRIDES=("env.algorithm.algorithm=happo" "env.algorithm.enable_teacher=true" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=false") ;;
  happo_ams_only)
    ALGO_OVERRIDES=("env.algorithm.algorithm=happo" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=true") ;;
  happo_meta_ams)
    ALGO_OVERRIDES=("env.algorithm.algorithm=happo" "env.algorithm.enable_teacher=true" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=true") ;;
  harl_mappo_shared)
    ALGO_OVERRIDES=("env.algorithm.algorithm=mappo" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=false" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=ppo" "env.algorithm.happo_ams_enabled=false") ;;
  harl_haa2c)
    ALGO_OVERRIDES=("env.algorithm.algorithm=haa2c" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=a2c" "env.algorithm.happo_ams_enabled=false") ;;
  harl_hatrpo)
    ALGO_OVERRIDES=("env.algorithm.algorithm=hatrpo" "env.algorithm.enable_teacher=false" "env.algorithm.happo_use_factor=true" "env.algorithm.happo_share_actor=true" "env.algorithm.happo_actor_update_mode=trpo" "env.algorithm.happo_ams_enabled=false" "env.algorithm.happo_trpo_kl_threshold=0.01" "env.algorithm.happo_trpo_cg_iters=10" "env.algorithm.happo_trpo_damping=0.1" "env.algorithm.happo_trpo_line_search_steps=10") ;;
  *) echo "[ERROR] unsupported ALGORITHM_LABEL=${ALGORITHM_LABEL}" >&2; exit 5 ;;
esac

mkdir -p "${OUTPUT_ROOT}"
{
  echo "checkpoint=${CHECKPOINT}"
  echo "algorithm=${ALGORITHM_LABEL}"
  echo "levels=off light easy_0 easy_a easy_b med_a med_b hard_a hard_b"
  echo "eval_num_envs=${EVAL_NUM_ENVS}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "video_steps=${VIDEO_STEPS}"
  echo "seed=${SEED}"
  echo "path=start_x:-7.75,length:6.0,start_y:-0.065,end_y:-1.160"
  echo "road_edges=near:0.47,far:-1.72"
  echo "created=$(date --iso-8601=seconds)"
} > "${OUTPUT_ROOT}/manifest.txt"

LEVELS=(off light easy_0 easy_a easy_b med_a med_b hard_a hard_b)
BASE_LEVELS=(off light easy easy easy medium medium hard hard)
FDI_POS=(0.0 0.75 1.00 1.25 1.50 1.75 2.00 2.75 4.00)
FDI_ACC=(0.0 0.10 0.18 0.25 0.35 0.45 0.50 0.75 1.30)
DOS_RATE=(0.0 0.02 0.035 0.05 0.07 0.09 0.10 0.12 0.18)
if [[ -n "${SINGLE_LEVEL:-}" ]]; then
  LEVELS=("${SINGLE_LEVEL}")
  BASE_LEVELS=("${SINGLE_BASE_LEVEL:?}")
  FDI_POS=("${SINGLE_FDI_POS:?}")
  FDI_ACC=("${SINGLE_FDI_ACC:?}")
  DOS_RATE=("${SINGLE_DOS_RATE:?}")
fi
printf 'actual_levels=%s\n' "${LEVELS[*]}" >> "${OUTPUT_ROOT}/manifest.txt"

# Validate all explicit/single-level budgets and retain a decimal point when
# Hydra reconstructs float-valued config fields (e.g. an off budget of 0).
for idx in "${!LEVELS[@]}"; do
  for array_name in FDI_POS FDI_ACC DOS_RATE; do
    declare -n budget_values="${array_name}"
    value="${budget_values[$idx]}"
    if [[ ! "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "[ERROR] invalid nonnegative budget ${array_name}[${idx}]=${value}" >&2
      exit 2
    fi
    [[ "${value}" == *.* ]] || budget_values[$idx]="${value}.0"
    unset -n budget_values
  done
done

if [[ "${DIAGNOSTIC_CLEAN_ONLY}" != "0" ]]; then
  if [[ "${DIAGNOSTIC_CLEAN_ONLY}" != "1" || "${LEVELS[*]}" != "off" || "${BASE_LEVELS[*]}" != "off" ]]; then
    echo "[ERROR] diagnostic export is restricted to a single off/clean rollout" >&2
    exit 2
  fi
  echo "[LADDER] diagnostic-only clean evidence; cannot promote a model or unlock attacks"
fi

for idx in "${!LEVELS[@]}"; do
  level="${LEVELS[$idx]}"
  base_level="${BASE_LEVELS[$idx]}"
  level_dir="${OUTPUT_ROOT}/${level}"
  mkdir -p "${level_dir}/video"
  enabled=true
  enable_args=(--enable_attack_eval)
  if [[ "${base_level}" == "off" ]]; then enabled=false; enable_args=(); fi
  ATTACK_OVERRIDES=(
    "env.algorithm.enable_attack=${enabled}"
    "env.algorithm.attack_level=${base_level}"
    "env.algorithm.attack_mode=profile"
    "env.algorithm.attack_curriculum_warmup_updates=0"
    "env.algorithm.attack_seed=${ATTACK_SEED}"
    "env.algorithm.attack_strength_start_scale=1.0"
    "env.algorithm.attack_strength_ramp_updates=0"
    "env.algorithm.attack_clean_env_fraction=0.0"
  )
  if [[ "${FDI_POS[$idx]}" != "-1" ]]; then
    ATTACK_OVERRIDES+=(
      "env.algorithm.max_fdi_pos=${FDI_POS[$idx]}"
      "env.algorithm.max_fdi_acc=${FDI_ACC[$idx]}"
      "env.algorithm.max_dos_rate=${DOS_RATE[$idx]}"
    )
  fi

  echo "[LADDER] aggregate evaluation: ${level}"
  if [[ -n "${VERIFIED_EVAL_DIR:-}" ]]; then
    "${PLOT_PYTHON}" -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts/tools"); from ackermann_validation import load_verified; row,spec=load_verified(sys.argv[1]); assert Path(spec["checkpoint"]) == Path(sys.argv[2])' "${VERIFIED_EVAL_DIR}" "${CHECKPOINT}"
    evaluation_dir="${VERIFIED_EVAL_DIR}"
  else
    evaluation_dir="${level_dir}/evaluation"
    "${PLOT_PYTHON}" scripts/tools/ackermann_validation.py run \
    --output "${evaluation_dir}" --checkpoint "${CHECKPOINT}" --steps "${EVAL_STEPS}" \
    --num-envs "${EVAL_NUM_ENVS}" --env-seed "${SEED}" --attack-seed "${ATTACK_SEED}" \
    --enabled "${enabled}" --level "${base_level}" \
    --pos "${FDI_POS[$idx]}" --acc "${FDI_ACC[$idx]}" --dos "${DOS_RATE[$idx]}" -- \
    "${SIM}/python.sh" scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs "${EVAL_NUM_ENVS}" \
    --seed "${SEED}" \
    --headless \
    --eval_checkpoints "${CHECKPOINT}" \
    --fresh_env_per_checkpoint \
    --eval_steps "${EVAL_STEPS}" \
    --warmup_steps 0 \
    --output_dir "${level_dir}/evaluation" \
    "${enable_args[@]}" \
    --kit_args="${KIT_ARGS}" \
    "${COMMON_OVERRIDES[@]}" \
    "${ALGO_OVERRIDES[@]}" \
      "${ATTACK_OVERRIDES[@]}"
  fi

  # Check the exact evaluation protocol even when borrowing verified CSVs.
  "${PLOT_PYTHON}" - "${evaluation_dir}" "${CHECKPOINT}" "${EVAL_STEPS}" \
    "${EVAL_NUM_ENVS}" "${SEED}" "${ATTACK_SEED}" "${enabled}" "${base_level}" \
    "${FDI_POS[$idx]}" "${FDI_ACC[$idx]}" "${DOS_RATE[$idx]}" "${DIAGNOSTIC_CLEAN_ONLY}" "${level_dir}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts/tools")
from ackermann_validation import load_verified, clean_passed, write_json
row, spec = load_verified(sys.argv[1])
expected = dict(checkpoint=str(Path(sys.argv[2]).resolve()), steps=int(sys.argv[3]),
                num_envs=int(sys.argv[4]), env_seed=int(sys.argv[5]), attack_seed=int(sys.argv[6]),
                enabled=sys.argv[7] == "true", level=sys.argv[8],
                pos=float(sys.argv[9]), acc=float(sys.argv[10]), dos=float(sys.argv[11]))
for key, value in expected.items():
    if spec[key] != value:
        raise ValueError(f"Borrowed evaluation protocol mismatch: {key}")
diagnostic = sys.argv[12] == "1"
if diagnostic:
    if spec["enabled"] or spec["level"] != "off" or any(spec[k] != 0 for k in ("pos", "acc", "dos")):
        raise ValueError("Diagnostic export must be attack-free with zero budgets")
    write_json(Path(sys.argv[13]) / "diagnostic_only.json",
               dict(diagnostic_only=True, attacks_unlocked=False, gate_passed=False,
                    observed_clean_passed=clean_passed(row), metrics=row))
if not spec["enabled"] and not clean_passed(row) and not diagnostic:
    raise SystemExit("Strict clean gate failed; attack ladder remains locked")
PY

  echo "[LADDER] real renderer + env-0 data: ${level}"
  if [[ "${RESUME_EXISTING}" == "1" && -s "${level_dir}/trajectory_env0.csv" ]] \
    && find "${level_dir}/video" -maxdepth 1 -name '*.mp4' -size +0c -print -quit | grep -q .; then
    echo "[LADDER] reuse completed renderer/video trajectory: ${level}"
  else
    "${SIM}/python.sh" scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Marl-Platoon-HAPPO-v0 \
    --num_envs 1 \
    --seed "${SEED}" \
    --checkpoint "${CHECKPOINT}" \
    --video \
    --video_length "${VIDEO_STEPS}" \
    --video_folder "${level_dir}/video" \
    --export_ros_replay \
    --ros_replay_path "${level_dir}/trajectory_env0.csv" \
    "${enable_args[@]}" \
    --headless \
    --kit_args="${KIT_ARGS}" \
    "env.viewer.eye=[0.0,-0.60,22.0]" \
    "env.viewer.lookat=[0.0,-0.60,0.0]" \
    "${COMMON_OVERRIDES[@]}" \
    "${ALGO_OVERRIDES[@]}" \
      "${ATTACK_OVERRIDES[@]}"
  fi

  step_csv="${evaluation_dir}/eval_steps_$(basename "${CHECKPOINT}" .pt).csv"
  "${PLOT_PYTHON}" scripts/tools/plot_spatial_attack_level.py "${step_csv}" \
    --level "${level}" --output "${level_dir}/reward_attack_safety.png"
  "${PLOT_PYTHON}" scripts/tools/plot_ros_replay_topdown.py "${level_dir}/trajectory_env0.csv" \
    --title "Spatial same-path lane-change — attack ${level}" \
    --output "${level_dir}/topdown_true_xy.png"
done

find "${OUTPUT_ROOT}" -type f ! -name sha256sums.txt -print0 | sort -z | xargs -0 sha256sum > "${OUTPUT_ROOT}/sha256sums.txt"
echo "[LADDER] complete: ${OUTPUT_ROOT}"
