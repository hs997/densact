# Debug Notes

## Current Goal

Continue debugging `Isaac-Marl-Platoon-HAPPO-v0` play/training so the HAPPO-controlled platoon leader actually moves forward instead of staying near zero velocity.

## Context

- Workspace: `/home/cnc/SSD_1T/xzw/IsaacLab-main`
- Task: `Isaac-Marl-Platoon-HAPPO-v0`
- Main run/checkpoint inspected:
  - `logs/rsl_rl/platoon_happo/2026-06-16_20-54-42/model_final.pt`
- Relevant files:
  - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - `source/my_exts/marl_platoon/wrappers.py`
  - `source/my_exts/marl_platoon/algorithms/router.py`
  - `source/my_exts/marl_platoon/algorithms/happo/runner.py`
  - `source/my_exts/marl_platoon/algorithms/happo/actor.py`
  - `Melodic_wheeltec_robot_src_250707/src/turn_on_wheeltec_robot/urdf/senior_4wd_bs_robot.urdf`

## Latest Symptom

Long training after speed/logging changes showed unstable learning:

- RSL log around iteration `20117/3020000` reported `Episode_Termination/reset_on_bad_ori = 1.0000`.
- HAPPO update reported very large critic values, e.g. `value_loss ~= 1.84e9` and `critic_grad_norm ~= 9.07e7`.
- Latest CSV tail from `logs/rsl_rl/platoon_happo/2026-06-17_23-42-42/platoon_metrics.csv` showed:
  - command speed around `0.89-0.91 m/s`
  - leader speed often negative, around `-0.61` to `-0.78 m/s`
  - speed absolute error around `0.95-0.99`
  - physical cost around `18-20`
  - attack mode `cagan`, attack objective clipped at `20.0`
  - shield trigger rate near `1.0`

Interpretation: the task is running, but the training distribution has become too aggressive and numerically unstable.

Before the wheel-axis adapter, play log showed:

- `Loaded task-local HAPPO state and enabled deterministic HAPPO play mode.`
- `[Platoon HAPPO] action override active: joint_actions=(1, 6, 4), flat_action_dim=24`
- `command_vel[0]` around `0.796 m/s`
- `actual_vel[0]` near zero
- `leader_motion` reward near zero
- Carb GPU warning appeared, but it is likely unrelated performance noise.

After adding the URDF wheel-axis adapter, the leader no longer stays near zero velocity, but the current checkpoint still drives too slowly:

- step 20: `actual_vel[0].x = 0.150 m/s`
- step 80: `actual_vel[0].x = 0.108 m/s`
- step 140: `actual_vel[0].x = 0.129 m/s`
- command remains about `0.796 m/s`

## Findings

- HAPPO action override path is active.
- The newer checkpoint `2026-06-16_20-54-42/model_final.pt` contains `platoon_happo_state`.
- Older inspected checkpoint `2026-05-26_19-30-03/model_final.pt` did not contain `platoon_happo_state`, so use the exact run path when debugging.
- The `senior_4wd_bs_robot.urdf` has opposite wheel joint axes:
  - left wheels use axis `0 -1 0`
  - right wheels use axis `0 1 0`
- Forward driving likely requires a signed left/right wheel pattern, not identical signs for all four wheels.
- Deterministic HAPPO actor output for the leader in the inspected checkpoint was small and mixed/same-sign, for example approximately:
  - `[-0.08, -0.21, -0.11, -0.12]`
- This can cancel translation or produce weak wheel commands, explaining near-zero chassis velocity.
- IsaacLab action term joint order is `['lb_wheel_joint', 'lf_wheel_joint', 'rb_wheel_joint', 'rf_wheel_joint']`.
- With this URDF, the correct adapter sign pattern is `[1, 1, -1, -1]` for each agent.
- After the adapter, the leader processed action at step 20 became approximately `[-0.93, -2.57, 1.44, 1.45]`, and actual wheel velocities closely followed the processed targets.
- Therefore actuator tracking is working; the remaining speed gap is mainly because the current checkpoint's HAPPO raw action magnitude is small, not because IsaacLab is ignoring velocity targets.
- The post-change instability is not caused by `Mean reward: 0.00`; that value is expected for the frozen outer PPO path. The real red flags are `reset_on_bad_ori=1.0`, negative leader velocity, and exploding HAPPO critic loss.
- Likely causes of the instability:
  - `TARGET_SPEED_RANGE=(0.8, 1.0)` and `WHEEL_ACTION_SCALE=20.0` were a large jump from the previous setup.
  - `happo_action_clip=1.0` allows saturation to `20 rad/s` wheel targets.
  - `happo_action_warmup_updates=0` starts full HAPPO/attack/shield control immediately, with no low-speed stabilization period.
  - `enable_attack=True`, `attack_mode=cagan`, and medium attack budgets make early learning harsher.
  - `termination_bad_heading_chain()` terminates whenever any robot heading x component is below `0`, so short yaw flips or attack-induced spins make `reset_on_bad_ori` dominate.

## Changes Made

- In `config.py`, `reward_leader_velocity_matching()` debug print was delayed until `common_step_counter >= 20`.
- That debug now prints:
  - command velocity
  - actual root velocity
  - leader raw wheel action
  - leader processed wheel action
  - leader wheel joint velocity
  - error mean and reward mean
- In `wrappers.py`, added optional HAPPO action diagnostics when `happo_log_level == "debug"` and action steps reach 20:
  - raw leader HAPPO action
  - clipped leader HAPPO action
  - final env leader action after pipeline postprocessing/shield
- In `wrappers.py`, added `happo_urdf_wheel_sign_adapter` at the final env-action出口, after pipeline attacker/shield postprocessing.
  - The adapter resolves wheel joint names and applies `[1, 1, -1, -1]` for the senior 4WD URDF.
  - This keeps task-internal HAPPO/shield semantics as "same sign = same driving direction" and only converts at the IsaacLab joint target boundary.
- In `config.py`, declared env-side HAPPO fields including `happo_log_level`, `happo_action_clip`, `happo_action_warmup_updates`, and `happo_urdf_wheel_sign_adapter`.
- In `config.py`, leader motion debug now prints at steps `20, 80, 140, ...` instead of only once.
- Follow-up training cleanup:
  - `PlatoonHAPPOEnvCfg.__post_init__()` now explicitly sets `self.algorithm.happo_urdf_wheel_sign_adapter = True`.
  - Wrapper startup log now prints `wheel_axis_adapter=True/False` so new training logs prove which action convention was used.
  - No temporary post-adapter action gain was added; retraining should learn under the corrected wheel-axis convention directly.
- Speed/data logging update for paper experiments:
  - `TARGET_SPEED_RANGE` is now `(0.8, 1.0)` m/s.
  - Wheel action scale is now `20.0`, wheel velocity limit is now `20.0 rad/s`, and `happo_action_clip` is now `1.0`.
  - `happo_action_warmup_updates` is now `0`, so training starts with task-local HAPPO actions instead of executing random outer PPO warmup actions.
  - Wrapper clamps final flat actions immediately before `env.step()`; this also bounds attacker/shield postprocessed actions.
  - `PlatoonAlgorithmRouter` writes `platoon_metrics.csv` in each run directory with update/time, command/actual speed, spacing/lateral errors, reward terms, HAPPO losses, attack stats, physical costs, Teacher metrics, and shield stats.
  - CSV now also includes `termination_time_out` and `termination_reset_on_bad_ori` for automatic stage promotion checks.
  - Added `scripts/tools/plot_platoon_metrics.py` to generate paper-style plots from the CSV.
  - Added `scripts/tools/check_platoon_stage.py` to check promotion criteria from recent CSV rows.
  - Added `scripts/tools/run_platoon_curriculum.sh` to run staged curriculum training automatically:
    1. `stage1_stable`: low-speed, no attack.
    2. `stage2_speedup`: higher speed, no attack.
    3. `stage3_easy_attack`: easy profile attack.
    4. `stage4_cagan`: medium CA-GAN attack.

## Verification

Syntax check passed:

```bash
python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/wrappers.py
```

Latest syntax check also passed:

```bash
python3 -m py_compile source/my_exts/marl_platoon/wrappers.py source/my_exts/marl_platoon/algorithms/router.py source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/tasks/__init__.py source/my_exts/marl_platoon/tasks/platoon/agents.py scripts/tools/plot_platoon_metrics.py
```

Short train smoke test passed:

```bash
PYTHONPATH=/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_rl:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_tasks:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/my_exts:/home/cnc/SSD_1T/xzw/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 /home/cnc/SSD_1T/xzw/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Marl-Platoon-HAPPO-v0 --num_envs 1 --headless --max_iterations 1 --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

Key output:

- startup line includes `wheel_axis_adapter=True, action_clip=1.0, warmup_updates=0`
- step 20 action manager raw action is bounded, e.g. `[-0.017, 1.0, 1.0, -1.0]`
- CSV was written to `logs/rsl_rl/platoon_happo/2026-06-17_23-38-51/platoon_metrics.csv`
- `python3` currently lacks `matplotlib`; plotting script now reports `python3 -m pip install pandas matplotlib` if dependencies are missing.

Play verification:

```bash
PYTHONPATH=/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_rl:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_tasks:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/my_exts:/home/cnc/SSD_1T/xzw/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 /home/cnc/SSD_1T/xzw/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Marl-Platoon-HAPPO-v0 --num_envs 1 --checkpoint /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-16_20-54-42/model_final.pt --headless --enable_cameras --video --video_length 150 --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

Key output after adapter:

- `[Platoon HAPPO] URDF wheel-axis adapter active: signs=[[1.0, 1.0, -1.0, -1.0], ...]`
- step 20: `actual_vel[0]=[0.1501, -0.0006]`, reward mean `0.124`
- step 80: `actual_vel[0]=[0.1077, 0.0037]`, reward mean `0.093`
- step 140: `actual_vel[0]=[0.1285, -0.0036]`, reward mean `0.108`

## Next Steps

Parameter recommendation after comparing the stable 3000-iteration run with the unstable high-speed run:

- Do not train from scratch with `TARGET_SPEED_RANGE=(0.8, 1.0)`, `WHEEL_ACTION_SCALE=20.0`, `happo_action_clip=1.0`, `happo_action_warmup_updates=0`, and `medium+cagan` attack all at once.
- A better stable baseline is close to the old run but with the URDF wheel adapter:
  - `TARGET_SPEED_RANGE=(0.6, 0.8)`
  - `WHEEL_ACTION_SCALE=12.0` or at most `14.0`
  - `WHEEL_VELOCITY_LIMIT=12.0`
  - `happo_action_clip=0.4`
  - `happo_action_warmup_updates=10`
  - start with `enable_attack=False`
- After stable tracking, resume and increase in stages:
  - speed stage: `TARGET_SPEED_RANGE=(0.7, 0.9)`, `WHEEL_ACTION_SCALE=14.0`, `happo_action_clip=0.5`
  - easy attack stage: `enable_attack=True`, `attack_level=easy`, `attack_mode=profile`
  - full attack stage: `attack_level=medium`, `attack_mode=cagan`, only after `reset_on_bad_ori` is low and critic loss is finite/stable.
- Monitor promotion criteria in `platoon_metrics.csv`: positive `leader_speed_mean`, decreasing `speed_error_abs_mean`, non-diverging `gap_error_abs_mean`, low `reset_on_bad_ori`, finite `value_loss`, and bounded `critic_grad_norm`.

1. Keep the wheel-axis adapter enabled for this URDF.
2. Retrain or fine-tune HAPPO with the adapter enabled; the old checkpoint was trained under the wrong wheel sign convention and outputs weak actions.
3. If quick checkpoint salvage is needed, test a configurable post-adapter drive gain or higher `JointVelocityActionCfg.scale`, but treat that as a temporary diagnostic because it changes the action semantics.
4. For Hydra debug overrides, use env cfg path `algorithm.happo_log_level=debug`; `agent.platoon.happo_log_level=debug` does not affect the wrapper because the wrapper reads `env_cfg.algorithm`.
5. If retraining with adapter still cannot approach the `0.5-0.8 m/s` command, then inspect actuator limits/friction/contact settings. Current evidence does not point there because wheel targets and actual wheel velocities match closely.

Recommended retraining command:

```bash
PYTHONPATH=/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_rl:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_tasks:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/my_exts:/home/cnc/SSD_1T/xzw/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 /home/cnc/SSD_1T/xzw/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Marl-Platoon-HAPPO-v0 --num_envs 2048 --headless --max_iterations 2000 --kit_args="--/rtx/verifyDriverVersion/enabled=false"
```

Expected startup line should include `wheel_axis_adapter=True`.

Recommended visualization command after training:

```bash
python3 scripts/tools/plot_platoon_metrics.py logs/rsl_rl/platoon_happo/<RUN_DIR>/platoon_metrics.csv --smooth 10
```

The script saves figures under `logs/rsl_rl/platoon_happo/<RUN_DIR>/figures/` and writes `summary.csv`.

One-click curriculum command:

```bash
scripts/tools/run_platoon_curriculum.sh
```

Optional resume from an existing stable run:

```bash
START_RUN=2026-06-17_17-34-19 START_CHECKPOINT=model_final.pt scripts/tools/run_platoon_curriculum.sh
```

2026-06-18 Hydra override fix:

- IsaacLab task configs are registered in Hydra under two top-level keys: `env` and `agent`.
- The curriculum script previously used bare overrides such as `commands.base_velocity.ranges.lin_vel_x=...`, `algorithm.happo_action_clip=...`, and `actions.joint_vel_1.scale=...`.
- That causes `Key 'commands' is not in struct` because `commands` is inside the environment config, not the Hydra root.
- Fixed `scripts/tools/run_platoon_curriculum.sh` so all environment overrides use `env.` prefixes:
  - `env.commands.base_velocity.ranges.lin_vel_x=[...]`
  - `env.algorithm.*`
  - `env.actions.joint_vel_*.scale=...`
- `bash -n scripts/tools/run_platoon_curriculum.sh` passes after the fix.
- The user-facing command is unchanged:

```bash
START_RUN=2026-06-17_17-34-19 START_CHECKPOINT=model_final.pt scripts/tools/run_platoon_curriculum.sh
```

2026-06-18 CUDA OOM fix and validation:

- Failure mode: `torch.OutOfMemoryError` occurred around HAPPO update 20 with `NUM_ENVS=4096`; RTX 3080 10GB had only about `40 MiB` free while IsaacSim plus PyTorch were active.
- Root cause: 4096 envs leaves too little headroom for task-local HAPPO updates. The old HAPPO runner also evaluated action log-probs on the full rollout batch when updating the sequential HAPPO factor.
- Fixes:
  - `scripts/tools/run_platoon_curriculum.sh` default `NUM_ENVS` changed from `4096` to `2048`.
  - Script now exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Script now supports `STOP_AFTER_STAGE=stage1_stable` for real single-stage smoke tests.
  - HAPPO factor update now uses stored rollout old log-probs and computes new log-probs in chunks.
  - Added `happo_num_mini_batches=16` and `happo_factor_chunk_size=4096` to lower update-time memory peaks.
- Validation:
  - `python3 -m py_compile ...` and `bash -n scripts/tools/run_platoon_curriculum.sh` pass.
  - Real 3-iteration stage1 resume test passed with 2048 envs and wrote CSV/checkpoints:
    - run: `logs/rsl_rl/platoon_happo/2026-06-18_12-08-28_stage1_stable`
  - Real 22-iteration stage1 resume test passed with 2048 envs, reached HAPPO update 22, saved `model_final.pt`, and wrote 22 CSV rows:
    - run: `logs/rsl_rl/platoon_happo/2026-06-18_12-09-33_stage1_stable`
    - final observed speed: about `13712 steps/s`
    - final `value_loss` about `5.68`, `critic_grad_norm` about `14.64`
    - post-run GPU memory returned to about `857 MiB` used / `9148 MiB` free.
- Behavioral note: resuming from `2026-06-17_17-34-19/model_final.pt` still gives high `termination_reset_on_bad_ori` in the short validation window because that checkpoint was trained before the corrected wheel-axis/action curriculum. This is now a policy-quality issue, not an execution/OOM issue. For clean curriculum training, prefer starting stage1 from scratch unless checkpoint salvage is specifically needed.

Useful environment variables:

- `NUM_ENVS=2048`
- `STAGE1_ITERS=3000`
- `STAGE2_ITERS=2000`
- `STAGE3_ITERS=2000`
- `STAGE4_ITERS=3000`
- `AUTO_PROMOTE=0` to run each stage without automatic PASS/FAIL gating.

The curriculum script stops after a stage if `check_platoon_stage.py` detects negative/weak leader speed, high speed error, high `termination_reset_on_bad_ori`, exploding `value_loss`, or exploding `critic_grad_norm`.

2026-06-18 stage4 instability diagnosis and fix:

- Live run inspected:
  - command: `--num_envs 32 --run_name stage4_cagan --resume --load_run 2026-06-18_16-16-14_stage3_easy_attack`
  - it was intentionally/explicitly running with 32 envs, so speed around `400 steps/s` is expected and much slower than the RTX 3080 2048-env validation.
- Latest curriculum results before the fix:
  - stage1 `2026-06-18_13-09-14_stage1_stable`: PASS
  - stage2 `2026-06-18_14-58-28_stage2_speedup`: PASS
  - stage3 `2026-06-18_16-16-14_stage3_easy_attack`: PASS
  - stage4 `2026-06-18_17-31-46_stage4_cagan`: bad despite old checker saying PASS.
- Old stage4 last-window summary:
  - `leader_speed_mean ~= 0.11`
  - `speed_error_abs_mean ~= 0.80`
  - `gap_error_max_abs ~= 8.19`
  - `termination_reset_on_bad_ori ~= 0.31`
  - `shield_trigger_rate ~= 0.99`
  - attack was still in `profile` warmup, not true CA-GAN yet.
- Key cause: stage4 jumped to `0.8-1.0 m/s + medium attack`, and medium `max_fdi_acc=1.5` was applied directly to normalized actions with `happo_action_clip=0.6`. The action perturbation was larger than the action range, so it randomized/saturated actions before the policy was robust.
- Current bad stage4 run was stopped to avoid wasting GPU time; the stage3 checkpoint remains available.
- Fixes made:
  - Added env/agent config fields `max_fdi_pos`, `max_fdi_acc`, and `max_dos_rate`.
  - Router now treats negative values as “use attack preset” and uses nonnegative values as explicit overrides.
  - Curriculum changed from 4 stages to 5 stages:
    1. `stage1_stable`
    2. `stage2_speedup`
    3. `stage3_easy_attack` with reduced action attack: `max_fdi_acc=0.20`, `max_dos_rate=0.03`
    4. `stage4_medium_profile`: `speed=[0.75,0.9]`, `attack_mode=profile`, `max_fdi_acc=0.25`, `max_dos_rate=0.05`
    5. `stage5_cagan`: `speed=[0.8,0.95]`, `attack_mode=cagan`, `max_fdi_acc=0.30`, `max_dos_rate=0.06`, `attack_curriculum_warmup_updates=500`
  - `run_platoon_curriculum.sh` now warns when `NUM_ENVS < 512`.
  - `run_platoon_curriculum.sh` default `NUM_ENVS` is restored to `2048`; a local/default value of `32` was the direct reason for the observed `~400 steps/s` speed.
  - `run_platoon_curriculum.sh` now supports `START_STAGE` so training can resume directly from a later curriculum stage.
  - `check_platoon_stage.py` now checks speed, speed error, gap mean/max, lateral error, bad heading reset, shield trigger rate, value loss, and critic grad norm.
  - Old bad stage4 CSV now correctly FAILs with:
    - `leader_speed_mean < 0.25`
    - `speed_error_abs_mean > 0.7`
    - `gap_error_max_abs > 4.0`
    - `shield_trigger_rate > 0.97`
    - `termination_reset_on_bad_ori > 0.25`
- Verification:
  - `python3 -m py_compile ...` passed.
  - `bash -n scripts/tools/run_platoon_curriculum.sh` passed.
  - New checker still passes stage1/stage2/stage3 historical CSVs and fails the bad stage4 CSV.
  - Real smoke test passed for new `stage4_medium_profile` overrides using stage3 checkpoint:
    - run: `logs/rsl_rl/platoon_happo/2026-06-18_17-41-59_stage4_medium_profile_smoke`
    - loaded `2026-06-18_16-16-14_stage3_easy_attack/model_final.pt`
    - ran 3 HAPPO updates and saved `model_final.pt`
    - CSV confirms `attack_max_fdi_acc=0.25` and `attack_max_dos_rate=0.05`.
    - post-run GPU memory returned to about `855 MiB` used / `9151 MiB` free.

Recommended resume from the good stage3 checkpoint into the new gentler stage4:

```bash
cd /home/cnc/SSD_1T/xzw/IsaacLab-main
START_STAGE=stage4_medium_profile START_RUN=2026-06-18_16-16-14_stage3_easy_attack START_CHECKPOINT=model_final.pt scripts/tools/run_platoon_curriculum.sh
```

For full retraining from scratch, use:

```bash
cd /home/cnc/SSD_1T/xzw/IsaacLab-main
scripts/tools/run_platoon_curriculum.sh
```

2026-06-19 stage5_cagan checkpoint confirmation:

- Run inspected: `logs/rsl_rl/platoon_happo/2026-06-18_19-12-27_stage5_cagan`.
- CSV has 3000 rows, update column `1..3000`; RSL checkpoint number maps approximately as `model_N.pt -> csv update N - 8996`.
- `model_final.pt` is bad and should not be used for qualitative demos:
  - last-window `leader_speed_mean ~= 0.09`
  - `termination_reset_on_bad_ori = 1.0`
  - `value_loss ~= 3.16e6`, `critic_grad_norm ~= 2.64e5`
  - debug play showed raw leader action around `[-8.1, -13.8, 2.1, 6.8]`, clipped/saturated, and actual leader speed near zero.
- `model_10950.pt` is a valuable stable checkpoint:
  - around csv update `1954`, local metrics are healthy: `leader_speed_mean ~= 0.50`, `speed_error_abs_mean ~= 0.38`, `gap_error_abs_mean ~= 0.35`, `lateral_error_abs_mean ~= 0.19`, `termination_reset_on_bad_ori = 0`, `value_loss ~= 2.5`.
  - 50-row practical rolling-window ranking put the best stable band around RSL iter `10905..10914`, so `model_10950.pt` sits on the same stable plateau.
  - short play with training-consistent overrides (`happo_action_clip=0.6`, wheel action scale `14.0`, command speed `[0.8,0.95]`) gave leader speeds:
    - step 20: `0.730 m/s`
    - step 80: `0.604 m/s`
    - step 140: `0.608 m/s`
  - raw leader action was around `[-1.05, -0.91, -1.23, -0.97]`, clipped to `-0.6`, then wheel-axis adapted to `[-0.6, -0.6, 0.6, 0.6]`; processed wheel targets were `[-8.4, -8.4, 8.4, 8.4]`.
- `model_best.pt` is not a symlink/copy of `model_10950.pt`:
  - `model_best.pt` size/hash differs from `model_10950.pt`.
  - short play with the same training-consistent overrides was very similar or slightly better at the first sample:
    - step 20: `0.762 m/s`
    - step 80: `0.602 m/s`
    - step 140: `0.613 m/s`
  - raw leader action was around `[-1.43, -1.50, -0.98, -0.57]`, clipped/wheel-axis adapted to a sane forward-drive pattern.
- Important play/eval caveat:
  - Plain `play.py --checkpoint ...` does not automatically replay the stage5 Hydra overrides saved in `params/env.yaml`.
  - Without explicit overrides, current source defaults use `happo_action_clip=1.0`, `WHEEL_ACTION_SCALE=20.0`, and `TARGET_SPEED_RANGE=(0.8,1.0)`, which differs from stage5 training (`clip=0.6`, action scale `14.0`, speed `[0.8,0.95]`) and can exaggerate saturation/lateral drift.
- Current interpretation:
  - The original "leader not moving" issue is resolved for good mid-stage checkpoints; it remains only for the collapsed final checkpoint.
  - The current qualitative issue is now "the platoon moves but one or more followers can drift laterally."
  - This matches CSV: the good window still has `lateral_error_abs_mean ~= 0.18..0.22`, so individual vehicles can visibly run toward the side even when leader speed and bad-heading resets are good.

Recommended play command for stable visual checks:

```bash
cd /home/cnc/SSD_1T/xzw/IsaacLab-main
CHECKPOINT=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-18_19-12-27_stage5_cagan/model_10950.pt
PYTHONPATH=/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_rl:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/isaaclab_tasks:/home/cnc/SSD_1T/xzw/IsaacLab-main/source/my_exts:/home/cnc/SSD_1T/xzw/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 \
/home/cnc/SSD_1T/xzw/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Marl-Platoon-HAPPO-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --kit_args="--/rtx/verifyDriverVersion/enabled=false" \
  env.algorithm.happo_action_clip=0.6 \
  env.commands.base_velocity.ranges.lin_vel_x=[0.8,0.95] \
  env.actions.joint_vel_1.scale=14.0 \
  env.actions.joint_vel_2.scale=14.0 \
  env.actions.joint_vel_3.scale=14.0 \
  env.actions.joint_vel_4.scale=14.0 \
  env.actions.joint_vel_5.scale=14.0 \
  env.actions.joint_vel_6.scale=14.0 \
  "env.viewer.eye=[8.0,-10.0,6.0]" \
  "env.viewer.lookat=[-2.5,0.0,0.4]"
```

Next debugging direction:

- Prefer `model_10950.pt` or `model_best.pt` for demos; avoid `model_final.pt`.
- For lateral drift, next code-level work should focus on:
  - increasing lateral/yaw/heading alignment terms, not just leader speed;
  - adding or strengthening action smoothness / action-rate / jerk penalties;
  - considering a follower lateral/yaw safety projection in the shield, because the current shield only handles longitudinal gap/brake/catch-up behavior;
  - adding eval metrics/video labels for per-robot lateral error so the drifting robot is identifiable instead of relying only on aggregate `lateral_error_abs_mean`.

2026-06-19 status clarification:

- Completed: confirmed the current `stage5_cagan` result, identified `model_10950.pt` and `model_best.pt` as usable stable checkpoints, verified them with training-consistent play overrides, and documented the required play command.
- Not yet completed: code-level changes for the remaining lateral drift issue. The next implementation pass should modify reward/shield/metrics around lateral and heading stability.

2026-06-19 lateral drift mitigation implemented:

- Files changed for the lateral-drift pass:
  - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - `source/my_exts/marl_platoon/algorithms/shield.py`
  - `source/my_exts/marl_platoon/algorithms/router.py`
  - `scripts/tools/check_platoon_stage.py`
  - `scripts/tools/plot_platoon_metrics.py`
- Reward changes:
  - `reward_lateral_penalty_hard()` now has a `0.05 m` dead-zone and a stronger quadratic penalty after `0.25 m` visible lateral drift.
  - Added `reward_lateral_velocity_penalty()` to penalize follower side-slip / lateral velocity mismatch.
  - Added `reward_heading_alignment_penalty()` to penalize yaw/heading disagreement between the platoon and between adjacent vehicles.
  - Reward weights changed:
    - `action_rate`: `-0.06 -> -0.10`
    - `lateral_correct`: `-1.5 -> -3.0`
    - new `lateral_velocity`: `-0.8`
    - new `heading_align`: `-2.0`
- Shield changes:
  - Existing shield only handled longitudinal gap/catch-up and did not steer a drifting follower back to the predecessor centerline.
  - Added bounded lateral differential correction before the URDF wheel-axis adapter:
    - `lateral_tol=0.08`
    - `lateral_crit=0.35`
    - `lateral_turn_gain=0.70`
    - `lateral_turn_clip=0.18`
    - `lateral_velocity_gain=0.25`
  - Corrected the shield action convention for this URDF/HAPPO path:
    - pre-adapter negative same-sign commands drive forward.
    - `brake_action` default changed to `0.0`.
    - `catchup_action` default changed to `-0.35`.
  - New shield stats:
    - `shield_lateral_rate`
    - `shield_lateral_critical_rate`
    - `shield_lateral_turn_mean`
- Metrics/checking changes:
  - CSV now logs `lateral_error_abs_max`, `lateral_worst_pair`, `lateral_pair_1_abs_mean` through `lateral_pair_5_abs_mean`.
  - CSV now logs `heading_error_abs_mean`, `heading_error_abs_max`, `pair_heading_error_abs_mean`, and `pair_heading_error_abs_max`.
  - CSV now logs reward terms for `lateral_correct`, `lateral_velocity`, `heading_align`, and `action_rate`.
  - `check_platoon_stage.py` now optionally gates on `lateral_error_abs_max` when the column exists.
  - `plot_platoon_metrics.py` now plots lateral pair errors, heading errors, and new shield lateral stats.
- Verification:
  - Syntax check passed:

```bash
python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/algorithms/shield.py source/my_exts/marl_platoon/algorithms/router.py scripts/tools/check_platoon_stage.py scripts/tools/plot_platoon_metrics.py
```

  - Real one-iteration smoke test passed with `num_envs=1`, `happo_action_clip=0.6`, wheel scales `14.0`, and command speed `[0.8,0.95]`.
  - Smoke run: `logs/rsl_rl/platoon_happo/2026-06-19_03-00-13`.
  - Smoke output confirmed 14 reward terms are active, including `lateral_velocity` and `heading_align`.
  - Smoke CSV row confirmed new fields are written, e.g.:
    - `lateral_error_abs_max = 0.1317`
    - `lateral_worst_pair = 5`
    - `heading_error_abs_mean = 0.0519`
    - `pair_heading_error_abs_mean = 0.0841`
    - `shield_lateral_rate = 0.4375`
    - `shield_lateral_turn_mean = 0.1001`
  - Post-change `model_10950.pt` play validation passed for 250 steps using the stage5-compatible overrides (`happo_action_clip=0.6`, wheel scale `14.0`, speed `[0.8,0.95]`).
    - The run completed without shield/runtime errors.
    - Leader velocity samples stayed healthy: step 20 `0.730 m/s`, step 80 `0.604 m/s`, step 140 `0.608 m/s`, step 200 `0.624 m/s`.
    - New video path: `logs/rsl_rl/platoon_happo/2026-06-18_19-12-27_stage5_cagan/videos/play/rl-video-step-0.mp4`.
- Expected impact:
  - Existing `model_10950.pt` / `model_best.pt` can be played with the new shield projection, but the reward changes mainly affect retraining/fine-tuning.
  - For a clean evaluation, replay `model_10950.pt` with the same stage5 overrides and inspect whether the formerly drifting follower is corrected by the new lateral shield.
  - For best final behavior, resume/fine-tune from `model_10950.pt` or `model_best.pt` under the new lateral reward/shield settings rather than using `model_final.pt`.

2026-06-19 current result-data assessment:

- Meaningful performance data still comes from the completed 2026-06-18 curriculum/stage5 CSVs.
- The 2026-06-19 lateral-fix smoke run (`2026-06-19_03-00-13`) has only one CSV row, so it validates runtime wiring and new metrics, not final task performance.
- Stage progression from old data:
  - stage1 stable last50: `leader_speed_mean ~= 0.345`, `lateral_error_abs_mean ~= 0.048`, `reset_on_bad_ori = 0`.
  - stage2 speedup last50: `leader_speed_mean ~= 0.487`, `lateral_error_abs_mean ~= 0.168`, `reset_on_bad_ori = 0`.
  - stage3 easy attack last50: `leader_speed_mean ~= 0.411`, `lateral_error_abs_mean ~= 0.153`, `reset_on_bad_ori = 0`.
  - stage4 medium profile last50: `leader_speed_mean ~= 0.452`, `lateral_error_abs_mean ~= 0.228`, `reset_on_bad_ori = 0`.
- Stage5 stable checkpoint window (`model_10950.pt` vicinity) is usable but not perfect:
  - `command_speed_mean ~= 0.875`
  - `leader_speed_mean ~= 0.478`
  - `platoon_speed_mean ~= 0.487`
  - `speed_error_abs_mean ~= 0.388`
  - `gap_error_abs_mean ~= 0.352`
  - `lateral_error_abs_mean ~= 0.222`
  - `reset_on_bad_ori = 0`
  - `value_loss ~= 2.97`, `critic_grad_norm ~= 6.96`
  - Interpretation: stable, moves forward, no heading-reset collapse, but still only tracks about 55% of commanded speed and has visible lateral drift.
- Best stable 50-row window around RSL iter `10910` is slightly better laterally:
  - `leader_speed_mean ~= 0.478`
  - `speed_error_abs_mean ~= 0.390`
  - `gap_error_abs_mean ~= 0.353`
  - `lateral_error_abs_mean ~= 0.199`
  - `reset_on_bad_ori = 0`
  - `value_loss ~= 3.06`
- Stage5 final checkpoint is a failed endpoint:
  - last50 `leader_speed_mean ~= 0.091`
  - `reset_on_bad_ori = 1.0`
  - `value_loss ~= 3.16e6`
  - `critic_grad_norm ~= 2.64e5`
  - `leader_action_abs_mean ~= 11.3`
  - Interpretation: CA-GAN late-stage training destabilized the policy/critic; do not use `model_final.pt`.
- Overall assessment:
  - Current best usable result is good enough for qualitative demonstration using `model_10950.pt` or `model_best.pt`.
  - It is not yet a fully successful final experiment for a paper-quality claim because command speed tracking remains weak (`~0.48` actual vs `~0.875` command) and lateral drift remains visible (`~0.20-0.22 m` mean lateral error in the stable attack windows).
  - The new lateral reward/shield changes are the right next fix, but need a fine-tune/resume run before claiming data-level improvement.

Recommended next run after lateral fixes:

- Do not restart from scratch first.
- Do not resume from `model_final.pt`; it is collapsed.
- Start a short lateral-fix fine-tune from:
  - `logs/rsl_rl/platoon_happo/2026-06-18_19-12-27_stage5_cagan/model_10950.pt`
- Revised judgment after reviewing the proposed command:
  - The principle is correct: first stabilize lateral behavior at lower speed, with no attack or only light attack, before returning to medium CA-GAN.
  - Use a smaller `num_envs` first (`64`, or `32` if needed) to reduce RTX 3080 10GB OOM risk.
  - However, `attack_level=light` is not currently a supported preset; valid values are `off`, `easy`, `medium`, `hard`. Unknown values fall back to `medium`, so do not use `attack_level=light`.
  - `happo_action_clip=0.3` with wheel scale `10.0` is very conservative. It may reduce lateral drift, but it may also make speed tracking too weak. If speed collapses below `0.35-0.40 m/s`, use `happo_action_clip=0.35-0.4` and wheel scale `10-12`.
- Recommended first fine-tune budget: `300-500` iterations, then inspect the new CSV before running longer.
- Main success criterion for the fine-tune:
  - keep `leader_speed_mean >= 0.35` for the low-speed lateral stabilization pass, then recover to `>=0.45` before going back to high speed/attack
  - reduce `lateral_error_abs_mean` from `~0.20-0.22` toward `<0.15`
  - keep `termination_reset_on_bad_ori = 0`
  - keep `value_loss` and `critic_grad_norm` finite and non-exploding.

2026-06-20 latest run assessment:

- Latest run inspected: `logs/rsl_rl/platoon_happo/2026-06-19_14-39-00_stage5_cagan_light_from_10950`.
- It ran 800 CSV rows / updates and produced checkpoints `model_10950.pt` through `model_11749.pt`.
- The run used:
  - `num_envs=64`
  - command speed `[0.6,0.8]`
  - `happo_action_clip=0.3`
  - wheel scale `10.0`
  - `attack_level=light`, `attack_mode=cagan`, `max_fdi_acc=0.1`, `max_dos_rate=0.02`
- Important config issue:
  - `attack_level=light` is still not a supported preset in the code.
  - Because of fallback behavior, the run logged `attack_max_fdi_pos=5.0`, i.e. medium-style observation FDI amplitude, despite the run name saying light.
  - If light attack is desired, either add a real `light` preset in `router.py` or use `attack_level=easy` plus explicit `max_fdi_pos=1.0`, `max_fdi_acc=0.05..0.10`, `max_dos_rate=0.01..0.02`.
- Latest run result:
  - Final last50 fails `stage5_cagan` check.
  - Final last50:
    - `leader_speed_mean ~= 0.154`
    - `speed_error_abs_mean ~= 0.581`
    - `gap_error_abs_mean ~= 0.238`
    - `lateral_error_abs_mean ~= 0.428`
    - `lateral_error_abs_max ~= 1.78`
    - `reset_on_bad_ori = 0`
    - `shield_trigger_rate ~= 0.999`
    - `shield_lateral_rate ~= 0.910`
    - `shield_lateral_critical_rate ~= 0.581`
    - `value_loss ~= 24.6`, `critic_grad_norm ~= 36.4`
  - Interpretation: no catastrophic reset/critic explosion, but the platoon is too slow and lateral drift gets worse over training.
- Best 50-row window in this latest run is around updates `228..277`, nearest checkpoint `model_11200.pt`:
  - `leader_speed_mean ~= 0.151`
  - `speed_error_abs_mean ~= 0.575`
  - `lateral_error_abs_mean ~= 0.163`
  - `lateral_error_abs_max ~= 0.87`
  - `value_loss ~= 0.73`
  - `critic_grad_norm ~= 1.24`
  - It improves lateral mean compared with old stage5 (`~0.20-0.22`) but sacrifices too much speed.
- Main problems:
  - `happo_action_clip=0.3` and wheel scale `10.0` are too conservative for this checkpoint. Effective wheel command is capped at about `3 rad/s`, so leader speed collapses to about `0.15 m/s`.
  - Lateral shield is triggering almost all the time; with low action clip, the lateral correction consumes too much of the available action range and fights forward drive.
  - Unsupported `attack_level=light` leaves `max_fdi_pos=5.0`, so the run is not actually a clean light-attack experiment.
- Recommended improvement:
  1. Do not continue this latest run as final.
  2. For a no-attack lateral stabilization pass, retry from original `2026-06-18_19-12-27_stage5_cagan/model_10950.pt` with:
     - `happo_action_clip=0.4`
     - wheel scale `12.0`
     - speed `[0.6,0.8]`
     - `enable_attack=false`
     - reduce shield lateral override: `lateral_turn_gain=0.45`, `lateral_turn_clip=0.10`, `lateral_tol=0.12`.
  3. If adding light CA-GAN later, use `attack_level=easy` or implement `light`; also explicitly lower `max_fdi_pos` to `1.0`.
- Play recommendation:
  - To inspect the latest run's best lateral-but-slow point, use `2026-06-19_14-39-00_stage5_cagan_light_from_10950/model_11200.pt`.
  - For the best qualitative moving demo, still prefer the older `2026-06-18_19-12-27_stage5_cagan/model_10950.pt` or `model_best.pt`.

2026-06-20 why the latest training is still poor:

- The run combined several stabilizing choices that together removed too much control authority.
- `happo_action_clip=0.3` and wheel scale `10.0` cap env wheel targets at about `3 rad/s`; the previously usable checkpoint needed roughly `8.4 rad/s` (`clip=0.6`, scale `14.0`) to move around `0.6 m/s`.
- The new lateral shield then consumes a large fraction of that small action budget:
  - `shield_trigger_rate ~= 0.999`
  - `shield_lateral_rate ~= 0.91`
  - `shield_lateral_critical_rate ~= 0.58`
  - `lateral_turn_clip=0.18`, which is 60% of `happo_action_clip=0.3`.
  - Result: the shield is almost always steering/braking/correcting, leaving little forward drive.
- `attack_level=light` is unsupported and falls back to medium preset behavior for unspecified fields:
  - CSV shows `attack_max_fdi_pos=5.0`.
  - Even with `max_fdi_acc=0.1` and `max_dos_rate=0.02`, observation FDI is not actually light.
- There is still an action semantics mismatch in the HAPPO wrapper:
  - The env executes clipped/postprocessed/shielded/adapted actions.
  - The HAPPO buffer still stores the original policy `pending_actions`.
  - With `clip=0.3`, policy outputs around `leader_action_abs_mean ~= 1.0`, but the env executes no more than `0.3`.
  - This mismatch is much worse than in the old `clip=0.6` run and can make the policy update chase actions the robot never actually executed.
- The latest run is therefore not simply "needs more iterations"; it is under-actuated, over-shielded, not truly light-attack, and training on action data that differs from executed actions.
- Highest-priority fixes before another long run:
  1. Use enough action authority for forward drive: try `happo_action_clip=0.4`, wheel scale `12.0`.
  2. Weaken lateral shield relative to the clip: `lateral_turn_gain=0.35-0.45`, `lateral_turn_clip=0.08-0.10`, `lateral_tol=0.12`.
  3. Disable attack for the first lateral pass, or use a real supported light setting: `attack_level=easy`, `max_fdi_pos=1.0`, `max_fdi_acc=0.05-0.10`, `max_dos_rate=0.01-0.02`.
  4. Fix the wrapper/router so HAPPO stores the executed clipped/postprocessed action, or avoid very small clips until that mismatch is fixed.

2026-06-20 suitable configuration applied:

- Code/config changes made for the next run:
  - Default task speed changed from `TARGET_SPEED_RANGE=(0.8,1.0)` to `(0.6,0.8)`.
  - Default wheel action scale changed from `20.0` to `12.0`.
  - Default HAPPO action clip changed from `1.0` to `0.4`.
  - HAPPO task entry now defaults to `enable_attack=false`, `attack_level=off`, `attack_mode=profile`.
  - Legacy `AttackCfg` now defaults to disabled/light budgets: `max_fdi_pos=1.0`, `max_fdi_acc=0.10`, `max_dos_rate=0.02`.
  - Safety shield lateral correction weakened to avoid taking over the small action budget:
    - `lateral_tol=0.12`
    - `lateral_crit=0.45`
    - `lateral_turn_gain=0.45`
    - `lateral_turn_clip=0.10`
    - `lateral_velocity_gain=0.15`
  - `router.py` now supports a real `attack_level=light` preset:
    - `max_fdi_pos=1.0`
    - `max_fdi_acc=0.10`
    - `max_dos_rate=0.02`
  - Unknown `attack_level` now raises `ValueError` instead of silently falling back to medium.
  - `run_platoon_curriculum.sh` default `NUM_ENVS` changed to `64`, warns above `128`, and final `stage5_cagan` is now light CA-GAN:
    - speed `[0.7,0.85]`
    - `happo_action_clip=0.45`
    - wheel scale `12.0`
    - `attack_level=light`, `max_fdi_pos=1.0`, `max_fdi_acc=0.10`, `max_dos_rate=0.02`
    - default `STAGE5_ITERS=1000`
- Verification:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/tasks/platoon/agents.py source/my_exts/marl_platoon/tasks/__init__.py source/my_exts/marl_platoon/algorithms/router.py` passed.
  - `bash -n scripts/tools/run_platoon_curriculum.sh` passed.
- Next run should not continue the poor `2026-06-19_14-39-00_stage5_cagan_light_from_10950` result as final.
- Start from the older useful checkpoint:
  - `logs/rsl_rl/platoon_happo/2026-06-18_19-12-27_stage5_cagan/model_10950.pt`
- Recommended immediate run:
  - run name: `stage5_lateral_ft_v2`
  - `num_envs=64`
  - `enable_attack=false`
  - `happo_action_clip=0.4`
  - command speed `[0.6,0.8]`
  - wheel scale `12.0`
  - budget `300-500` iterations first, then inspect CSV before longer training.

2026-06-22 current modification inventory:

- `git diff --name-only` currently shows 16 tracked modified files:
  - `scripts/reinforcement_learning/rsl_rl/train.py`
  - `source/my_exts/marl_platoon/algorithms/attacker.py`
  - `source/my_exts/marl_platoon/algorithms/happo/actor.py`
  - `source/my_exts/marl_platoon/algorithms/happo/runner.py`
  - `source/my_exts/marl_platoon/algorithms/pipeline.py`
  - `source/my_exts/marl_platoon/algorithms/router.py`
  - `source/my_exts/marl_platoon/algorithms/shield.py`
  - `source/my_exts/marl_platoon/algorithms/student.py`
  - `source/my_exts/marl_platoon/algorithms/teacher.py`
  - `source/my_exts/marl_platoon/smoke_test_harl_adapter.py`
  - `source/my_exts/marl_platoon/tasks/__init__.py`
  - `source/my_exts/marl_platoon/tasks/platoon/agents.py`
  - `source/my_exts/marl_platoon/tasks/platoon/attacks.py`
  - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - `source/my_exts/marl_platoon/tasks/platoon/teacher.py`
  - `source/my_exts/marl_platoon/wrappers.py`
- Important untracked-but-edited paths:
  - `debug_notes.md`
  - `scripts/tools/run_platoon_curriculum.sh`
  - `scripts/tools/check_platoon_stage.py`
  - `scripts/tools/plot_platoon_metrics.py`
- Current key functional modifications to remember:
  - HAPPO/stage defaults are now lateral-stability-first: speed `[0.6,0.8]`, wheel scale `12.0`, `happo_action_clip=0.4`, default attack off.
  - Real `attack_level=light` now exists in `router.py`; unknown attack levels now raise instead of falling back to medium.
  - Safety shield lateral correction has been weakened: `lateral_tol=0.12`, `lateral_crit=0.45`, `lateral_turn_gain=0.45`, `lateral_turn_clip=0.10`, `lateral_velocity_gain=0.15`.
  - Curriculum stage5 now uses light CA-GAN instead of medium CA-GAN and defaults to `NUM_ENVS=64`, `STAGE5_ITERS=1000`.
  - Metrics/check/plot tooling includes lateral max, per-pair lateral error, heading error, and shield lateral stats.
  - Wrapper/router/HAPPO code includes task-local checkpointing/inference support, URDF wheel-axis adapter handling, HAPPO metrics CSV logging, and CA-GAN/teacher/shield pipeline instrumentation.

2026-06-22 pair_2 targeted lateral fix:

- New diagnosis from latest metrics:
  - `pair_2` (`robot_2 -> robot_3`) is the dominant worst lateral pair:
    - last50 worst count `36/50`
    - last100 worst count `86/100`
    - last200 worst count `141/200`
  - Wheel joint mapping is not the likely cause because all robots share the same wheel joint names and adapter signs.
  - Current issue is local: mean lateral error can look acceptable while `pair_2` remains the main outlier.
- Code changes applied:
  - `reward_lateral_penalty_hard()` now aggregates lateral penalty as:
    - `mean_pair_penalty + 0.5 * max_pair_penalty + 0.8 * pair_2_penalty`
    - Pair penalty still uses the existing `0.05 m` dead-zone and stronger growth after `0.25 m`.
  - Added constants:
    - `LATERAL_MAX_PAIR_WEIGHT=0.5`
    - `LATERAL_PAIR2_EXTRA_WEIGHT=0.8`
    - `LATERAL_PAIR2_INDEX=1`
  - Added pair_2-specific shield settings in `SafetyShieldCfg`:
    - `pair2_lateral_gain_scale=1.25`
    - `pair2_lateral_clip_scale=1.20`
    - `pair2_lateral_clip_max=0.12`
  - `shield.py` now applies the above only when `agent_id == 2`, i.e. `robot_3` relative to `robot_2`; other followers keep the global weaker shield.
  - `router.py` metrics now record signed lateral means:
    - `lateral_pair_1_signed_mean` through `lateral_pair_5_signed_mean`
  - `plot_platoon_metrics.py` now includes a `lateral_pairs_signed` plot group.
- Verification:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/algorithms/shield.py source/my_exts/marl_platoon/algorithms/router.py scripts/tools/plot_platoon_metrics.py` passed.
- Next evaluation:
  - Run the same lateral fine-tune setup from `model_10950.pt`.
  - After 100-200 updates, inspect `lateral_pair_2_abs_mean` and `lateral_pair_2_signed_mean`.
  - If signed mean stays consistently positive/negative, treat it as pair_2 bias.
  - If signed mean is near zero while abs remains high, treat it as oscillation and reduce pair_2 shield clip/gain or add action-rate damping.

2026-06-22 centerline lateral constraint:

- New reasoning:
  - Heading alignment and lateral centering are separate constraints.
  - A follower can have the same heading as its predecessor while staying parallel in the wrong lane.
  - To remove lateral offset, the follower may need a temporary heading difference, so over-weighting `heading_align` can preserve a parallel lateral bias.
- Code changes applied:
  - Added absolute road centerline penalty with target `world_y=0`:
    - `CENTERLINE_Y_TARGET=0.0`
    - `CENTERLINE_DEADZONE=0.05`
    - `CENTERLINE_CRITICAL_OFFSET=0.35`
    - `CENTERLINE_MAX_WEIGHT=0.5`
  - Added `reward_centerline_lateral_penalty()`:
    - penalizes each vehicle's absolute `root_pos_w[:, 1]` offset from the road centerline
    - aggregates as `mean_vehicle_penalty + 0.5 * max_vehicle_penalty`
  - Added reward term:
    - `centerline_lateral = RewardTermCfg(..., weight=-2.0)`
  - Reduced heading alignment weight:
    - `heading_align` changed from `-2.0` to `-0.8`
    - This keeps heading as a stabilizer, while allowing short corrective yaw to return to centerline.
  - `router.py` now records centerline metrics:
    - `centerline_error_abs_mean`
    - `centerline_error_abs_max`
    - `centerline_robot_1_abs_mean` through `centerline_robot_6_abs_mean`
    - `centerline_robot_1_signed_mean` through `centerline_robot_6_signed_mean`
    - `reward_centerline_lateral`
  - `plot_platoon_metrics.py` now plots:
    - centerline aggregate errors in `formation_errors`
    - per-robot abs centerline error in `centerline_robots`
    - per-robot signed centerline error in `centerline_robots_signed`
  - `check_platoon_stage.py` now optionally checks centerline gates for new CSVs:
    - stage1: mean `<0.25`, max `<0.80`
    - stage2: mean `<0.30`, max `<1.00`
    - stage3: mean `<0.35`, max `<1.20`
    - stage4: mean `<0.45`, max `<1.50`
    - stage5: mean `<0.50`, max `<1.75`
- Verification:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/algorithms/router.py scripts/tools/plot_platoon_metrics.py scripts/tools/check_platoon_stage.py` passed.
- Next run diagnostics:
  - Use centerline metrics to distinguish:
    - local pairwise drift: high `lateral_pair_i_abs_mean`
    - global lane drift: high `centerline_robot_i_abs_mean`
    - systematic left/right bias: signed centerline metric consistently positive/negative
  - For the next lateral fine-tune, watch both `lateral_pair_2_abs_mean` and `centerline_robot_3_abs_mean`.

2026-06-22 centerline coordinate-system fix:

- Run `2026-06-22_15-23-03_stage5_centerline_lateral_ft_v1` is invalid and should not be used or played.
- Failure cause:
  - `reward_centerline_lateral_penalty()` and centerline metrics used `root_pos_w[:, 1]` directly.
  - In IsaacLab parallel envs, `root_pos_w` includes each environment's `env_origin`.
  - This made normal lane-centered vehicles appear to have `centerline_error_abs_mean ~= 50` and `centerline_error_abs_max ~= 88`.
  - Signed centerline means stayed near zero only because positive/negative env origins canceled across envs.
  - Result: `Episode_Reward/centerline_lateral` became huge negative and the critic exploded.
- Code fix:
  - `config.py` now uses `_env_local_y(env, asset_name)`:
    - `local_y = root_pos_w[:, 1] - env.scene.env_origins[:, 1]` when `env_origins` exists.
    - Falls back to world y for single-env/non-origin contexts.
  - `router.py` now uses the same env-local y logic for all centerline metrics.
  - Pairwise relative metrics/rewards still use world-position differences, which are correct because same-env origins cancel in pair differences.
  - `LATERAL_PAIR2_EXTRA_WEIGHT` is currently `0.0`; this was left unchanged.
- Verification:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py source/my_exts/marl_platoon/algorithms/router.py scripts/tools/plot_platoon_metrics.py scripts/tools/check_platoon_stage.py` passed.
  - `num_envs=1` resume probe loaded `2026-06-22_02-11-55_stage5_lateral_ft_v2/model_final.pt` successfully.
  - Full 30-iter sanity run completed:
    - run: `logs/rsl_rl/platoon_happo/2026-06-22_17-48-07_stage5_centerline_sanity_localy`
    - source checkpoint: `2026-06-22_02-11-55_stage5_lateral_ft_v2/model_final.pt`
    - `num_envs=64`, attack off, speed `[0.6,0.8]`, clip `0.4`, wheel scale `12.0`
- Sanity result:
  - Centerline coordinate bug is fixed:
    - last30 `centerline_error_abs_mean ~= 0.091`
    - last30 `centerline_error_abs_max ~= 0.657`
    - last10 `centerline_error_abs_mean ~= 0.186`
    - last10 `centerline_error_abs_max ~= 1.276`
    - `Episode_Reward/centerline_lateral` stayed around `0` to `-0.95`, not `-10000+`.
  - Stability did not collapse:
    - last30 `termination_reset_on_bad_ori = 0`
    - last30 `value_loss ~= 8.13`
    - last30 `critic_grad_norm ~= 12.87`
    - final log around update 30: `value_loss ~= 12.25`, `critic_grad_norm ~= 15.78`
  - This sanity run is not a final performance model:
    - last30 `leader_speed_mean ~= 0.123`
    - last30 `speed_error_abs_mean ~= 0.619`
    - last30 `shield_trigger_rate ~= 0.917`
    - `check_platoon_stage.py --stage stage1_stable --window 30` fails on speed and shield trigger.
- Next step:
  - Do not use `2026-06-22_15-23-03_stage5_centerline_lateral_ft_v1/model_final.pt`.
  - The coordinate fix is good; rerun a longer lateral/centerline fine-tune from `2026-06-22_02-11-55_stage5_lateral_ft_v2/model_final.pt`, but treat `2026-06-22_17-48-07_stage5_centerline_sanity_localy` as sanity only.
  - Watch `centerline_robot_2_signed_mean`, `centerline_robot_3_signed_mean`, `lateral_pair_2_abs_mean`, and `lateral_pair_4_abs_mean` in the next run.

2026-06-22 5-car platoon conversion, delete old white car:

- User request:
  - Original 6-car visual order: blue, white, gray, red, teal, purple.
  - New 5-car order must delete the old white car only:
    - new robot_1 = old robot_1, blue
    - new robot_2 = old robot_3, gray
    - new robot_3 = old robot_4, red
    - new robot_4 = old robot_5, teal
    - new robot_5 = old robot_6, purple
- Code changes made:
  - `PLATOON_ROBOTS` is now `["robot", "robot_2", "robot_3", "robot_4", "robot_5"]`.
  - `PlatoonAlgorithmCfg.num_agents` defaults are now `5` in both task config paths.
  - Scene config removed old white `robot_2`; old gray/red/teal/purple materials were shifted into new `robot_2` through `robot_5`.
  - Action config now exposes only `joint_vel_1` through `joint_vel_5`.
  - HAPPO wrapper default, router default, teacher robot list, shield robot list, smoke-test default, plotting columns, and curriculum action-scale overrides were synchronized to 5 cars.
  - Metrics now log 4 pairwise lateral columns and 5 centerline robot columns.
- Important compatibility note:
  - This is a real 5-agent task change, not just hiding a visual car.
  - Observation/action dimensions and HAPPO `num_agents` changed from 6 to 5.
  - Existing 6-car HAPPO checkpoints cannot be resumed directly in this 5-car task; start a new 5-car run, or write explicit checkpoint migration code if needed later.
- Verification:
  - `python3 -m py_compile` passed for the changed Python modules.
  - `bash -n scripts/tools/run_platoon_curriculum.sh` passed.
  - Static search found no remaining runtime `robot_6`, `joint_vel_6`, `lateral_pair_5`, or `centerline_robot_6` references under `source/my_exts`, `scripts/tools`, and the RSL-RL entrypoint path.
  - Direct config import with plain Python failed due missing Isaac USD runtime libraries (`pxr`/`libusd_tf.so`), so no Isaac app startup smoke test was completed in this turn.
- Next step:
  - Run a fresh 5-car HAPPO training job without `--resume` from old 6-car checkpoints.
  - Start conservatively with `--num_envs 64`, attack off, speed `[0.6,0.8]`, `happo_action_clip=0.4`, and wheel scales `12.0`.

2026-06-22 analysis of recurring second-car lateral drift:

- Full search did not find a unique physical/action mapping for `robot_2`:
  - all platoon vehicles use the same URDF/articulation config;
  - action terms are symmetric (`joint_vel_i` with the same wheel regex/scale);
  - wheel-axis adapter signs were previously verified as identical across cars.
- Deleting the old white car did not remove the pattern:
  - latest 5-car run `2026-06-22_19-19-53_platoon5_drop2_stage1_noattack` still shows the new `robot_2` as the largest centerline offender.
  - last100 metrics:
    - `centerline_robot_1_abs_mean ~= 0.032`
    - `centerline_robot_2_abs_mean ~= 0.317`  <-- largest
    - `centerline_robot_3_abs_mean ~= 0.127`
    - `centerline_robot_4_abs_mean ~= 0.077`
    - `centerline_robot_5_abs_mean ~= 0.063`
    - `lateral_pair_1_abs_mean ~= 0.319`, `lateral_pair_2_abs_mean ~= 0.332`, with worst pair mostly pair_1/pair_2.
  - Therefore the old white asset is unlikely to be the root cause; the vulnerable slot is the first follower / second vehicle position.
- Checkpoint actor inspection gives direct evidence:
  - For `2026-06-22_19-19-53_platoon5_drop2_stage1_noattack/model_final.pt`, evaluating each HAPPO actor at an ideal straight-following observation gave:
    - agent 1 turn proxy ~= `-0.005`
    - agent 2 turn proxy ~= `+1.165`  <-- very large fixed steering bias
    - agent 3 turn proxy ~= `-0.322`
    - agent 4 turn proxy ~= `-0.151`
    - agent 5 turn proxy ~= `+0.130`
  - This means the second vehicle's own actor has learned a deterministic left/right wheel imbalance even when the input corresponds to straight tracking.
- Most likely causes:
  1. HAPPO uses separate actors per agent and fixed agent update order. The second vehicle is not sharing parameters with other followers, so it can learn a unique biased policy.
  2. `to_agent_rewards()` repeats the same scalar IsaacLab env reward to every agent. There is no local per-agent reward/advantage that strongly says "robot_2 itself is off center"; this weakens credit assignment.
  3. The first follower has a different control problem from later followers: it follows the speed-commanded leader directly, while later cars follow already filtered follower motion.
  4. The policy observation is local predecessor-relative plus own velocity/yaw-rate and `role_id`; it does not include env-local centerline y directly, so centerline reward is only weakly observable by the actor.
  5. Execution action differs from raw policy action: wrapper clips actions, applies shield, clips again, then applies the wheel-axis adapter before physics. The HAPPO buffer still stores the raw pending actions. This can let a bad actor rely on clipping/shielding instead of learning a centered action, especially when `shield_trigger_rate ~= 1` and `shield_lateral_turn_mean ~= 0.089` near the `0.10` clip.
- Practical implication:
  - Deleting the old white car is a workaround for the visual symptom but does not fix the structural cause.
  - Real fixes should target the first-follower policy/training path:
    - add per-agent or per-slot local lateral/centerline reward;
    - add env-local centerline y and heading/yaw error to each agent observation;
    - log per-agent action left/right turn bias;
    - consider shared follower actor parameters or randomized HAPPO agent update order;
    - reduce shield dependence or align the stored/trained action with the executed clipped/shielded action through a bounded/squashed policy design.

2026-06-22 recommended fix plan for second-car drift:

- Highest priority change:
  - Add directly observable centerline/yaw information to each agent observation.
  - Current policy sees predecessor-relative geometry and own velocity/yaw-rate, but not env-local centerline y. A centerline reward is hard to learn when the actor cannot directly observe its own lane offset.
  - Recommended per-agent obs extension:
    - `centerline_y = root_pos_w[:, 1] - env_origin[:, 1]`
    - `heading_y` or signed yaw/lateral heading component
    - optionally predecessor-relative heading error
- Second priority:
  - Replace fully global scalar reward replication with a mixed global + local per-agent reward for HAPPO.
  - Keep the existing env reward as the global component, but add local penalties before inserting into the HAPPO buffer:
    - own centerline penalty per robot;
    - predecessor-pair lateral penalty for each follower;
    - small left/right wheel turn-bias/action-difference penalty per agent.
  - This gives the second actor direct negative credit when `robot_2` drifts, instead of relying only on platoon mean reward.
- Third priority:
  - Add per-agent action diagnostics:
    - `action_turn_agent_i = mean(right_wheels) - mean(left_wheels)` from raw and executed actions.
    - This would catch the current failure directly: actor 2 has a large turn proxy even on ideal straight-following input.
- Fourth priority:
  - Reduce shield dependence during training:
    - keep shield as a safety limiter, but log when it is near saturation;
    - avoid letting the policy learn "bad raw action + shield correction" as the normal behavior.
  - Long-term cleaner fix is to store/train against the same bounded action convention that is executed, or use a squashed actor whose sampled action already respects the env action bounds.
- Fifth priority:
  - Consider sharing follower actor parameters or initializing all follower actors from the same weights.
  - Separate actors allow one slot to drift into a unique bad solution; shared follower parameters would make the first follower less likely to develop a one-off steering bias.
- Recommended next code change order:
  1. Add centerline/yaw observation terms and retrain 5-car from scratch.
  2. Add mixed local HAPPO reward shaping in `router.to_agent_rewards()` or immediately before `HAPPOStudentModule.observe()`.
  3. Add action-turn metrics to CSV.
  4. Only after that, revisit shared follower actor or executed-action storage, because those are more invasive.

2026-06-22 consistency check against `/home/cnc/Desktop/bare_jrnl.pdf`:

- The paper's core structure:
  - lower-level Student agents use HATRL/HAPPO-style sequential policy updates from local corrupted observations;
  - upper-level Reward Teacher uses uncompromised physical states and executed control inputs during centralized training;
  - Teacher provides shaped intrinsic reward `R_phi(y_i, u_i) = r_env + F_phi(y_i, u_i)`;
  - physical objective includes per-vehicle tracking errors, velocity/acceleration consistency, jerk, overspeed, and collision cost;
  - safety projection is a training-time action projection; it stabilizes exploration but is not the main source of performance.
- Consistent proposed fixes:
  - Adding centerline/yaw terms to Student observations is consistent only if those terms are treated as locally available/corruptible observations, not privileged clean Teacher-only state.
  - Adding per-agent local reward components is consistent if implemented as the Student's shaped intrinsic reward or Teacher shaping signal, because the paper's lower objective averages `R_phi(y_i, u_i)` across agents.
  - Adding lateral/centerline physical terms is a reasonable 2D IsaacLab extension of the paper's 1D longitudinal spacing/velocity/acceleration physical objective, but should be described as a high-fidelity implementation extension.
  - Logging per-agent action turn bias is purely diagnostic and does not alter the algorithm.
- Potentially inconsistent or more invasive changes:
  - Shared follower actor parameters would deviate from the paper's heterogeneous-agent policy parameterization `{theta_i}`; keep this as an ablation, not the main MGRS reproduction.
  - Randomized HAPPO update order is a HAPPO variant; the paper's Algorithm 1 states sequential `i = 1,...,m`, so fixed order is closer to the text.
  - Do not simply replace HAPPO buffer raw actions with shielded/executed actions. The policy sampled raw `u`, while the shield produces executed `u_hat`; training HAPPO on `u_hat` without the correct log probability would break on-policy consistency. Use `u_hat` for Teacher physical-cost evaluation and diagnostics, or move to a properly modeled squashed/bounded policy if changing the action distribution.
- Implementation boundary:
  - Best next implementation that remains paper-consistent:
    1. extend local observation with lane/heading signals that are allowed to be part of `y_i`;
    2. add per-agent shaped reward terms as `F_phi`/local intrinsic shaping, not as clean privileged state leakage;
    3. keep raw policy actions in HAPPO storage;
    4. use executed actions after attack/shield/adapter only for physical cost/Teacher metrics.

2026-06-22 user decision: use 5-car platoon as the target setup:

- User will revise the PDF from the previous 6-total-car simulation setup to 5 total cars.
- Therefore, the current 5-car implementation is the intended target rather than a temporary ablation.
- The remaining issues are not the number of cars, but:
  1. second-slot/follower policy still tends to learn a turn bias;
  2. Student observation still lacks directly observable lane/centerline and heading-error features;
  3. HAPPO still receives mostly replicated global reward instead of a clear per-agent shaped reward;
  4. shield usage remains high and can hide poor raw actions;
  5. metrics do not yet log per-agent raw/executed action turn bias;
  6. all old 6-car checkpoints remain incompatible with the 5-car task.
- Next implementation should target the 5-car task directly:
  - add local centerline/yaw observation features;
  - add 5-car per-agent reward shaping;
  - add per-agent action-turn metrics;
  - train from scratch under the 5-car config.

2026-06-22 implemented 5-car second-slot drift fixes:

- Implemented the three requested changes.
- Observation extension:
  - `obs_platoon_chain()` now outputs 11 dims per car instead of 8:
    - old: `rel_x, rel_y, rel_vx, rel_vy, own_vx, own_vy, own_yaw_rate, role_id`
    - new: old fields plus `centerline_y`, `heading_y`, `pair_heading_sin`
  - Full 5-car policy obs is now `55` dims.
  - This makes lane/heading correction directly observable to the Student policy.
- Per-agent local reward shaping:
  - HAPPO now keeps the original IsaacLab env reward for logs, then adds internal per-agent shaping before storing rewards in the HAPPO buffer.
  - New config fields:
    - `local_reward_shaping=True`
    - `local_reward_centerline_coef=0.4`
    - `local_reward_pair_lateral_coef=0.4`
    - `local_reward_heading_coef=0.2`
    - `local_reward_turn_coef=0.02`
  - The shaping penalizes each agent's own centerline error, follower pair lateral error, heading error, and raw left/right turn bias.
  - This is kept inside the MGRS/HAPPO reward path and does not alter the frozen outer PPO reward.
- Action-turn diagnostics:
  - Wrapper now records:
    - raw actor output before clip/shield;
    - executed semantic action after clip/attack/shield and before URDF wheel-axis adapter.
  - CSV now includes:
    - `raw_action_turn_abs_mean`
    - `executed_action_turn_abs_mean`
    - `raw_action_turn_agent_1` through `raw_action_turn_agent_5`
    - `executed_action_turn_agent_1` through `executed_action_turn_agent_5`
    - `local_reward_shaping_mean`
    - local reward component penalty means.
  - `plot_platoon_metrics.py` now has `action_turn_bias` and `local_reward_shaping` plot groups.
- Verification:
  - `python3 -m py_compile` passed for changed Python modules.
  - Static search found no reintroduced `robot_6`, `joint_vel_6`, `lateral_pair_5`, or `centerline_robot_6` runtime fields.
  - IsaacLab smoke test passed:
    - run: `logs/rsl_rl/platoon_happo/2026-06-22_22-04-56_platoon5_obs11_localreward_smoke`
    - command used `--num_envs 1 --max_iterations 1`
    - Action Manager shape: `20` = 5 cars x 4 wheels
    - Observation Manager policy shape: `(55,)` = 5 cars x 11 obs dims
    - HAPPO runner built with `obs_dim=11`, `share_obs_dim=55`
    - CSV wrote the new turn-bias and local-reward fields successfully.
- Compatibility note:
  - Because obs dim changed from 8 to 11 per agent, all older 5-car and 6-car HAPPO checkpoints are incompatible with this implementation.
  - Next real training must start fresh under the new 5-car obs11 config.

2026-06-22 pre-training code compatibility check:

- Compared the current 5-car obs11/local-reward implementation against the previously trainable task-side HAPPO flow.
- Hard training path still lines up:
  - `PLATOON_ROBOTS = ["robot", "robot_2", "robot_3", "robot_4", "robot_5"]`.
  - Action manager has only `joint_vel_1` through `joint_vel_5`, total action dim `20`.
  - Observation manager outputs policy obs dim `55`, which reshapes cleanly to `5 x 11`.
  - HAPPO task registration sets `algorithm.num_agents = 5`, `use_happo_actions = True`, and `freeze_outer_ppo = True`.
  - Wrapper reshapes obs to `[num_envs, 5, 11]`, router builds HAPPO with `obs_dim=11`, `act_dim=4`, `share_obs_dim=55`.
  - Local reward shaping returns `[num_envs, 5, 1]`, matching HAPPO buffer reward shape.
  - Raw/executed action-turn metrics are recorded before the URDF wheel-axis adapter, so diagnostics are in the semantic action convention.
- Existing smoke run confirms startup and one update completed:
  - run: `logs/rsl_rl/platoon_happo/2026-06-22_22-04-56_platoon5_obs11_localreward_smoke`
  - produced checkpoints, TensorBoard event, and `platoon_metrics.csv`.
  - first CSV row has `num_agents=5`, centerline errors near zero, no `centerline_abs≈50` coordinate bug, and local reward/turn-bias metrics populated.
- Main caveat:
  - This implementation cannot resume old checkpoints because per-agent obs changed from 8 to 11.
  - Long training should be launched from scratch; old `--resume --checkpoint ...` is expected to fail or produce invalid behavior.
- Risk level before full training:
  - Code-level startup risk is low because smoke training already passed.
  - Learning-quality risk remains moderate: convergence still needs a 30-iteration sanity run before long training, but there is no obvious code mismatch that should prevent training from running.

2026-06-22 training plan for obs11/local-reward 5-car version:

- Modification rationale:
  - The old second-slot drift problem was not treated as a wheel mapping issue after joint order/sign checks passed.
  - The fix gives every car direct local state needed to correct drift:
    - signed env-local centerline offset;
    - heading lateral component;
    - predecessor-relative heading error.
  - The fix also changes HAPPO credit assignment:
    - keep global IsaacLab reward for logging;
    - add per-agent local shaping before storing HAPPO rewards;
    - penalize own centerline drift, follower pair lateral drift, heading error, and raw turn bias.
  - The new action-turn metrics make it possible to see whether a specific agent is still outputting asymmetric wheel commands before shield correction.
- Training rule:
  - Do not resume old checkpoints because obs changed from 8 to 11 per agent.
  - Start with a 30-iteration sanity run.
  - If `centerline_error_abs_mean`, `lateral_error_abs_mean`, `value_loss`, `critic_grad_norm`, and `reset_on_bad_ori` are normal, then run the longer fresh training.
- Recommended launch profile:
  - `--num_envs 64` first for RTX 3080 10GB safety.
  - no attack for this phase.
  - speed range `[0.6,0.8]`, action scale `12.0`, HAPPO action clip `0.4`.

2026-06-23 first-follower drift follow-up:

- User observed that the second visible car is still white/gray and still drifts after the 5-car run.
- Code check:
  - Old 6-car/old-white-car slot is not still present in the current task.
  - Current robots are `robot`, `robot_2`, `robot_3`, `robot_4`, `robot_5`.
  - Initial x positions are compressed and continuous: `1.0, -0.5, -2.0, -3.5, -5.0`, so there is no missing old-slot gap.
  - The visible pale second car is the current `robot_2`; its material was too bright/metallic and could look white.
- Latest run inspected:
  - run: `logs/rsl_rl/platoon_happo/2026-06-22_22-57-11_platoon5_obs11_localreward_stage1_fresh`
  - 1500 rows present.
  - Last 100 updates:
    - `centerline_robot_2_abs_mean ≈ 0.3335`, highest among robots.
    - `centerline_robot_2_signed_mean ≈ -0.3150`, long-term one-sided bias.
    - `lateral_pair_1_abs_mean ≈ 0.3047`.
    - `lateral_pair_2_abs_mean ≈ 0.4146`, worst pair remains pair 2.
    - `raw_action_turn_agent_2 ≈ +0.9071`, meaning actor 2 learned a strong raw turn bias.
    - `executed_action_turn_agent_2 ≈ -0.0712`, meaning shield is fighting the actor, but the correction is still insufficient.
    - `shield_lateral_turn_mean ≈ 0.0921` with clip `0.10`, so the existing shield is already near saturation.
  - Best early rolling window was near update 50; `model_50.pt` had much smaller `robot_2` centerline and raw turn bias than the final model.
- Conclusion:
  - The issue is not the deleted old white asset and not uncompressed initial spacing.
  - The issue is the first-follower slot (`robot_2`) learning a persistent turn bias while the shield nearly saturates to counteract it.
- Code changes made:
  - Darkened `robot_2` material to graphite gray so it no longer looks like the old white car.
  - Added first-follower-specific shield parameters:
    - `first_follower_lateral_gain_scale=1.25`
    - `first_follower_lateral_clip_scale=1.6`
    - `first_follower_lateral_clip_max=0.16`
    - `first_follower_centerline_gain=0.35`
    - `first_follower_centerline_clip=0.06`
  - Shield now adds a small env-local centerline correction only for `agent_id == 1` / `robot_2`.
  - Added first-follower local reward scales:
    - `local_reward_first_follower_centerline_scale=2.0`
    - `local_reward_first_follower_pair_lateral_scale=1.5`
    - `local_reward_first_follower_turn_scale=2.0`
  - Added CSV metrics:
    - `shield_first_follower_centerline_rate`
    - `shield_first_follower_centerline_turn_mean`
- Verification:
  - `python3 -m py_compile` passed for changed modules.
  - IsaacLab smoke run passed:
    - `logs/rsl_rl/platoon_happo/2026-06-23_12-14-55_platoon5_first_follower_centerline_shield_smoke`
    - action shape remains `20`.
    - policy obs shape remains `55`.
    - HAPPO still builds with `obs_dim=11`, `act_dim=4`, `share_obs_dim=55`.
    - New shield CSV columns are present.
- Training implication:
  - Obs/action dimensions did not change, so current obs11 checkpoints remain load-compatible.
  - Recommended next run is a fine-tune from `model_50.pt` of `2026-06-22_22-57-11_platoon5_obs11_localreward_stage1_fresh`, not from `model_final.pt`, because final has a much stronger learned `robot_2` raw turn bias.

2026-06-23 weak all-agent centerline + HAPPO resume evaluation:

- Code changes made in this round:
  - Replaced the previous strong first-follower-only centerline shield with weak all-agent centerline correction plus mild first-follower emphasis.
    - all agents: `centerline_turn_gain=0.10`, `centerline_turn_clip=0.02`
    - first follower / `robot_2`: `first_follower_centerline_gain=0.16`, `first_follower_centerline_clip=0.03`
    - pair-lateral shield kept weaker: `lateral_turn_gain=0.35`, `lateral_turn_clip=0.08`
  - Reduced first-follower local reward scales to avoid over-pinning `robot_2`:
    - centerline `1.3`, pair lateral `1.2`, turn `1.3`
  - Added all-agent shield centerline CSV metrics:
    - `shield_centerline_rate`
    - `shield_centerline_turn_mean`
  - Fixed a critical resume bug in `train.py`:
    - RSL-RL checkpoint resume was loading the outer runner only.
    - The task-local `platoon_happo_state` was saved in checkpoints but not restored.
    - `train.py` now loads task-local HAPPO state after `runner.load(...)`.
  - Fixed local reward / metrics turn convention in `router.py`:
    - turn penalty now computes physical left-right difference after accounting for the URDF right-wheel sign adapter.
    - This avoids treating the normal pre-adapter forward pattern as a turn.
- Verification:
  - `python3 -m py_compile` passed for the changed Python modules.
  - Effective resume logs now show:
    - `Loaded task-local HAPPO state from: .../model_500.pt`
    - HAPPO update count resumes around `502`, confirming state restoration.
- Invalid comparison runs:
  - Runs before the `train.py` HAPPO-state load fix are not valid for checkpoint quality, because task-local HAPPO was reinitialized.
- Valid run 1:
  - `logs/rsl_rl/platoon_happo/2026-06-23_17-15-20_platoon5_weak_centerline_from500_happoload_eval50`
  - Source checkpoint: `2026-06-22_22-57-11_platoon5_obs11_localreward_stage1_fresh/model_500.pt`
  - Attack off: `attack_enabled=0`, `attack_level=off`, `max_fdi_acc=0`, `max_dos_rate=0`.
  - Last 20 updates:
    - command speed mean `0.705`
    - leader speed mean `0.312`
    - platoon speed mean `0.268`
    - speed error abs mean `0.436`
    - centerline mean `0.258`
    - `robot_3` centerline abs mean `0.811`
    - lateral mean `0.460`
    - `pair_2` lateral abs mean `1.039`
    - worst pair: `pair_2` in all last 20 rows
    - `reset_on_bad_ori=0`, timeout `1.0`
    - value loss about `197`, critic grad about `282`
  - Conclusion: valid resume shows `robot_2` is no longer the main drift point, but the error moved downstream to `robot_3` / `pair_2`; speed is not normal.
- Valid run 2 after physical turn penalty fix:
  - `logs/rsl_rl/platoon_happo/2026-06-23_17-22-02_platoon5_turnfix_weak_centerline_eval50`
  - Same source checkpoint and attack-off settings.
  - Last 20 updates:
    - command speed mean `0.705`
    - leader speed mean `0.312`
    - platoon speed mean `0.268`
    - speed error abs mean `0.437`
    - centerline mean `0.262`
    - `robot_3` centerline abs mean `0.823`
    - lateral mean `0.472`
    - `pair_2` lateral abs mean `1.057`
    - worst pair: `pair_2` in all last 20 rows
    - `reset_on_bad_ori=0`, timeout `1.0`
    - value loss about `242`, critic grad about `309`
  - Conclusion: turn convention fix is correct for future training and diagnostics, but it did not rescue this short fine-tune from `model_500.pt` in 50 iterations.
- Current assessment:
  - This weak-all-agent + mild-first-follower configuration is not ready to keep as a final training recipe.
  - It prevents bad orientation resets, but it still runs too slowly and has a strong downstream lateral drift at `robot_3` / `pair_2`.
  - The current `model_final.pt` from the two evaluation runs should not be used as the main policy.
- Next Steps:
  - Do not continue from the `platoon5_weak_centerline_from500_happoload_eval50` or `platoon5_turnfix_weak_centerline_eval50` final checkpoints.
  - Keep the `train.py` HAPPO-state resume fix and the physical-turn penalty fix.
  - Rework the shield/reward balance before the next run:
    - avoid individual per-agent centerline correction fighting pairwise formation;
    - either use a weaker/common platoon-level centerline correction, or reduce centerline shield and rely more on reward;
    - separately improve leader speed tracking, because current leader speed is about `0.31 m/s` for a `0.70 m/s` command.
  - Any next sanity run should again be attack-off first; CAGAN should remain disabled until speed and pair_2 lateral are stable.

2026-06-23 explanation for drift with attack disabled:

- `attack_enabled=0` only removes adversarial FDI/DoS perturbations; it does not guarantee the learned policy, reward shaping, shield projection, or low-level wheel semantics are centered.
- The latest drift is not caused by attack. It is endogenous:
  - the resumed checkpoint was trained before the physical-turn reward convention fix, so its actor/critic already carry a biased control habit;
  - the first-follower correction pulled `robot_2` closer to center, but pushed the main error downstream into `robot_3 / pair_2`;
  - centerline shield and pairwise formation shield can fight each other when applied per-agent, creating a persistent pair_2 offset;
  - speed tracking is still weak, with leader speed about `0.31 m/s` for a `0.70 m/s` command, so gap/catch-up/lateral corrections stay active and amplify instability.
- Training time is a factor but not the root cause:
  - 50 update fine-tune is too short to erase a biased checkpoint after reward semantics change.
  - However, simply training this same configuration longer is risky because value loss and critic grad rose while pair_2 stayed worst.
- Current best interpretation:
  - Need a clean no-attack stabilization run after the turn-convention fix and after reducing conflicting deterministic centerline corrections.
  - Do not add CAGAN until no-attack speed and pair_2 lateral are stable.

2026-06-23 clean no-attack stabilization result:

- Clean no-attack training completed:
  - Run: `logs/rsl_rl/platoon_happo/2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200`
  - Source checkpoint: `2026-06-23_21-39-58_platoon5_clean_noattack_v1_fresh250/model_150.pt`
  - Training length: 200 additional updates, ending at iteration/update about `350`.
  - Attack was disabled throughout:
    - `env.algorithm.enable_attack=false`
    - `attack_level=off`
    - eval `attack_enabled=0`
  - Training stayed numerically stable:
    - `reset_on_bad_ori=0`
    - normal timeout episodes
    - critic/value did not explode
    - final checkpoint saved with task-local HAPPO state.
- Deterministic no-attack 1000-step evaluation:
  - Output: `.../2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200/eval_noattack_030_045_1000/eval_summary.csv`
  - Speed command range: `[0.30, 0.45] m/s`
  - `model_300.pt`:
    - command speed `0.378`, leader speed `0.352`, platoon speed `0.304`
    - speed error `0.079`
    - centerline mean `0.086`
    - lateral mean `0.136`
    - pair lateral means `[0.135, 0.221, 0.076, 0.112]`
    - bad-orientation reset `0`
  - `model_best.pt`:
    - command speed `0.363`, leader speed `0.352`, platoon speed `0.304`
    - speed error `0.065`
    - centerline mean `0.085`
    - lateral mean `0.140`
    - pair lateral means `[0.139, 0.231, 0.074, 0.113]`
    - bad-orientation reset `0`
  - `model_final.pt`:
    - command speed `0.374`, leader speed `0.352`, platoon speed `0.304`
    - speed error `0.075`
    - centerline mean `0.082`
    - lateral mean `0.133`
    - pair lateral means `[0.139, 0.203, 0.070, 0.118]`
    - bad-orientation reset `0`
- Interpretation:
  - `model_final.pt` is the best overall checkpoint from this clean no-attack run.
  - The system now completes the no-attack 1000-step evaluation without abnormal reset, with normal leader speed and acceptable mean centerline/lateral error.
  - Remaining issue is mild late-episode downstream drift: in the last 100 eval steps, lateral mean is around `0.16`, and the last pair can reach about `0.29-0.30`. This is visible as mild tail deviation, not the previous large side-lane failure.
  - `model_best.pt` is the speed-biased alternative, but its pair_2 mean is worse than `model_final.pt`.
- Extra shield test:
  - Tested stronger shield override on `model_final.pt`:
    - `lateral_tol=0.030`
    - `lateral_turn_gain=0.35`
    - `lateral_turn_clip=0.10`
    - `centerline_turn_gain=0.30`
    - `centerline_turn_clip=0.10`
    - `lateral_velocity_gain=0.12`
  - Summary looked slightly better on average for pair_2, but the late tail became worse:
    - default tail 950-999: lateral about `0.168`, pair_4 about `0.304`
    - shield_plus tail 950-999: lateral about `0.272`, pair_4 about `0.368`
  - Conclusion: do not adopt shield_plus. Keep the current default shield values.
- Current recommendation:
  - Use `model_final.pt` from `2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200` as the current no-attack baseline.
  - Keep attack disabled for visual play and any further stability checks.
  - Do not move to CAGAN until visual play confirms the mild tail drift is acceptable.

2026-06-23 GitHub backup decision:

- User visually played `model_final.pt` from the clean no-attack run and reported that the effect is good.
- Backup target:
  - remote: `git@github.com:hs997/densact.git`
  - current branch before backup: `freeze/cagan-step3-dualchannel-logging`
- Backup scope:
  - include HAPPO/platoon source changes, `train.py`, `play.py`, `eval_happo_platoon.py`, `debug_notes.md`, small debug patch records, and `.gitignore`.
  - exclude `.venv/`, `logs/`, `*.pt`, `*.pth`, and the large `Melodic_wheeltec_robot_src_250707/` resource tree from the code backup.
- Reason:
  - GitHub should receive the reproducible code/config/debug notes, not local virtual environments, training logs, model checkpoints, or GB-scale robot asset archives.
- Push result:
  - Commit created: `8a9d434 Backup platoon HAPPO no-attack baseline`
  - Pushed successfully to `origin/freeze/cagan-step3-dualchannel-logging`.
  - Large local files remain intentionally untracked and local-only.
- Restore/download commands for this backup:
  - New clone:
    - `git clone -b freeze/cagan-step3-dualchannel-logging git@github.com:hs997/densact.git`
  - Existing repo:
    - `git fetch origin`
    - `git switch freeze/cagan-step3-dualchannel-logging`
    - `git pull --ff-only origin freeze/cagan-step3-dualchannel-logging`
  - Exact backup commit if needed:
    - `git switch -c restore-platoon-noattack-baseline 4dfe279`
  - Note: model checkpoints under `logs/` and `*.pt` were intentionally not pushed to GitHub.

## Restore Command For Future Chats

Tell Codex:

```text
请先读取 /home/cnc/SSD_1T/xzw/IsaacLab-main/debug_notes.md，然后继续执行里面的 Next Steps，优先排查 HAPPO leader 不动的问题。每次回复结束前把新的关键结论更新回这个 md 文件。
```

2026-06-24 speed-up test after clean no-attack baseline:

- Baseline checkpoint kept as source:
  - `logs/rsl_rl/platoon_happo/2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200/model_final.pt`
  - This is still the checkpoint the user visually played and reported as good.
- A 120-update speed fine-tune was tested:
  - Run: `logs/rsl_rl/platoon_happo/2026-06-24_11-45-47_platoon5_speed_ft_scale125_noattack`
  - Training config: attack off, command `[0.30,0.45]`, wheel scale `12.5`, mild stronger local reward.
  - Deterministic 1000-step eval showed all fine-tuned checkpoints improved speed but degraded centerline/lateral compared with the original baseline:
    - `model_400.pt`: leader `0.365`, platoon `0.317`, speed_err `0.069`, center `0.089`, lateral `0.141`
    - `model_450.pt`: leader `0.365`, platoon `0.317`, speed_err `0.057`, center `0.102`, lateral `0.134`
    - `model_final.pt`: leader `0.365`, platoon `0.318`, speed_err `0.068`, center `0.102`, lateral `0.135`
  - Decision: do not adopt the fine-tuned speed run checkpoints.
- Fair single-checkpoint baseline eval:
  - Output: `eval_baseline_scale120_single_cmd030_045_1000/eval_summary.csv`
  - Baseline scale `12.0`: leader `0.352`, platoon `0.304`, speed_err `0.078`, center `0.082`, lateral `0.133`, gap `0.290`, bad reset `0`, attack `0`.
- Best tested speed-up without changing checkpoint:
  - Use original `model_final.pt`, keep attack off, set wheel scale to `12.5`.
  - Add mild shield overrides:
    - `env.safety_shield.lateral_turn_gain=0.32`
    - `env.safety_shield.centerline_turn_gain=0.28`
  - Keep shield clips at defaults (`0.08`); do not use the earlier strong shield.
  - Eval output: `eval_speedup_scale125_shield_mild_cmd030_045_1000/eval_summary.csv`
  - Result:
    - leader `0.365` vs baseline `0.352`
    - platoon `0.316` vs baseline `0.304`
    - speed_err `0.070` vs baseline `0.078`
    - center `0.082` vs baseline `0.082`
    - center max mean `0.174` vs baseline `0.180`
    - lateral `0.132` vs baseline `0.133`
    - lateral max mean `0.254` vs baseline `0.266`
    - pair_1 `0.130` vs baseline `0.139`
    - pair_2 `0.191` vs baseline `0.203`
    - bad reset `0`, attack `0`
  - Tradeoff:
    - gap worsens from `0.290` to `0.302`
    - pair_3 worsens from `0.070` to `0.086`
    - pair_4 worsens from `0.118` to `0.122`
  - Interpretation: this is the best current aggregate speed-up config, but it is not a strict every-metric improvement.
- Additional catch-up tests:
  - `scale=12.5`, mild shield, `catchup_action=-0.38`:
    - leader `0.365`, platoon `0.319`, speed_err `0.067`, center `0.083`, lateral `0.127`, gap `0.284`
    - improves gap/lateral/speed more than mild-only, but center max peak worsens (`0.269` vs baseline `0.252`) and pair_3/pair_4 still worsen.
  - `scale=12.5`, mild shield, `catchup_action=-0.40`:
    - leader `0.365`, platoon `0.320`, speed_err `0.067`, center `0.086`, lateral `0.127`, gap `0.275`
    - too much centerline degradation; do not recommend as default.
- Current recommendation:
  - For visual play and speed-up testing, prefer the original baseline checkpoint with `scale=12.5` plus mild shield overrides.
  - If strict gap is more important than centerline max, test `catchup_action=-0.38` visually as an alternate, but do not make it the default yet.
  - Do not use `2026-06-24_11-45-47_platoon5_speed_ft_scale125_noattack/model_final.pt` as the main policy.

2026-06-24 why higher speed worsens some metrics:

- The current no-attack policy often saturates wheel actions near `happo_action_clip=0.4`; increasing wheel scale raises actual speed mostly by amplifying saturated actions, not by learning a smoother high-speed controller.
- Higher forward speed increases the same steering/formation correction delay:
  - lateral error has less time to decay before the vehicle moves farther forward;
  - heading correction overshoot becomes larger;
  - downstream pair errors propagate faster from pair_1/pair_2 into pair_3/pair_4.
- Stronger speed/catch-up action can fix gap and platoon speed, but it also pushes followers harder while they are laterally correcting, which increases centerline max or tail-pair deviation.
- Mild shield helps because it adds deterministic lateral/centerline correction without retraining, but if shield or catch-up is too strong it can fight pairwise formation and move the error downstream.
- Current best practical conclusion:
  - `scale=12.5 + mild shield` is a good aggregate speed-up.
  - `catchup_action=-0.38` is a stronger speed/gap variant but needs visual validation because centerline max and tail pairs worsen slightly.
  - A fully clean high-speed model likely needs a dedicated high-speed curriculum, not just scaling actions after a low-speed policy has saturated.

2026-06-24 high-speed curriculum attempt and final speed-up decision:

- Code changes made for controlled experiments:
  - Added optional `local_reward_gap_coef`, `local_reward_first_follower_gap_scale`, and `local_reward_last_follower_gap_scale` in `PlatoonAlgorithmRouter`.
  - Added optional shield controls for `pair3_lateral_*` and `pair4_lateral_*` so tail-pair correction can be tested without changing defaults.
  - Added optional guarded catch-up controls:
    - `env.safety_shield.catchup_lateral_limit`
    - `env.safety_shield.catchup_centerline_limit`
  - All new controls default to disabled/no-op behavior except local gap metrics logging, so the previous good baseline behavior is preserved unless overrides are passed.
  - Compile check passed:
    - `python3 -m py_compile source/my_exts/marl_platoon/algorithms/shield.py source/my_exts/marl_platoon/algorithms/router.py source/my_exts/marl_platoon/tasks/platoon/config.py`
- High-speed curriculum S1/S2/S3 training did not produce a better checkpoint:
  - S1 run:
    - `logs/rsl_rl/platoon_happo/2026-06-24_12-17-24_platoon5_highspeed_curriculum_s1_scale125_tail`
    - `model_400.pt` had speed improvement but gap worsened to `0.313`; do not adopt.
  - S2 run with gap reward:
    - `logs/rsl_rl/platoon_happo/2026-06-24_12-35-42_platoon5_highspeed_curriculum_s2_gap_from400`
    - Training metrics looked better, but deterministic eval worsened center/lateral; do not adopt.
  - S3 run from the clean baseline with low LR and gap shaping:
    - `logs/rsl_rl/platoon_happo/2026-06-24_12-54-57_platoon5_highspeed_curriculum_s3_lowrl_gap_from_clean`
    - No bad-orientation resets during training, but deterministic eval was worse than the original checkpoint parameter-only speed-up:
      - `model_350.pt`: leader `0.365`, platoon `0.316`, speed_err `0.070`, center `0.084`, lateral `0.133`, gap `0.314`
      - `model_400.pt`: command mean only `0.363`, not a fair speed-up, and lateral/pair metrics worsened
      - `model_450.pt`, `model_best.pt`, `model_final.pt`: center/lateral/pair/gap worse; do not adopt.
- Additional parameter tests after S3:
  - `scale=12.1 + mild shield`:
    - leader `0.351`, platoon `0.305`, speed_err `0.077`
    - too little actual speed improvement; not useful as the high-speed answer.
  - `scale=12.5 + catchup_action=-0.38 + guarded catchup_lateral_limit=0.16 + catchup_centerline_limit=0.18`:
    - leader `0.365`, platoon `0.315`, speed_err `0.070`
    - center worsened to `0.097`, center max peak `0.309`, pair_1 became worst; reject.
  - `scale=12.5 + centerline_turn_gain=0.28 + lateral_turn_gain=0.30`:
    - leader `0.365`, platoon `0.316`, speed_err `0.070`, center `0.082`
    - close to the best setting, but gap `0.306` and pair_3/pair_4 still worse than baseline; not better than `lateral_turn_gain=0.32`.
- Final current recommendation:
  - Do not use any newly trained high-speed curriculum checkpoint.
  - Use the original clean no-attack checkpoint:
    - `logs/rsl_rl/platoon_happo/2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200/model_final.pt`
  - Use this parameter-only speed-up for play/testing:
    - wheel scale `12.5`
    - `env.safety_shield.lateral_turn_gain=0.32`
    - `env.safety_shield.centerline_turn_gain=0.28`
    - attack disabled
  - This is not a strict every-metric improvement, but it is the best safe aggregate speed-up found:
    - leader speed improves `0.352 -> 0.365`
    - platoon speed improves `0.304 -> 0.316`
    - speed error improves `0.078 -> 0.070`
    - center mean/max/peak improve slightly
    - lateral mean/max/peak improve slightly
    - pair_1/pair_2 improve
    - no bad-orientation resets and attack is off
    - known tradeoff: gap `0.290 -> 0.302`, pair_3 `0.070 -> 0.086`, pair_4 `0.118 -> 0.122`
  - Interpretation: the current policy is action-saturated; a strictly no-degradation high-speed controller likely requires architecture/reward changes beyond short curriculum fine-tuning.

2026-06-24 note on whether longer high-speed training would help:

- Extending the current S3-style high-speed fine-tune is not recommended as-is.
- Evidence from S3:
  - early checkpoint `model_350.pt` was already worse than the parameter-only speed-up on gap/pair metrics;
  - later checkpoints (`model_450.pt`, `model_best.pt`, `model_final.pt`) further worsened centerline/lateral/gap instead of recovering;
  - training had no bad-orientation resets, so the failure is not instability/OOM, but objective/action-distribution mismatch.
- Main cause:
  - the policy is frequently saturated near `happo_action_clip=0.4`;
  - wheel scale `12.5` increases real speed by amplifying saturated actions, but it also amplifies steering and formation corrections;
  - gap/catch-up rewards can improve speed/gap but push followers while they are still laterally correcting, which moves error into centerline peak and tail pairs.
- Therefore more iterations under the same reward/shield/action setup are likely to overfit the speed/gap objective and degrade formation further.
- A useful longer training run would need a redesigned curriculum, not just more iterations:
  - gradual scale/speed schedule instead of jumping directly to high-speed scale;
  - explicit anti-saturation/action-head regularization or a cleaner throttle-vs-turn control decomposition;
  - stronger max-pair/tail-pair constraints while avoiding hard catch-up during lateral correction;
  - deterministic eval after each stage and rollback to the best checkpoint.

2026-06-24 high-speed redesign follow-up and adopted no-attack speed config:

- New training/eval attempt:
  - Ran `platoon5_highspeed_curriculum_v2_s1_scale122_tail_nogap` from the clean no-attack checkpoint.
  - Eval directory:
    - `logs/rsl_rl/platoon_happo/2026-06-24_14-32-13_platoon5_highspeed_curriculum_v2_s1_scale122_tail_nogap/eval_noattack_deploy_scale125_mild_1000`
  - Result:
    - `model_350.pt`: leader `0.365`, platoon `0.316`, speed_err `0.069`, center `0.084`, lateral `0.132`, gap `0.313`
    - `model_400.pt`: center/lateral/gap worsened further
    - `model_best.pt` and `model_final.pt`: center `0.088`, lateral `0.140`, gap `0.335`
  - Conclusion: continuing high-speed policy training still degrades formation/gap; do not adopt this new checkpoint.
- Added optional execution-layer controls for diagnostics:
  - `env.safety_shield.forward_bias_gain`
  - `env.safety_shield.forward_bias_clip`
  - `env.safety_shield.forward_bias_speed_margin`
  - `env.safety_shield.forward_bias_min_command`
  - `env.safety_shield.forward_bias_min_gap`
  - Defaults are disabled/no-op, so old behavior is preserved.
  - Compile check passed after adding these and prior gap/tail controls.
- Forward-bias tests were rejected as default:
  - `scale=12.0`, `clip=0.45`, forward bias `0.60/0.045`:
    - leader `0.395`, platoon `0.337`, speed_err `0.055`
    - but center `0.119`, lateral `0.185`; too much formation degradation.
  - softer forward bias:
    - leader `0.369`, platoon `0.319`, speed_err `0.067`
    - but center max peak `0.304`, lateral `0.146`; still too aggressive.
  - leader-only forward bias did not solve the downstream lateral issue.
- Scale scan without forward bias:
  - `scale=12.2`: leader `0.355`, platoon `0.308`, lateral `0.130`; stable but too little speed gain.
  - `scale=12.3`: leader `0.356`, platoon `0.310`; not enough speed gain.
  - `scale=12.4`: leader `0.364`, platoon `0.314`; close, but not better than `scale=12.5`.
- Tail-pair shield tuning:
  - Reducing tail correction worked better than increasing it:
    - `pair3_lateral_gain_scale=0.85`
    - `pair4_lateral_gain_scale=0.90`
  - With `scale=12.5`, mild shield, and no catch-up change:
    - leader `0.365`, platoon `0.316`, speed_err `0.069`
    - center `0.081`, lateral `0.132`, gap `0.305`
    - pair_4 improves vs old 12.5 mild (`0.122 -> 0.119`)
    - gap remains slightly worse.
- Final adopted no-attack high-speed deployment config:
  - Keep the clean checkpoint:
    - `logs/rsl_rl/platoon_happo/2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200/model_final.pt`
  - Use overrides:
    - `env.actions.joint_vel_*.scale=12.5`
    - `env.safety_shield.lateral_turn_gain=0.32`
    - `env.safety_shield.centerline_turn_gain=0.28`
    - `env.safety_shield.pair3_lateral_gain_scale=0.85`
    - `env.safety_shield.pair4_lateral_gain_scale=0.90`
    - `env.safety_shield.catchup_action=-0.355`
    - attack disabled
  - Eval output:
    - `eval_speedup_scale125_tailreduce_catchup355_cmd030_045_1000/eval_summary.csv`
  - Result vs old `scale=12.5 + mild shield`:
    - leader `0.365 -> 0.365`
    - platoon `0.3159 -> 0.3166`
    - speed_err `0.0697 -> 0.0690`
    - center mean `0.0822 -> 0.0814`
    - center max mean `0.1744 -> 0.1745` (essentially unchanged and still below 12.0 baseline `0.1802`)
    - center peak `0.2425 -> 0.2514` (higher than old 12.5 mild but still below 12.0 baseline `0.2519`)
    - lateral mean `0.1323 -> 0.1307`
    - lateral peak `0.4124 -> 0.4049`
    - gap `0.3023 -> 0.3015`
    - pair_2 `0.1914 -> 0.1878`
    - pair_4 `0.1221 -> 0.1206`
    - bad reset `0`, attack `0`
  - More aggressive alternatives:
    - `catchup_action=-0.36`: better gap/lateral/speed, but center peak rises to `0.2540`; use only if visual play still looks good.
    - `catchup_action=-0.37`: stronger average improvements, but center peak `0.2653`; not default.
  - Current conclusion: do not continue high-speed training from the tested curriculum checkpoints. The best result so far is the clean checkpoint plus tuned high-speed execution/shield parameters above.

2026-06-24 clarification on "high-speed retraining" vs current recommendation:

- There is no intended contradiction:
  - In principle, a truly better high-speed controller should be obtained by a redesigned high-speed curriculum, because the low-speed clean policy is action-saturated near `happo_action_clip=0.4`.
  - In practice, the high-speed retraining attempts actually tested so far did not improve the controller; they worsened center/lateral/gap metrics in deterministic evaluation.
- Therefore the current operational recommendation is:
  - Do not continue the already-tested S1/S2/S3/v2 high-speed fine-tunes or simply extend their training time.
  - Use the clean checkpoint plus tuned execution/shield parameters as the current best no-attack high-speed deployment config.
- The future research recommendation is different:
  - If retraining is attempted again, it should not be "same setup, more iterations".
  - It needs a redesigned training setup: gradual speed/scale schedule, better throttle-vs-turn decomposition or anti-saturation regularization, max-pair/tail-pair constraints, and deterministic eval/rollback at each stage.
- Short version:
  - "High-speed retraining is likely needed for a fundamentally better controller" is the long-term diagnosis.
  - "The high-speed retraining runs we tried are not usable" is the current experimental result.
  - For now, use the clean model with the tuned high-speed parameters.

2026-06-24 current best practical scheme:

- Current best usable no-attack scheme is not a newly trained checkpoint.
- Use the clean checkpoint:
  - `logs/rsl_rl/platoon_happo/2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200/model_final.pt`
- Use tuned high-speed execution/shield overrides:
  - attack off
  - wheel scale `12.5`
  - `happo_action_clip=0.4`
  - `lateral_turn_gain=0.32`
  - `centerline_turn_gain=0.28`
  - `pair3_lateral_gain_scale=0.85`
  - `pair4_lateral_gain_scale=0.90`
  - `catchup_action=-0.355`
  - command speed range `[0.30, 0.45]`
- This is currently preferred over all newly trained high-speed curriculum checkpoints.

2026-06-24 high-speed long fine-tune result and new best no-attack config:

- Ran longer no-attack high-speed fine-tune:
  - run: `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack`
  - source checkpoint: `logs/rsl_rl/platoon_happo/2026-06-23_22-37-20_platoon5_clean_noattack_v8_centerline_from150_ft200/model_final.pt`
  - training was stopped around iteration `734/1049` because post-700 training logs showed centerline/lateral degradation.
- Deterministic eval of saved checkpoints under no-attack high-speed config showed:
  - `model_650.pt`: platoon `0.3247`, speed_err `0.0620`, center `0.0816`, lateral `0.1174`, gap `0.3985`
  - `model_700.pt`: platoon `0.3251`, speed_err `0.0631`, center `0.0825`, lateral `0.1065`, gap `0.4123`
  - `model_best.pt` was not best for formation metrics.
- Diagnosis:
  - `model_700.pt` learned much better lateral/pair behavior, but default shield drop distance caused longitudinal gap compression/drift.
  - Changing `env.safety_shield.d_drop` is the key fix.
  - `d_drop=1.75` with weaker catch-up made the platoon too slow and gap too large.
  - `d_drop=1.45` with `catchup_action=-0.355` balanced speed, gap, centerline, and lateral metrics.
- New best no-attack high-speed result, 2000-step eval:
  - checkpoint: `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/model_700.pt`
  - eval output: `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/eval_model700_ddrop145_2000/eval_summary.csv`
  - command speed `0.3807`
  - leader speed `0.3625`
  - platoon speed `0.3308`
  - speed error `0.0607`
  - centerline mean `0.0157`, max mean `0.0331`, peak `0.0647`
  - lateral mean `0.0070`, max mean `0.0137`, peak `0.0317`
  - gap mean `0.2112`
  - heading mean `0.0051`, pair heading mean `0.0010`
  - pair lateral means: `[0.0062, 0.0073, 0.0076, 0.0072]`
  - bad-orientation reset `0`, attack `0`
  - last 500 eval steps stayed stable: platoon `0.3320`, speed_err `0.0590`, center `0.0238`, lateral `0.0084`, gap `0.2486`, min pair gap `1.4989`.
- Code default update:
  - set `SafetyShieldCfg.d_drop=1.45` in `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - set fallback `BadHeadingShieldCfg.d_drop=1.45` in `source/my_exts/marl_platoon/algorithms/shield.py`
  - compile check passed for `config.py`, `shield.py`, and `router.py`.
- Current recommendation:
  - Use `model_700.pt` from this run.
  - Use no attack, speed range `[0.30, 0.45]`, wheel scale `12.5`, action clip `0.4`, lateral gain `0.32`, centerline gain `0.28`, pair3 scale `0.85`, pair4 scale `0.90`, catchup action `-0.355`, and `d_drop=1.45`.
