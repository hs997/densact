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
- GitHub backup:
  - Pushed to `git@github.com:hs997/densact.git`
  - branch: `freeze/cagan-step3-dualchannel-logging`
  - commit: `78ec599 Add high-speed platoon model 700 backup`
  - included tracked backup artifacts:
    - `model_700.pt`
    - `eval_model700_ddrop145_2000/eval_summary.csv`
    - `eval_model700_ddrop145_2000/eval_steps_model_700.csv`

2026-06-24 city straight-road scene visual update:

- Goal: keep the current long straight road and platoon dynamics, but make play/rendering look less empty and closer to a CARLA/Gazebo-style urban straight-road scene.
- Code change:
  - Added `_city_box()` helper in `source/my_exts/marl_platoon/tasks/platoon/config.py`.
  - Added global visual-only static city assets under `/World/CityScenery/*`, not under `{ENV_REGEX_NS}`.
  - Added sidewalks, curbs, edge lines, center dashed line, low roadside barriers, mixed-height buildings, window strips, streetlights, and brighter cool dome lighting.
- Important implementation detail:
  - City scenery uses `CuboidCfg` without `collision_props`, `rigid_props`, or physics material.
  - The original `HighwayCollision` physical road surface is unchanged.
  - Because city assets are global prims rather than env-replicated prims, they are visible for `--num_envs 1` play/video but are not copied once per training environment.
  - Rewards, observations, actions, shield logic, attack logic, robot count, and trained checkpoints were not changed.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - A 1-env / 1-iteration headless IsaacLab smoke run passed and created the environment successfully:
    - run: `logs/rsl_rl/platoon_happo/2026-06-24_23-37-08`
    - confirmed 5 active action terms, policy observation shape `(55,)`, and 15 reward terms.
  - A headless video play test with the current best no-attack checkpoint also passed:
    - checkpoint: `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/model_700.pt`
    - video: `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
    - extracted frame: `/tmp/platoon_city_frame_40.png`
  - Visual frame check confirmed the city scene is visible around the straight road and does not block the platoon or road path.
- Note:
  - The first sandboxed smoke attempt failed because the sandboxed Isaac process could not see CUDA (`No CUDA GPUs are available`), while `nvidia-smi` showed the RTX 3080 was available. The successful validation used the approved non-sandbox GPU run.

2026-06-24 training command note after city scene update:

- The city straight-road scenery is now part of `PlatoonHAPPOEnvCfg.SceneCfg`; no extra Hydra override is needed to use it.
- Existing training commands remain valid because the visual city assets do not change robot dynamics, reward, observation, action, shield, or attack configuration.
- For continuing the previous light-attack checkpoint into the easy-attack sanity stage, use the existing `platoon5_easyattack_from_light_sanity100` command with `--load_run 2026-06-24_22-11-13_platoon5_lightattack_from_model700_ft_continue400` and `--checkpoint model_final.pt`.

2026-06-25 city visual scale/color refinement:

- User feedback:
  - The first city scene was usable, but buildings were too large, making the robots look too small.
  - Buildings and streetlights looked too colorless and the building blocks did not read enough like houses.
- Code change in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - Imported `CylinderCfg` and added `_city_cylinder()` for rounder streetlight poles.
  - Reduced building height/footprint and moved front faces farther from the road.
  - Replaced the previous large block-like buildings with smaller colored houses/buildings.
  - Added roof slabs, doors, individual window blocks, streetlight arms, darker metal poles, and warm emissive lamp heads.
  - Kept all city objects under `/World/CityScenery/*` as visual-only global prims.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless video play with `model_700.pt` passed using the current best no-attack play overrides.
  - Final play video path:
    - `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Extracted final frame:
    - `/tmp/platoon_city_v3_frame_40.png`
  - Visual check: buildings are smaller, colored, and include roof/door/window cues; streetlights have visible dark poles, arms, and warm lamp heads; the road and five-vehicle platoon remain unobstructed.
- Scope note:
  - This is still a lightweight IsaacLab primitive city scene, not an imported CARLA/Gazebo asset pack.
  - Dynamics, reward, attack, observations, actions, shield, and trained checkpoint files were not changed.

2026-06-25 city block USD module update:

- User feedback:
  - Buildings still did not look enough like buildings.
  - Buildings were too scattered and too far from the road edge.
  - Requested using more ready-made building-like models, scaled down and arranged like a normal city street.
- Local asset search result:
  - No directly usable local city/building USD asset pack was found in the repo assets or standard Isaac Sim install paths.
  - Isaac Sim had `warehouse_creator`/`scene_blox` related tooling, but not a simple local building USD pack that could be reliably referenced offline.
- Code/asset update:
  - Added local reusable USD building modules under `assets/city_straight/`:
    - `building_shop.usda`
    - `building_office_low.usda`
    - `building_apartment.usda`
  - Added `CITY_SHOP_USD_PATH`, `CITY_OFFICE_USD_PATH`, `CITY_APARTMENT_USD_PATH` and `_city_usd()` to `source/my_exts/marl_platoon/tasks/platoon/config.py`.
  - Replaced the old scattered `_city_box()` building/roof/window/door collection with repeated `UsdFileCfg` instances.
  - New layout uses continuous rows of scaled building modules on both sides of the straight road:
    - north row around `y=4.05`
    - south row around `y=-4.05`
    - spacing roughly every `3.5 m` along `x`
  - South-side buildings are rotated 180 degrees so storefronts face the road.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play with `model_700.pt` passed; Kit loaded the new local USD building modules and completed scene creation.
  - Final play video path:
    - `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Extracted frame:
    - `/tmp/platoon_city_usd_block_frame_40.png`
  - Visual check: buildings now form a more continuous street/block layout and sit much closer to the road edge. Storefronts, doors, windows, roof caps, office glass rows, and warmer shop materials are visible.
- Scope note:
  - This is not an external CARLA/Gazebo city asset import because no local offline pack was found.
  - It is now local USD-model based rather than only config-level cube boxes, making future replacement with real downloaded assets straightforward.
  - Road physics, robots, trained checkpoints, reward, observation, action, shield, and attack logic were not changed.

2026-06-25 city visual realism/scale pass:

- User raised two remaining visual issues:
  - Buildings still did not read as realistic building facades.
  - Buildings/road looked too large relative to the five small robot cars.
- Key design decision:
  - Do not scale the robot URDF or change the physical road width for this visual issue.
  - Scaling the robots or collision road would risk invalidating the trained policy and vehicle dynamics.
  - Instead, fix perceived scale with visual-only USD/config changes: narrower visible road, closer sidewalks, closer/smaller buildings, and a tighter viewer.
- Code/asset changes:
  - Narrowed the visible `assets/highway_straight/highway_straight.usda` road:
    - asphalt half-width reduced from `1.5` to `0.90`
    - shoulders, guardrails, grass, and highway light positions moved closer to the centerline
    - lane lines moved from `y=±0.75` to `y=±0.62`
  - Updated `source/my_exts/marl_platoon/tasks/platoon/config.py` city layout:
    - sidewalks moved from `y=±2.05` to `y=±1.55`
    - curbs moved from `y=±1.58` to `y=±1.13`
    - edge lines moved from `y=±1.25` to `y=±0.78`
    - sidewalk back edges moved from `y=±2.45` to `y=±1.88`
    - building rows moved from around `y=±3.15/±3.18` to `y=±2.40/±2.43`
    - building Y/Z scale reduced from about `(0.78, 0.66~0.72)` to `(0.58, 0.56~0.62)`
    - streetlight poles/arms/heads moved closer to the sidewalks.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - First sandboxed play failed because the sandbox could not access CUDA/GPU (`No CUDA GPUs are available`), not because of the scene edits.
  - Re-ran play with non-sandbox GPU access; RTX 3080 was detected and the run completed.
  - Headless play loaded `model_700.pt`, created the 5-agent environment, and generated video successfully.
  - Latest video:
    - `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Latest extracted frame:
    - `/tmp/platoon_city_v5_scale_frame_40.png`
  - Visual check:
    - The road now reads narrower, the buildings are closer to the road, the vehicle platoon is unobstructed, and the scale relation is better than the previous wide-road view.
    - The city is still a lightweight low-poly/local USD scene, not a photoreal CARLA/Gazebo asset import.
 - Git note:
  - `.gitignore:13` ignores `**/*.usda`, so `assets/highway_straight/highway_straight.usda` and `assets/city_straight/*.usda` are not shown by normal `git status`.
  - If this visual scene needs GitHub backup, use `git add -f assets/highway_straight/highway_straight.usda assets/city_straight/*.usda` together with the config changes.

2026-06-25 external realistic building asset pass:

- User feedback:
  - Current size relation is acceptable, but buildings still look like block assemblies rather than real urban buildings.
  - Requested ready-made realistic city building models, imported and tested until usable.
- Asset search / licensing conclusions:
  - Sketchfab search found realistic downloadable models, but direct model download API returned `401`; not used because it requires an authenticated Sketchfab token.
  - Khronos sample models were checked but rejected for this use: `VirtualCity` is not cleanly project-usable for this repo, and `Sponza` is not a city street.
  - Poly Haven CC0 assets were selected as the main usable source:
    - `modular_urban_apartments_facade`
    - `modular_factory_facade`
  - Gazebo Fuel `Athoms/House 2` CC0 was downloaded as a fallback under `assets/city_realistic/gazebo_fuel/athoms_house2/`; its old Collada/DAE pipeline is less reliable, so it was not enabled in the active scene.
- Asset/code changes:
  - Downloaded Poly Haven 1k glTF packages and textures under:
    - `assets/city_realistic/polyhaven/source_gltf/modular_urban_apartments_facade/`
    - `assets/city_realistic/polyhaven/source_gltf/modular_factory_facade/`
  - Converted them to USD with Isaac `scripts/tools/convert_mesh.py --collision-approximation none`:
    - `assets/city_realistic/polyhaven/usd/modular_urban_apartments_facade_1k.usd`
    - `assets/city_realistic/polyhaven/usd/modular_factory_facade_1k.usd`
  - Updated `source/my_exts/marl_platoon/tasks/platoon/config.py`:
    - added Poly Haven USD path constants
    - added facade rotation constants so the glTF/Y-up facade modules stand vertically in Isaac
    - replaced active building rows with repeated Poly Haven facade USD instances
    - added visual-only backing walls, roof strips, and bases so the facade modules read as complete buildings rather than floating window/door panels
    - moved/lowed the south-side building mass so the default camera is not blocked by a blank wall
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed after the final edits.
  - Several headless play checks were run with non-sandbox GPU access using:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/model_700.pt`
    - `--num_envs 1`
    - no attack: `env.algorithm.enable_attack=false`
  - Final play completed successfully:
    - 5 active action terms
    - observation shape `(55,)`
    - HAPPO checkpoint loaded
    - no new city USD load error
    - video written to `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Final inspected frames:
    - `/tmp/platoon_city_final_frame_20.png`
    - `/tmp/platoon_city_final_frame_40.png`
    - `/tmp/platoon_city_final_frame_70.png`
  - Visual result: the scene now uses externally sourced CC0 building facade models with real texture maps, continuous right-side urban blocks, streetlights, narrowed road, and no vehicle obstruction in the tested camera view.
- Scope / git note:
 - This pass changed visual scene assets/config only; robot dynamics, rewards, observations, attack/shield logic, and trained checkpoints were not changed.
  - `.gitignore` ignores `**/*.usd`, `**/*.usda`, `**/*.usdc`, and `**/*.usdz`; backup to GitHub will require forced adding the new USD/USDA assets, for example `git add -f assets/city_realistic/ assets/city_straight/ assets/highway_straight/highway_straight.usda`.

2026-06-25 city composition / camera balance pass:

- User feedback:
  - Current building materials are acceptable, but the overall proportion and layout are not coordinated.
  - Problem observed: when the whole five-vehicle platoon is visible, the city buildings are not readable; when buildings are readable, the platoon is not fully visible.
- Diagnosis:
  - This is mainly a composition/viewer problem, not a policy/training problem.
  - The previous default viewer was too high and too top-down:
    - old `eye=(5.5, -2.2, 2.8)`
    - old `lookat=(-1.8, 0.0, 0.35)`
  - That made the road and platoon visible, but compressed the city into a background wall.
- Changes in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - Changed the default viewer to a lower, road-side trailing composition:
    - `eye=(4.6, -4.25, 2.05)`
    - `lookat=(-2.2, 0.12, 0.55)`
  - Slightly lowered the north-side building backing wall and roof strip:
    - `building_back_north` height reduced from `3.10` to `2.68`
    - `building_roof_north` z moved from `3.12` to `2.70`
  - Reason: keep the building windows/doors visible while removing excess blank upper wall area.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Tested two candidate camera compositions via play overrides.
  - Candidate 1 made building details clear but clipped the blue front car at the lower-right edge.
  - Candidate 2 kept all five vehicles visible and retained readable building/window/streetlight detail; this was applied to config.
  - Final default-config play completed successfully with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/model_700.pt`
    - `--num_envs 1`
    - no attack: `env.algorithm.enable_attack=false`
    - 5 active action terms
    - observation shape `(55,)`
  - Final video path:
    - `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Final inspected frames:
    - `/tmp/platoon_city_balanced_final_frame_20.png`
    - `/tmp/platoon_city_balanced_final_frame_45.png`
    - `/tmp/platoon_city_balanced_final_frame_75.png`
  - Visual result: all five vehicles remain in frame while building windows/doors/facades and streetlights are visible; the scene no longer has the earlier tradeoff where either the platoon or the city detail had to be sacrificed.
- Scope:
 - This pass changed only default viewer composition and visual building backing height.
  - No changes were made to robot dynamics, trained checkpoints, rewards, observations, shield, or attack logic.

2026-06-25 city building scale-up pass:

- User feedback:
  - The material/asset quality is now acceptable, but the building modules still look too small.
  - Doors/windows and whole facade modules appeared close to or smaller than the robot car scale.
- Diagnosis:
  - The previous composition fix made the city readable, but the active Poly Haven facade scale was still too small relative to the robot URDF.
  - The fix should be visual-only: enlarge the facade USD instances and their backing wall/roof, without scaling the robots or changing physics.
- Changes in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - Increased active facade scales:
    - `CITY_URBAN_FACADE_SCALE` from `(0.075, 0.160, 0.060)` to `(0.105, 0.220, 0.060)`
    - `CITY_FACTORY_FACADE_SCALE` from `(0.075, 0.100, 0.060)` to `(0.100, 0.135, 0.060)`
    - `CITY_FACADE_Z` from `0.30` to `0.46` so enlarged facades sit correctly on the ground.
  - Increased backing building mass to match:
    - `building_back_north` height from `2.68` to `3.92`
    - `building_roof_north` z from `2.70` to `3.94`
    - south-side low backing also raised modestly from `0.70` to `1.04` to support the enlarged near-side facades without becoming a blank high wall.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Default-config headless play completed with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/model_700.pt`
    - `--num_envs 1`
    - no attack: `env.algorithm.enable_attack=false`
    - 5 active action terms
    - observation shape `(55,)`
    - no new USD/scenery load error
  - Final video path:
    - `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Final inspected frames:
    - `/tmp/platoon_city_scaleup1_frame_20.png`
    - `/tmp/platoon_city_scaleup1_frame_45.png`
    - `/tmp/platoon_city_scaleup1_frame_75.png`
  - Visual result:
    - Door/window modules now read larger than the robot car body scale.
    - Buildings are noticeably more urban-scale while the full five-vehicle platoon remains visible and unobstructed.
- Scope:
  - Visual-only scene scale update.
  - No changes to trained checkpoints, rewards, observations, action mapping, safety shield, attack logic, or robot physics.

2026-06-25 city building scale-up2 pass:

- User feedback:
  - Buildings still looked too small; requested another roughly 2x enlargement.
- Diagnosis:
  - Scaling both road sides by 2x made the facade modules and windows closer to real building scale, but a naive 2x scale on the near/south side blocked the default camera with a blank wall/roof.
  - The better composition is asymmetric for rendering: far/north side carries the large readable city wall, while near/south side stays lower so the platoon remains visible.
- Changes in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - Enlarged far/north facade instances by about 2x relative to the previous scale-up pass:
    - `CITY_URBAN_FACADE_SCALE` is now `(0.210, 0.440, 0.060)`.
    - `CITY_FACTORY_FACADE_SCALE` is now `(0.200, 0.270, 0.060)`.
    - `CITY_FACADE_Z` is now `0.92`.
  - Increased far/north backing wall and roof to match:
    - `building_back_north` z/height now `3.92 / 7.84`.
    - `building_roof_north` z now `7.86`.
  - Added separate near-side scale constants to avoid default-camera occlusion:
    - `CITY_NEAR_URBAN_FACADE_SCALE = (0.105, 0.220, 0.060)`.
    - `CITY_NEAR_FACTORY_FACADE_SCALE = (0.100, 0.135, 0.060)`.
    - `CITY_NEAR_FACADE_Z = 0.46`.
  - Restored the south backing wall/roof to low height:
    - `building_back_south` z/height `0.52 / 1.04`.
    - `building_roof_south` z `1.06`.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Default-config headless play completed successfully with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/model_700.pt`
    - `--num_envs 1`
    - no attack: `env.algorithm.enable_attack=false`
    - 5 active action terms
    - observation shape `(55,)`
    - no new USD/scenery load error
  - Final video path:
    - `logs/rsl_rl/platoon_happo/2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack/videos/play/rl-video-step-0.mp4`
  - Final inspected frames:
    - `/tmp/platoon_city_scaleup2_fixed_frame_20.png`
    - `/tmp/platoon_city_scaleup2_fixed_frame_45.png`
    - `/tmp/platoon_city_scaleup2_fixed_frame_75.png`
  - Visual result:
    - Far-side building doors/windows now read clearly larger than the robot car body scale.
    - The full five-vehicle platoon remains visible in the default camera.
    - The earlier full-2x near-side occlusion is removed.
- Scope:
  - Visual-only city scale/layout update.
  - No changes to trained checkpoints, rewards, observations, action mapping, safety shield, attack logic, or robot physics.

2026-06-26 profile attack preset rebalance + new medium long run:

- User decision:
  - The old `medium` attack is no longer a normal training target because it jumps too hard from easy.
  - New intended preset table:
    - `light`: `max_fdi_pos=0.75`, `max_fdi_acc=0.10`, `max_dos_rate=0.02`
    - `easy`: `max_fdi_pos=1.25`, `max_fdi_acc=0.25`, `max_dos_rate=0.05`
    - `medium`: `max_fdi_pos=2.00`, `max_fdi_acc=0.50`, `max_dos_rate=0.10`
  - `hard` remains unchanged as a stress-test preset.
- Code changes:
  - Updated only `source/my_exts/marl_platoon/algorithms/router.py` `attack_presets` for `light/easy/medium`.
  - Did not change `off/hard`, `attack_mode=profile` logic, HAPPO network structure, reward terms, or safety shield.
  - Added `scripts/reinforcement_learning/rsl_rl/plot_platoon_metrics.py` to generate:
    - reward curves
    - tracking/formation errors
    - pairwise lateral errors
    - attack strength/budget curves
    - attack learning/objective curves
    - teacher/meta reward curves
    - optimizer/stability curves
    - shield curves
    - speed curves
    - `summary_last100.csv`
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/algorithms/router.py source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - `python3 -m py_compile scripts/reinforcement_learning/rsl_rl/plot_platoon_metrics.py` passed.
  - 10-iteration sanity run completed:
    - run: `logs/rsl_rl/platoon_happo/2026-06-26_00-00-06_platoon5_city_newmedium_attack_check10`
    - checkpoint source: `2026-06-25_21-26-34_platoon5_city_easyattack_from_light_ft400/model_final.pt`
    - CSV confirmed:
      - `attack_enabled=1.0`
      - `attack_level=medium`
      - `attack_mode=profile`
      - `attack_max_fdi_pos=2.0`
      - `attack_max_fdi_acc=0.5`
      - `attack_max_dos_rate=0.1`
    - Last sanity metrics:
      - `speed_error_abs_mean=0.1116`
      - `centerline_error_abs_mean=0.0240`
      - `lateral_error_abs_mean=0.0368`
      - `shield_lateral_turn_mean=0.0189`
      - `critic_grad_norm=11.75`
      - no bad-orientation reset was observed.
  - Plot script was tested on the sanity run and generated 9 plots under:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-00-06_platoon5_city_newmedium_attack_check10/plots`
- 20000-iteration training:
  - `nohup` background launch exited early in this shell environment, so the working launch method is `setsid -f`.
  - Active run:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000`
  - Main log:
    - `train_platoon5_city_newmedium_attack_20000.log`
  - Training Python PID at launch:
    - `2716507`
  - A watcher is running and will generate plots after PID `2716507` exits:
    - watcher PID: `2717603`
    - watcher log: `plot_platoon5_city_newmedium_attack_20000_after_finish.log`
  - Early long-run check after metrics row count `36`:
    - `update=1748`
    - `attack_max_fdi_pos=2.0`
    - `attack_max_fdi_acc=0.5`
    - `attack_max_dos_rate=0.1`
    - `speed_error_abs_mean=0.0831`
    - `centerline_error_abs_mean=0.0562`
    - `lateral_error_abs_mean=0.0740`
    - `lateral_error_abs_max=0.3799`
    - `shield_lateral_turn_mean=0.0272`
    - `critic_grad_norm=5.285`
    - `termination_reset_on_bad_ori=0.0`
    - `termination_time_out=1.0`
  - ETA from training log near iteration `1722/21704` was about `12:46:26`, so final plots are expected after the long run completes.

2026-06-26 new-medium 20000-iteration completion status:

- Training completed successfully.
  - Run:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000`
  - Main log:
    - `train_platoon5_city_newmedium_attack_20000.log`
  - Metrics rows:
    - `20000` training rows plus CSV header.
  - Final checkpoint:
    - `model_final.pt`
  - Best checkpoint:
    - `model_best.pt`
    - Last best save: `iter=13864`, `score=859.5993`
  - Training ended normally and saved `model_final.pt`.
- Plots were generated under:
  - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/plots`
  - Generated files include reward curves, tracking errors, pair lateral errors, attack strength, attack learning/meta-teacher curves, optimizer curves, shield curves, speed curves, and `summary_last100.csv`.
- Attack preset in the 20000 run was confirmed as the new `medium`:
  - `attack_max_fdi_pos=2.0`
  - `attack_max_fdi_acc=0.5`
  - `attack_max_dos_rate=0.1`
- Last-100 metrics for `model_final.pt` region:
  - `command_speed_mean=0.3752`
  - `leader_speed_mean=0.2574`
  - `platoon_speed_mean=0.2663`
  - `speed_error_abs_mean=0.1145`
  - `gap_error_abs_mean=0.2415`
  - `lateral_error_abs_mean=0.1338`
  - `lateral_error_abs_max=0.5406`
  - `centerline_error_abs_mean=0.0708`
  - `collision_rate=0.0`
  - `termination_time_out=1.0`
  - `termination_reset_on_bad_ori=0.0`
  - `shield_lateral_turn_mean=0.0406`
  - `shield_lateral_critical_rate=0.0342`
  - `physical_cost=0.8828`
  - `critic_grad_norm=11.0842`
- Around the best checkpoint window near `iter=13864`, behavior was cleaner than final:
  - `speed_error_abs_mean=0.0976`
  - `gap_error_abs_mean=0.2395`
  - `gap_error_max_abs=0.5954`
  - `lateral_error_abs_mean=0.0769`
  - `lateral_error_abs_max=0.3685`
  - `centerline_error_abs_mean=0.0642`
  - `centerline_error_abs_max=0.3045`
  - `heading_error_abs_mean=0.0416`
  - `collision_rate=0.0`
  - `termination_time_out=1.0`
  - `termination_reset_on_bad_ori=0.0`
  - `shield_lateral_turn_mean=0.0420`
  - `shield_lateral_critical_rate=0.0512`
  - `physical_cost=0.1642`
  - `critic_grad_norm=10.9338`
- Interpretation:
  - The 20000-iteration run is valid and robust under the rebalanced `medium` attack.
  - It completes episodes with no collision and no bad-orientation reset.
  - `model_final.pt` is usable but is not the best behavioral checkpoint because late training increased lateral error and physical cost.
  - Prefer `model_best.pt` for play/demo/evaluation.
- Play validation:
  - `model_best.pt` was played under new `medium` attack, profile mode, with city scene and video enabled.
  - Video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Extracted frames:
    - `/tmp/platoon_newmedium_best_play_frame_40.png`
    - `/tmp/platoon_newmedium_best_play_frame_100.png`
    - `/tmp/platoon_newmedium_best_play_frame_150.png`
  - Visual check on frame 100 showed all five vehicles visible, aligned on the road, with the enlarged city background visible and no obvious vehicle side-runaway in that frame.
- Current recommendation:
  - Use:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - Do not treat `model_final.pt` as the primary demo checkpoint unless specifically testing late-training behavior.

2026-06-26 city scene two-lane visual update:

- User request:
  - The previous city dressing still looked too close, too modular, and not like a mature CARLA/Gazebo-style street.
  - Road should be a real-looking two-lane road.
  - The platoon should drive in the lane center, on the lane closer to the building side.
- Code changed:
  - File:
    - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Added city helper assets with `CuboidCfg`, `CylinderCfg`, and `UsdFileCfg`.
  - Added a visual-only two-lane asphalt road with:
    - near lane center at `y=0.0`
    - other lane center at `y=-1.20`
    - yellow dashed divider at `y=-0.60`
    - sidewalks, curbs, white edge lines, crosswalks, streetlights, trees, and building/facade rows.
  - Kept the policy/reward centerline at `y=0.0`, so the trained 5-car policy still drives on its expected centerline while visually appearing in the near-building lane.
  - Moved the old highway USD downward and narrowed/offset the collision slab so the old markings do not visually conflict with the new two-lane road.
  - Rebalanced nearby city assets to avoid the previous large facade/wall occluding the cars from the default play camera.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play with `model_best.pt` from:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    completed successfully under `medium` profile attack.
  - Latest play video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Extracted check frames:
    - `/tmp/platoon_city_clear_frame_01.png`
    - `/tmp/platoon_city_clear_frame_02.png`
- Visual conclusion:
  - The scene now has a usable two-lane road layout, the platoon is visually centered in the near lane, and the large blank wall/occlusion issue was removed.
  - The current local assets are still mostly modular facade panels and simple USD blocks, so the result is improved and usable but not yet CARLA-grade full city geometry.
 - To reach true CARLA-like realism, the next step should be importing a complete city map/block asset package rather than continuing to assemble buildings from small facade modules.

2026-06-26 complete city/map asset import validation:

- User request:
  - The hand-built city still looked too poor.
  - Switch from manually assembled building blocks/facades to a complete city/map asset, then replay and verify it works.
- Asset search and import:
  - Tried Isaac Sim remote environment asset:
    - `Outdoor/Rivermark/rivermark.usd`
  - Rejected Rivermark for this project because play produced many missing sub-asset warnings such as missing vegetation props and invalid point-instancer prototypes, then stalled long enough that the process had to be killed.
  - Imported an external complete city block asset instead:
    - SourceCityToolkit GLB `Main_Intersection_v2.glb`
    - local GLB path:
      - `assets/city_complete/sourcecity_glb/Main_Intersection_v2.glb`
    - converted/fixed USD path:
      - `assets/city_complete/sourcecity_usda_fixed/Main_Intersection_v2.usda`
  - The converted USD originally contained broken absolute references like `/props/...`; these were exported to USDA and fixed to relative references under:
    - `assets/city_complete/sourcecity_usda_fixed/props`
    - `assets/city_complete/sourcecity_usda_fixed/textures`
- Code changed:
  - File:
    - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Added:
    - `CITY_SOURCECITY_USD_PATH`
    - `CITY_SOURCECITY_SCALE = (0.045, 0.045, 0.045)`
    - `city_full_map = _city_usd("sourcecity_main_intersection", CITY_SOURCECITY_USD_PATH, (0.0, 4.5, 0.015), scale=CITY_SOURCECITY_SCALE)`
  - Old modular facade rows were left in the config but moved down with:
    - `CITY_STREET_FACADE_Z = -20.0`
    - `CITY_BACK_FACADE_Z = -20.0`
    so they no longer dominate the visible scene.
  - The trained policy/reward coordinate system was preserved:
    - lane center remains `y=0.0`
    - custom two-lane road still provides the clean vehicle lane and collision surface
    - the imported complete city map is visual context around the road, not a new driving surface for the policy.
- Play validation:
  - Checkpoint:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - Attack/play settings:
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
    - speed command range `[0.30,0.45]`
  - Headless camera video play completed successfully after SourceCity import.
  - Latest video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Extracted validation frames:
    - `/tmp/sourcecity_play3_frame_01.png`
    - `/tmp/sourcecity_play3_frame_02.png`
    - `/tmp/sourcecity_play3_frame_03.png`
  - Control debug remained normal during play:
    - at step 20, leader actual x velocity was about `0.548` for command `0.448`, reward about `0.951`
    - at step 80, leader actual x velocity was about `0.374`, reward about `0.971`
- Visual conclusion:
  - The scene now uses a complete imported city block asset instead of hand-built block buildings.
  - The visible frame contains continuous street/building context, road boundaries, lane markings, lamps, and textured structures.
  - Vehicles are still visible, not blocked by the imported map, and continue driving in the intended lane.
  - This is tested usable for current play/demo.
  - Limitation: SourceCity is a stylized open city block asset, not a full CARLA-grade photorealistic town. For true CARLA-like realism, the next upgrade should use a higher-quality self-contained city USD/map package or a CARLA/OpenDRIVE-to-USD conversion pipeline.

2026-06-26 mature-city visual upgrade with NVIDIA Omniverse City Demo:

- User request:
  - SourceCity was still not realistic enough.
  - Target visual quality should be closer to a mature autonomous-driving simulation city, not a stylized or blocky city.
- Asset decision:
  - Downloaded NVIDIA Omniverse downloadable pack:
    - `AECO_CityDemoPack_NVD@10011`
    - local zip:
      - `assets/city_complete/AECO_CityDemoPack_NVD_10011.zip`
    - extracted directory:
      - `assets/city_complete/nvidia_city_demo_pack`
  - The relevant city assembly is:
    - `assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Assemblies/assembly_City.usd`
  - USD inspection showed:
    - `upAxis = Z`
    - `metersPerUnit = 0.01`
    - assembly size is about `190984 x 176288 x 14303` stage units, so the source is centimeter-scale and represents a very large city block.
- Code changed:
  - File:
    - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Added:
    - `CITY_NVIDIA_DEMO_USD_PATH`
    - `CITY_NVIDIA_DEMO_SCALE = (0.0045, 0.0045, 0.0045)`
  - Replaced the prior `SourceCity` visual map with:
    - `_city_usd("nvidia_city_demo", CITY_NVIDIA_DEMO_USD_PATH, (0.0, 12.0, -0.12), scale=CITY_NVIDIA_DEMO_SCALE)`
  - Added/expanded a road-side `urban_forecourt` visual slab to hide the imported asset's exposed dark grid ground and make the building side look like a continuous urban sidewalk/plaza.
  - Kept the custom two-lane road, collision slab, lane center, rewards, algorithm, and shield logic unchanged so the trained platoon policy remains compatible.
  - Added helper script:
    - `scripts/tools/inspect_usd_asset.py`
    for USD bounds/reference inspection through IsaacLab `AppLauncher`.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play was run with:
    - `model_best.pt` from `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
    - speed command range `[0.30,0.45]`
  - Play completed successfully.
  - No Rivermark-style missing sub-asset stall occurred.
  - Latest video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Final extracted frames:
    - `/tmp/nvidia_city_final_frame_01.png`
    - `/tmp/nvidia_city_final_frame_02.png`
    - `/tmp/nvidia_city_final_frame_03.png`
  - Control debug stayed consistent with previous play:
    - step 20 leader actual x velocity about `0.548` for command `0.448`, reward about `0.951`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`
- Visual conclusion:
  - The new scene is a clear upgrade over SourceCity and the hand-built city:
    - real-looking building facades
    - continuous urban street frontage
    - less blocky/artificial building geometry
    - vehicles remain visible and lane-centered
  - This is currently the best tested city visual version.
  - Remaining visual limitation:
    - the vehicle model is still a small robot platform rather than a realistic sedan/SUV, so even with a mature city background the overall screenshot will not fully match CARLA unless the vehicle mesh/scale is also replaced with more realistic car geometry.
    - some far-side grid ground remains outside the main vehicle/building view, but it does not affect the current demo camera.

2026-06-26 city/building alignment refinement:

- User request:
  - The NVIDIA building asset itself was acceptable, but the buildings were not visually in the same area as the vehicles.
  - Required repeated play/screenshot validation before reporting completion.
- Diagnosis from repeated extracted frames:
  - Early placements such as `city_full_map=(0.0, 12.0, -0.12)`, `(-25.0, 2.8, -0.12)`, `(25.0, -3.0, -0.12)`, and `(25.0, 2.8, -0.12)` were not acceptable:
    - some left a large exposed dark grid/plaza between the road and buildings,
    - some moved buildings too close and occluded the vehicles,
    - some made the road look like a separate elevated/isolated strip.
  - The remaining "wall/trench" look was traced to the old `highway_straight.usda` visual scene, especially its guardrail/shoulder geometry, not to the new city asset.
- Code changed:
  - File:
    - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Kept the NVIDIA Omniverse City Demo assembly:
    - `assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Assemblies/assembly_City.usd`
  - Final city placement:
    - `city_full_map = _city_usd("nvidia_city_demo", CITY_NVIDIA_DEMO_USD_PATH, (10.0, 1.8, -0.12), scale=CITY_NVIDIA_DEMO_SCALE)`
  - Hid the old highway visual asset by moving `HighwayScene` to `z=-20.0`.
    - The physical road/collision support remains through `HighwayCollision`.
  - Kept the custom two-lane visual road and added/retained thin city-side road dressing:
    - `two_lane_asphalt`
    - `near_lane_subtle_wear`
    - `far_lane_subtle_wear`
    - low `curb_near_buildings` / `curb_far_side`
    - thin `sidewalk_near_buildings`, `sidewalk_far_side`, and `urban_forecourt`
  - Lowered the play camera to a more street-level view:
    - `viewer.eye=(4.3, -2.75, 1.28)`
    - `viewer.lookat=(-2.25, 0.35, 0.46)`
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed after the final edits.
  - Headless play was run repeatedly with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
    - command range `[0.30,0.45]`
  - Final play completed successfully and wrote:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Final validation frames copied to:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_alignment_validation/nvidia_city_streetside5_frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_alignment_validation/nvidia_city_streetside5_frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_alignment_validation/nvidia_city_streetside5_frame_03.png`
  - Control debug remained normal during final play:
    - step 20 leader actual x velocity about `0.548` for command `0.448`, reward about `0.951`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`
- Visual conclusion:
  - The final frame shows the vehicle platoon, two-lane road, street lamp, sidewalk/plaza surface, and realistic building facade in the same visual street area.
  - The old highway guardrail/trench look is removed.
  - Vehicles remain visible and are not blocked by the imported city asset.
  - Remaining limitation: the robot chassis mesh is still not a real passenger-car mesh, so the city now looks much more mature but the vehicle model remains the main visual realism bottleneck.

2026-06-27 city/building yaw alignment fix:

- User request:
  - Fix the city scene where the building facade looked perpendicular to the road, like a building wall across the road end.
  - Do not change HAPPO, rewards, attack, shield, checkpoints, or training configuration.
- Coordinate conclusion:
  - The platoon and lane geometry use X as the driving direction:
    - robot initial positions are spaced along X,
    - command tracking uses `lin_vel_x`,
    - city lane/road boxes are long in X.
  - Therefore the city building long side/facade should be parallel to X and its depth/normal should extend along +/-Y.
- Code changed:
  - File:
    - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Added the city root rotation constant:
    - `CITY_NVIDIA_DEMO_ROT_Z90 = (0.7071068, 0.0, 0.0, 0.7071068)`
  - Changed the active NVIDIA city root transform:
    - before: position `(10.0, 1.8, -0.12)`, no explicit `rot`, yaw `0 deg`
    - after: position `(10.0, 1.1, -0.12)`, `rot=CITY_NVIDIA_DEMO_ROT_Z90`, yaw `+90 deg`
  - Road boxes, lane lines, vehicle initial positions, HAPPO, reward, attack, shield, and checkpoints were not changed for this request.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed with the existing checkpoint:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - Final validation video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_yaw_alignment_validation/nvidia_city_yaw90_y11_frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_yaw_alignment_validation/nvidia_city_yaw90_y11_frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_yaw_alignment_validation/nvidia_city_yaw90_y11_frame_03.png`
  - Visual result:
    - vehicles drive along the road while the building row runs parallel to the road,
    - buildings are on the road side rather than across the vehicle path,
    - no city building blocks the vehicle route in the checked frames.
 - Control debug remained normal during play:
    - step 20 leader actual x velocity about `0.548` for command `0.448`, reward about `0.951`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`

2026-06-27 city/building lateral placement refinement:

- User request:
  - Keep the yaw fix exactly as-is:
    - `rot = (0.7071068, 0.0, 0.0, 0.7071068)`
    - yaw `+90 deg`
  - Do not rotate again.
  - Move the city asset laterally toward the road so the buildings sit near the roadside/sidewalk rather than far behind a large blank area.
  - Do not move vehicles, road/lane geometry, HAPPO, reward, attack, shield, checkpoints, or training config.
- Coordinate conclusion:
  - Road and vehicle direction is still X.
  - Lateral road-normal adjustment should therefore be in Y.
- Tested placements:
  - Previous yaw-fixed baseline: `city_full_map pos = (10.0, 1.1, -0.12)` left too much empty pavement/black ground between road and buildings.
  - `Y=-2.0` and `Y=-6.0` improved little because the visible city block has substantial internal offset from the root prim.
  - `Y=-18.0` moved buildings closer but still left a wide empty roadside area.
  - Final selected placement: `city_full_map pos = (10.0, -30.0, -0.12)`.
- Code changed:
  - File:
    - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Final active transform:
    - `pos=(10.0, -30.0, -0.12)`
    - `rot=CITY_NVIDIA_DEMO_ROT_Z90`
  - Yaw was not changed from the prior fix.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_sidewalk_close_validation/nvidia_city_yaw90_y-300_frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_sidewalk_close_validation/nvidia_city_yaw90_y-300_frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_sidewalk_close_validation/nvidia_city_yaw90_y-300_frame_03.png`
  - Visual result:
    - building long edges remain parallel to the X-direction road,
    - buildings sit much closer to the road side,
    - the previous large empty/black roadside gap is substantially reduced,
    - vehicles remain visible and their route is not blocked.
  - Control debug stayed normal:
    - step 20 leader actual x velocity about `0.548` for command `0.448`, reward about `0.951`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`

2026-06-27 continuous streetwall scene pass:

- User request:
  - Keep the current road and vehicle route unchanged.
  - Keep `city_full_map` yaw at `rot=(0.7071068, 0.0, 0.0, 0.7071068)`.
  - Stop relying only on the single `city_full_map` root transform because its internal blocks leave large gaps along the straight road.
  - Build a continuous roadside building band along the X-direction road, with a consistent near-road front edge and no large black-grid holes.
- Code changed:
  - File: `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - Added visual-only global prims under `/World/CityScenery/*`:
    - `streetwall_forecourt`
    - `streetwall_bldg_01` through `streetwall_bldg_10`
    - `streetwall_facade_01` through `streetwall_facade_06`
    - `streetwall_storefront_01` through `streetwall_storefront_10`
    - `streetwall_windows_01` through `streetwall_windows_10`
  - The streetwall bodies are long Cuboids aligned with X, so their long edge is parallel to the road.
  - The main building body front edge is approximately `y=2.38`, outside the near sidewalk/streetlight band and away from the vehicle lane.
  - A grey `streetwall_forecourt` slab covers the sidewalk/building gap so the black grid does not show through.
  - The first six mid/far streetwall segments also use the Poly Haven urban facade USD for more realistic windows/doors; nearer segments stay as cleaner Cuboid bodies to avoid facade side-component artifacts near the camera.
- Scope guard:
  - No changes were made to vehicles, road/vehicle trajectory, HAPPO, reward, attack, shield, checkpoints, or training config.
  - `city_full_map` remains:
    - `pos=(10.0, -30.0, -0.12)`
    - `rot=CITY_NVIDIA_DEMO_ROT_Z90`
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed repeatedly using:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
  - Final validation video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_streetwall_final/frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_streetwall_final/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_streetwall_final/frame_03.png`
  - Visual result:
    - Road remains centered and unobstructed.
    - New streetwall buildings run parallel to the X-direction road and sit along the road side.
    - The previous large black-grid roadside gap is covered by the forecourt and continuous building row.
    - Streetlights, lane markings, vehicle trajectory, and the five-car platoon remain visible and unblocked.

2026-06-27 full-map correction after streetwall rejection:

- User correction:
  - The desired fix is to change/use the previous whole city map scene, not to add several building blocks near the road.
- Corrective action:
  - Removed all newly added `streetwall_*` prims from `source/my_exts/marl_platoon/tasks/platoon/config.py`.
  - Switched the active complete city map back to the imported SourceCity map:
    - prim name: `/World/CityScenery/sourcecity_main_intersection`
    - USD: `assets/city_complete/sourcecity_usda_fixed/Main_Intersection_v2.usda`
    - final pos: `(0.0, 6.5, 0.015)`
    - scale: `CITY_SOURCECITY_SCALE = (0.045, 0.045, 0.045)`
    - rot: default identity
  - The NVIDIA `city_full_map` instance is no longer the active full map in config.
- Scope guard:
  - No changes were made to vehicles, road physics, vehicle trajectory, HAPPO, reward, attack, shield, checkpoint, or training config.
  - Existing visual road/sidewalk/lane-line overlays remain to keep the trained policy lane readable.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed using:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
  - Final validation video:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/rl-video-step-0.mp4`
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/sourcecity_fullmap_y65_validation/frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/sourcecity_fullmap_y65_validation/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/sourcecity_fullmap_y65_validation/frame_03.png`
  - Visual result:
    - The visible city context now comes from a complete imported SourceCity map asset, not newly added building-block rows.
    - The full map is offset to the road side and does not block the five-car route.
    - Control debug remained normal during play:
      - step 20 leader actual x velocity about `0.548` for command `0.448`
      - step 80 leader actual x velocity about `0.374`

2026-06-28 current play command note:

- Current visual scene should be played with the SourceCity full-map config:
  - `city_full_map = sourcecity_main_intersection`
  - `pos=(0.0, 6.5, 0.015)`
  - `scale=CITY_SOURCECITY_SCALE`
- Recommended checkpoint for play:
  - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
- Recommended play overrides keep the tested medium profile attack settings:
  - `enable_attack=true`
  - `attack_level=medium`
  - `attack_mode=profile`
  - command speed range `[0.30,0.45]`

2026-06-28 city model clarification:

- Current active imported full-map model is SourceCity:
  - `assets/city_complete/sourcecity_usda_fixed/Main_Intersection_v2.usda`
  - prim name `sourcecity_main_intersection`
- This is not the previous NVIDIA Omniverse City Demo full city asset:
  - `assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Assemblies/assembly_City.usd`
- The switch happened after rejecting the added `streetwall_*` block approach, to return to a single complete imported map asset without adding roadside building blocks.

2026-06-28 complete-map route alignment update:

- User clarified the scene should keep using a complete city map as the basis, then either align roadside buildings with the vehicle route or choose a suitable path through the map.
- Code changed only in `source/my_exts/marl_platoon/tasks/platoon/config.py`; no changes were made to vehicles, HAPPO, rewards, attack, shield, checkpoints, or training configuration.
- Active complete map switched back from SourceCity to the more mature NVIDIA Omniverse City Demo full map:
  - prim: `/World/CityScenery/nvidia_city_demo`
  - USD: `assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Assemblies/assembly_City.usd`
  - pos: `(10.0, -30.0, -0.12)`
  - scale: `CITY_NVIDIA_DEMO_SCALE = (0.0045, 0.0045, 0.0045)`
  - rot: `CITY_NVIDIA_DEMO_ROT_Z90 = (0.7071068, 0.0, 0.0, 0.7071068)`
- Kept the trained policy route as the existing X-direction straight road. This is the safe choice because the current policy/reward/control stack is trained for a straight platoon route; planning a curved city-internal route would require trajectory/reward/control changes and likely retraining.
- Removed the previous SourceCity side-background effect. The NVIDIA full-map buildings now form the active complete-map backdrop and run parallel to the straight vehicle route.
- Reworked the road-side fill from an overly wide grey slab into a narrow sidewalk/plaza strip:
  - `city_urban_forecourt`: pos `(0.0, 2.95, 0.042)`, size `(150.0, 2.70, 0.010)`
  - `city_urban_forecourt_strip_01`: pos `(0.0, 1.66, 0.049)`
  - `city_urban_forecourt_strip_02`: pos `(0.0, 4.20, 0.049)`
  - Purpose: cover the black grid gap between road curb and complete-map building facades without adding new building blocks.
- Confirmed no `streetwall_*` prims remain in `config.py`.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed with checkpoint:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - Attack/play settings:
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
    - speed range `[0.30,0.45]`
  - Extracted validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/nvidia_fullmap_route_v2/frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/nvidia_fullmap_route_v2/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/nvidia_fullmap_route_v2/frame_03.png`
  - Visual conclusion:
    - Buildings are now from a complete city map and run parallel to the X-direction road.
    - Buildings sit along the road side instead of being a separate SourceCity background.
    - The vehicle route remains clear and unobstructed.
    - The black grid gap from the first NVIDIA test is covered by the narrower sidewalk/plaza strip.
  - Control debug remained normal during play:
    - step 20 leader actual x velocity about `0.548` for command `0.448`, reward about `0.951`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`

2026-06-28 play command note after NVIDIA full-map route alignment:

- Recommended interactive play command should use the current `config.py` scene, which now loads the NVIDIA full city map:
  - `/World/CityScenery/nvidia_city_demo`
  - pos `(10.0, -30.0, -0.12)`
  - rot `(0.7071068, 0.0, 0.0, 0.7071068)`
- Recommended checkpoint remains:
  - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
- Recommended play overrides remain:
  - `enable_attack=true`
  - `attack_level=medium`
  - `attack_mode=profile`
  - speed range `[0.30,0.45]`

2026-06-28 planned urban road route update:

- User said the current city map is correct and asked for a more reasonable road route.
- Key constraint:
  - The current HAPPO policy/reward/control stack is trained for an X-direction straight platoon route with lane center around `y=0`.
  - A curved route through the complete city map would require path-tracking observations/rewards/control changes and likely retraining.
  - Therefore the safe route plan keeps the vehicle path straight in X, but makes it read visually as a normal urban arterial through the city block.
- Code changed only in `source/my_exts/marl_platoon/tasks/platoon/config.py`.
- No changes were made to vehicles, HAPPO, reward, attack, shield, checkpoint, training config, or physical route.
- Active complete map remains NVIDIA full city:
  - `/World/CityScenery/nvidia_city_demo`
  - pos `(10.0, -30.0, -0.12)`
  - rot `(0.7071068, 0.0, 0.0, 0.7071068)`
- Planned visual route:
  - main driving direction: world X
  - active platoon lane center: `CITY_LANE_CENTER_Y = 0.0`
  - opposite lane center: `CITY_OTHER_LANE_CENTER_Y = -1.20`
  - lane divider: `CITY_LANE_DIVIDER_Y = -0.60`
  - near/building-side road edge: about `y=0.52~0.73`
  - far-side road edge: about `y=-1.72~-1.93`
- Added visual-only route/intersection prims:
  - `intersection_west_asphalt`: pos `(-62.0, -0.60, 0.052)`, size `(4.40, 5.30, 0.008)`
  - `intersection_mid_asphalt`: pos `(-12.0, -0.60, 0.052)`, size `(4.00, 5.30, 0.008)`
  - `intersection_east_asphalt`: pos `(58.0, -0.60, 0.052)`, size `(4.40, 5.30, 0.008)`
  - `crosswalk_mid_01` to `crosswalk_mid_04`: x positions `-12.75`, `-12.30`, `-11.85`, `-11.40`
  - `stop_bar_mid_near_lane` and `stop_bar_mid_far_lane`: x `-9.60`
  - west/east stop bars remain at x `-59.55` and `55.55`
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
    - speed range `[0.30,0.45]`
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/nvidia_fullmap_route_planned_v2/frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/nvidia_fullmap_route_planned_v2/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/nvidia_fullmap_route_planned_v2/frame_03.png`
  - Visual conclusion:
    - The route now reads as a straight urban arterial through the city, with a visible mid-block/cross-street intersection ahead of the platoon.
    - Buildings remain from the complete NVIDIA city map and stay parallel to the road.
    - Vehicle route remains unobstructed.
  - Control debug remained normal:
    - step 20 leader actual x velocity about `0.548` for command `0.448`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`

2026-06-28 no-path-wrap play command note:

- User requested a play command that avoids path wrapping/splitting issues.
- Recommended pattern:
  - define short variables `ROOT`, `SIM`, `RUN`, `CKPT`;
  - build `PYTHONPATH` from those variables;
  - pass `--checkpoint "$CKPT"` instead of pasting one long checkpoint path into the play command.
- This avoids accidental copy/paste breakage when the UI visually wraps long absolute paths.

2026-06-28 play command split-error diagnosis:

- User pasted a failed play log showing:
  - `gymnasium.error.NameNotFound: Environment Isaac- doesn't exist`
  - `Marl-Platoon-HAPPO-v0: command not found`
  - multiple `env.xxx: command not found`
  - RTX driver verification warning.
- Diagnosis:
  - The shell command was split by a real newline, not just visually wrapped.
  - `--task` received only `Isaac-`; the remaining `Marl-Platoon-HAPPO-v0` was interpreted as a separate shell command.
  - The Hydra overrides were also interpreted as separate shell commands because the play command line ended before them.
  - The RTX driver warning appeared because the intended `--kit_args="--/rtx/verifyDriverVersion/enabled=false"` likely did not remain attached to the same play command.
- Fix added:
  - New executable wrapper script: `play_platoon_city.sh`
  - Syntax checked with `bash -n play_platoon_city.sh`.
  - The user can now run:
    - `cd /home/cnc/SSD_1T/xzw/IsaacLab-main && ./play_platoon_city.sh`
  - This avoids copy/paste path wrapping and missing-backslash issues.

2026-06-28 play script maintenance rule:

- User requested that future model/algorithm changes must also update the play wrapper script.
- Rule going forward:
  - Whenever the active checkpoint/model changes, update `RUN`/`CKPT` in `play_platoon_city.sh`.
  - Whenever algorithm overrides change, update the corresponding Hydra overrides in `play_platoon_city.sh`.
  - This includes attack level/mode, shield parameters, action scales, command speed range, task name, and any play-time HAPPO settings.
  - After editing the script, run `bash -n play_platoon_city.sh` before reporting completion.
- Added a maintenance comment to `play_platoon_city.sh`.
- Validation:
  - `bash -n play_platoon_city.sh` passed.

2026-06-28 road-route limitation after user play check:

- User played the scene and reported that the planned route did not look meaningfully different from the previous version.
- Diagnosis:
  - The current NVIDIA `assembly_City.usd` is a complete city/building visual assembly, but it is not behaving like a CARLA-style drivable road-network map for this task.
  - The stable vehicle route remains the task's own X-direction straight road at lane center `y=0`.
  - The added intersections/crosswalks are visual-only overlays on that same road, so the default play camera may show only a small visual difference.
  - The asset does not currently provide a clearly empty, long, vehicle-scale straight road corridor that can simply replace the trained route without changing the policy/reward/control stack.
- Practical implication:
  - A genuinely different route through the city map would require either a different complete drivable-map asset or path-tracking/reward/control changes plus retraining.
  - For the current trained HAPPO checkpoint, the safe path is still a straight X-direction urban arterial, with the complete city map used as aligned roadside context.
- Next visual-improvement direction if requested:
  - make the custom straight road itself visibly more like a real city arterial: stronger asphalt texture/contrast, realistic lane widths, clearer curb/sidewalk, more visible cross streets near the camera, traffic signs/lights, and less subtle road markings.
  - This would remain visual-only and keep the current model/checkpoint valid.

2026-06-28 current straight-road length audit:

- User asked how much straight road is currently usable.
- Current config values:
  - Physical collision road: `highway_collision` size `(2000.0, 2.95, 0.1)` at x center `0.0`, so physics support exists from approximately `x=-1000` to `x=+1000`.
  - Visible city asphalt: `city_road_asphalt` size `(150.0, 2.52, 0.036)` at x center `0.0`, so the rendered city road spans approximately `x=-75` to `x=+75`.
  - Lane dash markers run from centers `x=-54` to `x=+54`, each length `3.2`, so the marked lane section spans about `x=-55.6` to `x=+55.6` (`111.2 m`).
  - Cross/intersection visual markers are at `x=-62`, `x=-12`, and `x=58`, all within the visible 150 m road.
  - Robot leader starts at `x=1.0` and drives positive X, so visible road ahead from the initial leader position is about `74 m`.
  - Environment episode length is `20 s`; with play command speed `[0.30,0.45] m/s`, target travel per episode is roughly `6~9 m`, so the current 74 m visible forward road is more than enough for normal play episodes.
- Important distinction:
  - The NVIDIA full-city asset is visual context and does not currently provide a validated CARLA-style drivable straight road network.
  - The actually usable straight route is the task-owned X-direction road described above.

2026-06-28 facade-aligned suitable straight-road length clarification:

- User clarified that the question is not about physical road length, but about the length of a suitable straight road that fits the simulation scene and aligns cleanly along the building edge.
- Corrected interpretation:
  - The 2000 m `highway_collision` is only physics support and should not be counted as a suitable city street.
  - The 150 m `city_road_asphalt` is the visible road layer, but the full 150 m is not equally "perfect" because the complete city facade context is strongest around the central/visible city block.
  - The most visually suitable facade-aligned usable corridor in the current setup is approximately the segment between the west/east city intersections:
    - conservative high-quality segment: `x=-62` to `x=+58`, about `120 m`
    - full visible road layer: `x=-75` to `x=+75`, about `150 m`
    - lane-marked section: about `x=-55.6` to `x=+55.6`, about `111 m`
  - With the current leader initial position at `x=1.0`, the forward usable high-quality facade-aligned segment is roughly:
    - to east intersection `x=+58`: about `57 m`
    - to full visible road end `x=+75`: about `74 m`
- Conclusion:
 - The current complete-city alignment gives a suitable building-edge straight street of roughly `110~120 m` if using the central high-quality corridor, or up to `150 m` if counting the full visual road slab.
 - To use the whole high-quality segment in one play, the platoon could be spawned further back around `x=-50`, but that would change initial positions and should be treated as a scene/experiment change.

2026-06-28 city edge-aligned visual correction after allowing any parameters:

- User allowed changing any scene/play parameters as long as the building edge aligns with the road and the visual result is good.
- Tested the previous idea of spawning the platoon around `x=-48` to use more of the apparent long corridor.
- Result:
  - Headless play ran and vehicle control stayed normal, but extracted frames showed the camera and road were too close to the complete NVIDIA city asset's building body.
  - A temporary top-down check at `x=-51` showed a large building/roof area occluding the view near that segment.
  - Conclusion: the `x=-50` segment is not the cleanest facade-aligned route in the current NVIDIA `assembly_City.usd` placement, even though the visual road slab exists there.
- Final correction in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - Kept the complete NVIDIA city map as the active city asset:
    - prim `/World/CityScenery/nvidia_city_demo`
    - pos `(10.0, -30.0, -0.12)`
    - scale `(0.0045, 0.0045, 0.0045)`
    - rot `(0.7071068, 0.0, 0.0, 0.7071068)` / yaw `+90 deg`
  - Restored the platoon to the cleaner central road segment:
    - `robot`: `(1.0, 0.0, 0.5)`
    - `robot_2`: `(-0.5, 0.0, 0.5)`
    - `robot_3`: `(-2.0, 0.0, 0.5)`
    - `robot_4`: `(-3.5, 0.0, 0.5)`
    - `robot_5`: `(-5.0, 0.0, 0.5)`
  - Restored the tested street-level viewer:
    - `eye=(4.3, -2.75, 1.28)`
    - `lookat=(-2.25, 0.35, 0.46)`
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - `bash -n play_platoon_city.sh` passed; no model/algorithm setting changed, so the script command remains valid.
  - Headless play completed with:
    - checkpoint `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`
    - `attack_level=medium`
    - `attack_mode=profile`
    - speed range `[0.30,0.45]`
  - Control remained normal:
    - step 20 leader actual x velocity about `0.548` for command `0.448`, reward about `0.951`
    - step 80 leader actual x velocity about `0.374`, reward about `0.971`
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_edge_aligned_final/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_edge_aligned_final/frame_04.png`
- Visual conclusion:
  - The current best visual route is the central segment around the original spawn, not the `x=-50` segment.
  - Buildings run parallel to the X-direction road and sit along the right-side road edge.
  - The vehicle platoon remains visible, lane-centered, and unobstructed.
  - Do not move the platoon back to `x=-50` again unless the NVIDIA city root placement or an explicit alternative corridor is reworked first.

2026-06-28 short high-quality city-road segment correction:

- User rejected the prior "central road restore" because it looked too close to the older sparse/flat road setup; new target is stricter:
  - even a `20~30 m` segment is acceptable,
  - but it must look like a mature autonomous-driving city street,
  - road/building alignment and visible composition matter more than claiming a long corridor.
- Implemented a shorter, more polished visual corridor in `source/my_exts/marl_platoon/tasks/platoon/config.py` without changing HAPPO, reward, attack, shield, checkpoint, or training parameters:
  - kept the complete NVIDIA city map active as the realistic facade/backdrop asset:
    - prim `/World/CityScenery/nvidia_city_demo`
    - USD `assets/city_complete/nvidia_city_demo_pack/Demos/AEC/TowerDemo/CityDemopack/Assemblies/assembly_City.usd`
    - pos `(10.0, -33.0, -0.12)`
    - scale `(0.0045, 0.0045, 0.0045)`
    - rot `(0.7071068, 0.0, 0.0, 0.7071068)` / yaw `+90 deg`
  - replaced the visible road slab with a custom textured short asphalt USD:
    - prim `/World/CityScenery/two_lane_asphalt`
    - USD `assets/city_realistic/road/short_asphalt_road.usda`
    - texture `assets/city_realistic/road/asphalt_dark_cityroad.png`
    - pos `(2.0, -0.54, 0.052)`
    - length about `52 m`, width about `2.38 m`
  - kept vehicle route and starts unchanged:
    - `robot`: `(1.0, 0.0, 0.5)`
    - `robot_2`: `(-0.5, 0.0, 0.5)`
    - `robot_3`: `(-2.0, 0.0, 0.5)`
    - `robot_4`: `(-3.5, 0.0, 0.5)`
    - `robot_5`: `(-5.0, 0.0, 0.5)`
  - hid the mid-road crosswalk/intersection/stop-bar geometry by moving those visual-only prims to `z=-20.0`; this avoids the visible segment looking artificially cut off.
  - changed lane/edge markings from thick raised cuboids to thinner, shorter, lower profile visual marks:
    - edge lines z `0.055`, height `0.003`, width `0.026`
    - lane divider dashes length `1.45`, width `0.038`, height `0.003`
  - concrete sidewalk/curb/forecourt use the local NVIDIA concrete MDL material to avoid black grid/empty ground around the building edge.
- Validation commands/results:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - `bash -n play_platoon_city.sh` passed.
  - Headless play completed using the existing city medium checkpoint:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - `enable_attack=true`, `attack_level=medium`, `attack_mode=profile`
    - speed range `[0.30,0.45]`
  - Extracted validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_edge_dark_asphalt_v1/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_edge_clean_marks_v1/frame_01.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_edge_clean_marks_v1/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_edge_clean_marks_v1/frame_04.png`
- Visual conclusion after frame inspection:
  - The current best visual solution is a compact `~52 m` street segment, with a clearly usable `20~30 m` high-quality visible corridor.
  - Building facades are parallel to the X-direction road, located along the right-side road edge, and no longer appear as perpendicular blockers.
  - Road no longer shows large black grid/empty terrain in the validated camera; asphalt is dark textured, with curb/sidewalk/building edge visible.
  - Vehicle platoon stays in the near-building lane and remains unobstructed in the validated frames.
  - Frame `01` still includes the normal Isaac initialization/settling moment where vehicles briefly appear high; frames `02` and `04` show the stable play view.

2026-06-28 vehicle placement and wheel-height visual fix:

- User reported that the platoon was too far forward along the straight road direction and the wheels looked partially sunk into the road surface.
- Root cause of the wheel-height issue:
  - visible asphalt road is at about `z=0.052`,
  - old physical collision road top was about `z=0.0`,
  - so the vehicles physically drove on a lower invisible surface while the visual asphalt was rendered higher, making wheels look embedded.
- Visual-only scene fix in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - shifted the whole 5-car platoon backward by `2.25 m` along X while preserving `1.5 m` spacing:
    - `robot`: from `(1.0, 0.0, 0.5)` to `(-1.25, 0.0, 0.56)`
    - `robot_2`: from `(-0.5, 0.0, 0.5)` to `(-2.75, 0.0, 0.56)`
    - `robot_3`: from `(-2.0, 0.0, 0.5)` to `(-4.25, 0.0, 0.56)`
    - `robot_4`: from `(-3.5, 0.0, 0.5)` to `(-5.75, 0.0, 0.56)`
    - `robot_5`: from `(-5.0, 0.0, 0.5)` to `(-7.25, 0.0, 0.56)`
  - raised the hidden physical road collision body from center `z=-0.05` to `z=0.002`, so its top surface aligns with the visible asphalt around `z=0.052`.
  - set the collision road cuboid to `visible=False` to avoid z-fighting or covering the custom asphalt texture.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - Headless play completed with the existing city medium checkpoint and unchanged attack/HAPPO/reward/shield settings.
  - Control remained normal:
    - step 20 command `0.448`, leader actual x velocity about `0.628`, leader reward about `0.850`
    - step 80 command `0.448`, leader actual x velocity about `0.386`, leader reward about `0.980`
  - Extracted validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v1/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v1/frame_04.png`
- Visual conclusion:
  - The platoon is now farther back in the validated view rather than crowding the foreground/right edge.
  - Wheels no longer appear obviously buried under the asphalt in stable frames.
  - No model, reward, attack, shield, checkpoint, or training command was changed.

2026-06-28 second vehicle-back/height adjustment:

- User reported the platoon was still too far forward and wheel sinking was still visible.
- Follow-up visual-only scene change in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - shifted the whole 5-car platoon another `2.5 m` backward along X, preserving `1.5 m` spacing:
    - `robot`: `(-1.25, 0.0, 0.56)` -> `(-3.75, 0.0, 0.62)`
    - `robot_2`: `(-2.75, 0.0, 0.56)` -> `(-5.25, 0.0, 0.62)`
    - `robot_3`: `(-4.25, 0.0, 0.56)` -> `(-6.75, 0.0, 0.62)`
    - `robot_4`: `(-5.75, 0.0, 0.56)` -> `(-8.25, 0.0, 0.62)`
    - `robot_5`: `(-7.25, 0.0, 0.56)` -> `(-9.75, 0.0, 0.62)`
  - raised initial vehicle z from `0.56` to `0.62`.
  - raised hidden physical road collision center from `z=0.002` to `z=0.027`, putting its top about `2.5 cm` above the visual asphalt plane to compensate for the visual wheel/collision mismatch.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - `bash -n play_platoon_city.sh` passed.
  - Headless play completed with existing checkpoint and unchanged HAPPO/reward/attack/shield settings.
  - Control stayed normal:
    - step 20 command `0.448`, leader actual x velocity about `0.560`, reward about `0.939`
    - step 80 command `0.448`, leader actual x velocity about `0.377`, reward about `0.973`
  - Extracted validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v2/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v2/frame_04.png`
- Visual conclusion:
  - Vehicle formation is now clearly farther back in the camera view and no longer crowds the foreground.
  - Stable frames do not show obvious wheel burial; no obvious floating was introduced.
  - Play script remains valid because only scene geometry/start placement changed, not model/algorithm/checkpoint overrides.

2026-06-28 third vehicle-back/height adjustment:

- User still reported the platoon was not far enough back and wheels still appeared to sink into the ground.
- Follow-up visual/scene-only change in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - shifted the whole 5-car platoon another `5.0 m` backward along X while preserving `1.5 m` spacing:
    - `robot`: `(-3.75, 0.0, 0.62)` -> `(-8.75, 0.0, 0.70)`
    - `robot_2`: `(-5.25, 0.0, 0.62)` -> `(-10.25, 0.0, 0.70)`
    - `robot_3`: `(-6.75, 0.0, 0.62)` -> `(-11.75, 0.0, 0.70)`
    - `robot_4`: `(-8.25, 0.0, 0.62)` -> `(-13.25, 0.0, 0.70)`
    - `robot_5`: `(-9.75, 0.0, 0.62)` -> `(-14.75, 0.0, 0.70)`
  - raised vehicle initial z from `0.62` to `0.70`.
  - raised hidden physical road collision center from `z=0.027` to `z=0.062`; collision top is now about `z=0.112`, roughly `6 cm` above the visual asphalt plane at `z=0.052`.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - `bash -n play_platoon_city.sh` passed.
  - Headless play completed with unchanged checkpoint/HAPPO/reward/attack/shield settings.
  - Control stayed normal and leader velocity tracking improved relative to the previous adjustment:
    - step 20 command `0.448`, leader actual x velocity about `0.451`, reward about `0.9998`
    - step 80 command `0.448`, leader actual x velocity about `0.391`, reward about `0.984`
  - Extracted validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v3/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v3/frame_04.png`
    - zoom check: `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_vehicle_back_height_v3/frame_02_zoom.png`
- Visual conclusion:
  - The platoon is now much farther back in the default view; it no longer crowds the foreground.
 - The hidden collision surface is now deliberately above the visible asphalt to counter the wheel/road visual intersection seen by the user.
 - Play script remains valid because the script uses this config and no command-line model/algorithm overrides changed.

2026-06-28 camera/white-ground/wheel visual cleanup:

- User reported that after moving the platoon backward, the default camera did not follow, black ground grid lines hurt the visual quality, and the wheels still appeared to sink into the ground.
- Scene/play-only changes were made; HAPPO, reward, attack, shield, checkpoint, and training configs were not changed.
- Changes in `source/my_exts/marl_platoon/tasks/platoon/config.py`:
  - raised all 5 vehicle initial root z values from `0.70` to `0.80` while preserving the X positions and `1.5 m` spacing:
    - `robot`: `(-8.75, 0.0, 0.70)` -> `(-8.75, 0.0, 0.80)`
    - `robot_2`: `(-10.25, 0.0, 0.70)` -> `(-10.25, 0.0, 0.80)`
    - `robot_3`: `(-11.75, 0.0, 0.70)` -> `(-11.75, 0.0, 0.80)`
    - `robot_4`: `(-13.25, 0.0, 0.70)` -> `(-13.25, 0.0, 0.80)`
    - `robot_5`: `(-14.75, 0.0, 0.70)` -> `(-14.75, 0.0, 0.80)`
  - raised hidden physical road collision center from `z=0.062` to `z=0.082` to further reduce visual wheel/road intersection.
  - changed the global terrain plane to a light off-white material: diffuse `(0.92, 0.92, 0.89)`.
  - added/adjusted light road-edge paving covers:
    - `near_edge_paving`: pos `(2.0, 0.50, 0.096)`, size `(52.0, 0.34, 0.012)`, color `(0.78, 0.78, 0.74)`.
    - `far_edge_paving`: pos `(2.0, -2.12, 0.094)`, size `(52.0, 0.78, 0.012)`, color `(0.90, 0.90, 0.87)`.
    - `sidewalk_far_side` and `clean_outer_ground` remain light box surfaces, replacing the visible blue/black grid-like exposed ground on the far side.
  - moved the default viewer to follow the new vehicle formation center:
    - `eye`: `(4.3, -2.75, 1.28)`/older foreground view -> `(-5.2, -2.95, 1.38)`
    - `lookat`: old near-front lookat -> `(-11.75, 0.28, 0.54)`
- Changes in `play_platoon_city.sh`:
  - kept the same checkpoint and all existing HAPPO/attack/shield/action overrides.
  - extended `--kit_args` with:
    - `--/app/viewport/grid/enabled=false`
    - `--/app/viewport/defaults/guide/grid/visible=false`
  - This prevents Omniverse's viewport guide grid from reappearing during manual play.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py` passed.
  - `bash -n play_platoon_city.sh` passed.
  - Headless play completed with the existing medium-profile checkpoint:
    - checkpoint: `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - unchanged overrides: `enable_attack=true`, `attack_level=medium`, `attack_mode=profile`, command speed `[0.30,0.45]`, and the current shield/action-scale overrides.
  - Control stayed normal:
    - step 20 command `0.448`, leader actual x velocity about `0.899`, reward about `0.361` during startup transient.
    - step 80 command `0.448`, leader actual x velocity about `0.381`, reward about `0.975`.
  - Final validation frames:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_camera_ground_wheel_v5/frame_02.png`
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/videos/play/city_camera_ground_wheel_v5/frame_04.png`
- Visual conclusion:
  - The camera now follows the rear-shifted 5-car platoon; vehicles stay in the usable view instead of being too far forward.
  - The far-side exposed blue/black grid-like strip is covered by light white/gray paving.
 - The default Omniverse viewport grid is disabled in the play script.
 - The wheel/ground intersection is improved; final frames no longer show obvious wheel burial. A thin dark curb seam remains on the building side, but it reads as a road-edge shadow rather than the previous exposed grid/ground artifact.

2026-06-28 paper-figure data audit:

- User asked whether current saved data is enough for top-journal style figures: mechanism/problem diagrams, attack profile plots, training curves, attack-level comparisons, ablations, safety-distance curves, tracking-error curves, and attack-response plots.
- Current data inventory:
  - `logs/rsl_rl/platoon_happo/*/platoon_metrics.csv` exists for many runs from 2026-06-17 through 2026-06-26.
  - Latest main run:
    - `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/platoon_metrics.csv`
    - 20000 rows, 148 columns.
    - includes checkpoints, TensorBoard event file, `params/agent.yaml`, `params/env.yaml`, and `git/IsaacLab-main.diff`.
  - Existing plotting scripts:
    - `scripts/tools/plot_platoon_metrics.py`
    - `scripts/reinforcement_learning/rsl_rl/plot_platoon_metrics.py`
  - Some older runs already have generated figures, e.g. `2026-06-18_19-12-27_stage5_cagan/figures`.
- Metrics available in `platoon_metrics.csv` are sufficient for:
  - reward/training curves: `reward_env_mean`, reward term columns, `policy_loss`, `value_loss`, grad norms.
  - tracking errors: `speed_error_abs_mean`, `gap_error_abs_mean`, `lateral_error_abs_mean`, `centerline_error_abs_mean`, heading errors.
  - per-pair/per-agent lateral/centerline diagnostics: `lateral_pair_i_*`, `centerline_robot_i_*`.
  - safety proxies: `min_pair_gap_mean`, `collision_rate`, `termination_time_out`, `termination_reset_on_bad_ori`.
  - attack strength curves: `attack_max_fdi_pos`, `attack_max_fdi_acc`, `attack_max_dos_rate`, `fdi_abs_mean`, `dos_rate`, `obs_fdi_abs_mean`, `obs_dos_rate`, `act_fdi_abs_mean`, `act_dos_rate`.
  - meta/teacher curves: `teacher_*`, `local_reward_*`.
  - shield curves: `shield_*`.
- Existing attack-level runs support a preliminary no/light/easy/medium comparison, but not a perfectly controlled final paper comparison:
  - no attack candidate: `2026-06-24_17-58-29_platoon5_highspeed_bestcfg_ft_long_noattack`, 388 rows.
  - light candidate: `2026-06-25_20-32-44_platoon5_city_lightattack_from_model700_ft400`, 400 rows.
  - easy candidate: `2026-06-25_21-26-34_platoon5_city_easyattack_from_light_ft400`, 400 rows.
  - medium candidate: `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000`, 20000 rows.
  - Caveat: older light/easy runs use old budgets (`light pos=1.0, acc=0.1, dos=0.02`; `easy pos=2.0, acc=0.5, dos=0.05`) while current code maps `light/easy/medium` to `0.75/0.10/0.02`, `1.25/0.25/0.05`, `2.0/0.50/0.10`. For paper-quality comparison, re-evaluate/retrain under one consistent attack table.
- Current saved data is insufficient for rigorous full-method vs no-shield/no-local-reward/no-curriculum ablations:
  - no reliable run names or params show `enable_shield=false`.
  - no reliable run names or params show `local_reward_shaping=false`.
  - no completed controlled run isolates `attack_curriculum_warmup_updates=0` or “no curriculum” while keeping other settings identical.
  - Several historical tuning runs exist, but they changed multiple knobs and are not clean ablations.
- Existing finite-step eval data:
  - many `eval_summary.csv` and `eval_steps_*.csv` exist under no-attack/highspeed runs.
  - step CSV contains per-step `eval_step`, `min_pair_gap_mean`, tracking errors, centerline/lateral pair metrics, shield metrics, attack columns, etc.
  - However, `scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py` currently calls `set_inference_mode(..., disable_attack=True)`, so existing eval step CSVs are no-attack unless the script is modified or a separate attack-enabled eval is run.
- Figure support conclusion:
  - Directly supported now: problem/method schematic figures, attack-budget table/curve, training curves, medium-run reward/physical/attack/teacher/shield curves, current-method tracking/safety curves, city visualization snapshots.
  - Partially supported: no/light/easy/medium bars, because attack budgets are not all from the current unified preset table.
  - Not yet supported enough for top-journal claims: clean ablation bars and medium-attack single-rollout attack-response time series.
- Recommended next data work:
  1. Run controlled evals for the final checkpoint under `off/light/easy/medium` with the current attack presets and save both `eval_summary.csv` and `eval_steps_*.csv`.
  2. Add an attack-enabled mode to `eval_happo_platoon.py` or create a separate eval script that does not force `disable_attack=True`.
  3. Run clean ablations with only one change at a time:
     - full method,
     - `env.safety_shield.enable_shield=false` if the code path supports it,
     - `env.algorithm.local_reward_shaping=false`,
     - no attack curriculum / direct medium,
     - optional obs8 vs obs11 if the older observation configuration is still reproducible.
  4. For safety-distance figures, prefer eval step CSV over training CSV because it is a true rollout time series rather than an update-level aggregate.

2026-06-28 100k hard-curriculum paper-data run:

- User requested a longer unified training/data run for paper figures, with gradually increasing attack strength up to `hard`, total `100000` training iterations, followed by data/figure packaging.
- Added/modified automation files:
  - `scripts/tools/run_platoon_paper100k_hard_curriculum.sh`
    - staged curriculum from the current city-newmedium baseline:
      - `off`: 10000 iters
      - `light`: 15000 iters
      - `easy`: 20000 iters
      - `medium`: 25000 iters
      - `hard`: 30000 iters
      - total: 100000 iters
    - default start checkpoint: `logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
    - keeps current validated settings: `num_envs=64`, speed `[0.30,0.45]`, `happo_action_clip=0.4`, shield/action overrides from the current city model.
    - after training, it evaluates final checkpoint under `off/light/easy/medium/hard`, creates paper figures, and packages key metrics/params/figures plus `model_best.pt` and `model_final.pt` only.
  - `scripts/tools/plot_platoon_paper_figures.py`
    - creates combined training curves, attack-strength curves, safety/shield curves, stage comparison bars, per-vehicle/per-pair bars, and eval response figures.
  - `scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py`
    - added `--enable_attack_eval` so evaluation can keep configured attack levels active instead of always forcing no-attack eval.
  - `scripts/tools/plot_platoon_metrics.py`
    - switched matplotlib to `Agg` backend so per-run figures can be generated headlessly.
- Launch/debug findings:
  - Direct `nohup train.py` works, but ordinary `nohup bash run_platoon_paper100k_hard_curriculum.sh` was killed before Python output in this Codex process tree.
  - `setsid ... bash run_platoon_paper100k_hard_curriculum.sh` was validated with a one-stage, one-iteration background test and completed successfully.
  - The final long job was therefore launched with `setsid`.
- Active long job:
  - PID: `3111903`
  - RUN_TAG: `paper100k_hard_curriculum_20260628_224124`
  - pipeline log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper100k_hard_curriculum_20260628_224124.log`
  - package root after completion: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper100k_hard_curriculum_20260628_224124_package`
  - package tar after completion: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper100k_hard_curriculum_20260628_224124_package.tar.gz`
- Initial status:
  - `off` stage has entered the HAPPO training loop successfully.
  - Early logs show `attack=False`, `shield=True`, `num_envs=64`, stable startup, and `termination_reset_on_bad_ori=0.0000`.
  - Early speed tracking is normal for startup: logged `Metrics/base_velocity/error_vel_xy` roughly `0.04~0.06`.
  - Current runtime estimate from stage logs is about `2.2~2.3 s/iter`; a full 100000-iter run plus evaluations/packaging will likely take multiple days, not hours.

2026-06-28 clarification on whether current training matches previous training:

- Current 100k run is the same base training setup as the previous validated city run in the parts that affect the learned controller:
  - same task: `Isaac-Marl-Platoon-HAPPO-v0`
  - same task-local HAPPO architecture/checkpoint format
  - same 5-car setup and current city scene config
  - same speed range `[0.30,0.45]`
  - same `happo_action_clip=0.4`
  - same shield/action overrides: `d_drop=1.45`, `catchup_action=-0.355`, `lateral_turn_gain=0.32`, `centerline_turn_gain=0.28`, pair3/pair4 gain scales, wheel action scale `12.5`
  - same `num_envs=64`
- It is not the exact same schedule as the previous single-stage run:
  - it starts from `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - then runs staged curriculum: `off -> light -> easy -> medium -> hard`
  - total training is `100000` additional iterations
  - first stage is deliberately no-attack for longer baseline data before increasing attack.
- It also adds post-training evaluation/plot/package automation; these additions do not change HAPPO, reward, shield, attack presets, or checkpoint loading.
- Current process check:
  - PID `3111903` is still running.
  - Current child command is the `off` stage with `env.algorithm.enable_attack=false`.
  - Current log continues to show `termination_reset_on_bad_ori=0.0000`, so startup remains stable.

2026-06-28 realtime log viewing:

- User asked how to view current training logs in real time.
- Active log file:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper100k_hard_curriculum_20260628_224124.log`
- Recommended realtime command:
  - `tail -f /home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper100k_hard_curriculum_20260628_224124.log`
- Useful status commands:
  - `ps -p 3111903 -o pid,etime,cmd`
  - `pgrep -P 3111903 -a`
  - `tail -80 /home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper100k_hard_curriculum_20260628_224124.log`
- Current observed status at this note:
  - PID `3111903` still running.
  - Log size about `551K`.
  - Latest shown stage remains stable with `termination_reset_on_bad_ori=0.0000`, `time_out=1.0000`, and `Metrics/base_velocity/error_vel_xy` about `0.0576`.

2026-06-29 stage-switch terrain failure and fix:

- User reported that the original long run completed `off` but failed when switching to `light`.
- Failed run:
  - RUN_TAG: `paper100k_hard_curriculum_20260628_224124`
  - completed stage: `2026-06-28_22-41-39_paper100k_hard_curriculum_20260628_224124_s_off`
  - failure stage: `light`
  - error: `FileNotFoundError: Unable to open the usd file at path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd`
- Root cause:
  - `TerrainImporterCfg(terrain_type="plane")` eventually instantiates IsaacLab `GroundPlaneCfg`, whose default `usd_path` is the remote Omniverse grid plane.
  - The first stage had succeeded, but the second stage recreated the scene and remote/cached USD loading failed.
  - This was a terrain asset dependency problem, not HAPPO/reward/attack/shield logic.
- Rejected intermediate fix:
  - A local hand-written `assets/local_ground/local_ground_plane.usda` was tried briefly.
  - It avoided the remote USD but produced bad physical behavior in a 20-iter test: high backward velocity, `reset_on_bad_ori` rising to about `0.9`.
  - That local USD file was deleted and is not used.
- Final fix:
  - `source/my_exts/marl_platoon/tasks/platoon/config.py`
  - changed terrain from remote-backed `terrain_type="plane"` to local procedural mesh terrain:
    - imports `TerrainGeneratorCfg` and `MeshPlaneTerrainCfg`
    - `terrain_type="generator"`
    - `TerrainGeneratorCfg(size=(2000.0, 2000.0), num_rows=1, num_cols=1, sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)})`
    - `use_terrain_origins=False`, preserving grid env origins via scene `env_spacing`
    - added local visual and physics material
  - No HAPPO, reward, shield, attack preset, checkpoint, or road/vehicle route logic was changed.
- Validation:
  - `python3 -m py_compile source/my_exts/marl_platoon/tasks/platoon/config.py scripts/tools/plot_platoon_paper_figures.py scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py scripts/tools/plot_platoon_metrics.py` passed.
  - `bash -n scripts/tools/run_platoon_paper100k_hard_curriculum.sh` passed.
  - 100-total-iter staged test passed with `20/20/20/20/20`:
    - RUN_TAG: `paper100_flatterrain_test_20260629_114627`
    - stages completed: `off`, `light`, `easy`, `medium`, `hard`
    - final exit: `TEST_EXIT=0`
    - no remote `default_environment.usd` failure occurred.
    - final 5-row mean by stage:
      - off: `speed_error_abs_mean=0.0732`, `lateral=0.0080`, `centerline=0.0099`, `min_pair_gap=1.4992`, `reset_bad_ori=0.0000`
      - light: `speed=0.0759`, `lateral=0.0067`, `centerline=0.0101`, `min_gap=1.4972`, `reset_bad_ori=0.0000`
      - easy: `speed=0.0826`, `lateral=0.0292`, `centerline=0.0303`, `min_gap=1.4299`, `reset_bad_ori=0.0000`
      - medium: `speed=0.1145`, `lateral=0.0671`, `centerline=0.0609`, `min_gap=1.3879`, `reset_bad_ori=0.0000`
      - hard: `speed=0.2686`, `lateral=0.1410`, `centerline=0.0689`, `min_gap=1.2054`, `reset_bad_ori=0.0000`
- New formal 100000-iter run after the successful 100-iter test:
  - PID: `3124577`
  - RUN_TAG: `paper100k_flatterrain_curriculum_20260629_115353`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper100k_flatterrain_curriculum_20260629_115353.log`
  - package root after completion: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper100k_flatterrain_curriculum_20260629_115353_package`
  - package tar after completion: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper100k_flatterrain_curriculum_20260629_115353_package.tar.gz`
  - initial status: entered `off` stage successfully with `reset_on_bad_ori=0.0000` and `Metrics/base_velocity/error_vel_xy` around `0.04~0.05`.

2026-06-29 current training speed/status check:

- User asked whether the current training speed/training situation is normal.
- Active run:
  - PID: `3124577`
  - stage: still `off`
  - current child command: `paper100k_flatterrain_curriculum_20260629_115353_s_off`
  - stage target: `Learning iteration .../23864`
  - latest checked CSV rows: about `6778`, update range `13874 -> 20651`.
- Throughput is normal:
  - recent log shows about `880-916 steps/s`.
  - collection time about `2.13-2.21s`, learning time about `0.10-0.11s`.
  - iteration time about `2.29-2.32s`.
  - ETA for finishing `off` stage was about `2h` at the time of check.
- Vehicle speed tracking is acceptable:
  - last 200 updates:
    - `command_speed_mean ~= 0.376`
    - `leader_speed_mean ~= 0.316`
    - `platoon_speed_mean ~= 0.321`
    - `leader_speed_error_mean ~= 0.060`
    - `speed_error_abs_mean ~= 0.067`
  - latest log `Metrics/base_velocity/error_vel_xy` fluctuates around `0.05-0.09`.
- But overall no-attack training quality is not fully normal:
  - last 100 updates degraded compared with the first 100:
    - `lateral_error_abs_mean`: `0.0108 -> 0.1832`
    - `centerline_error_abs_mean`: `0.0135 -> 0.3447`
    - `gap_error_abs_mean`: `0.1690 -> 0.2694`
    - `min_pair_gap_mean`: `1.4992 -> 1.1245`
  - last 200 per-agent/pair diagnostics:
    - `centerline_robot_1_abs_mean ~= 1.196`
    - `centerline_robot_2_abs_mean ~= 0.404`
    - `lateral_pair_1_abs_mean ~= 0.431`
    - `lateral_pair_2_abs_mean ~= 0.209`
  - `shield_trigger_rate ~= 1.0` and `shield_lateral_rate ~= 0.68`, meaning the shield is active almost constantly even under no attack.
  - `termination_reset_on_bad_ori=0.0` and `collision_rate=0.0`, so it is not physically crashing, but the formation/centerline behavior is drifting.
- Conclusion:
  - Compute/training speed: normal.
  - Speed tracking: basically normal.
  - Formation/centerline data quality: concerning for a clean no-attack baseline.
  - If the goal is paper-quality curriculum data, this run should likely be paused/restarted with a shorter no-attack stage or stronger no-attack centerline/formation guard before proceeding to attack stages.

2026-06-29 perceived slowdown check:

- User asked why later training feels obviously slower.
- Current active run remains:
  - PID: `3124577`
  - RUN_TAG: `paper100k_flatterrain_curriculum_20260629_115353`
  - stage: still `off`
  - latest observed iteration: about `20717/23864`, i.e. around `6850` additional updates into the `10000`-iter off stage.
- Measured training throughput does not show a real sustained slowdown:
  - first 50 compute samples: mean `892 steps/s`, collection `2.189s`, learning `0.106s`
  - mid 50 samples: mean `920 steps/s`, collection `2.125s`, learning `0.100s`
  - last 50 samples: mean `890 steps/s`, collection `2.193s`, learning `0.107s`
  - latest log lines still show about `897-899 steps/s`, iteration time about `2.28s`
  - GPU sample: about `40%` GPU util, `3787/10240 MB` memory, `139 W`; this task is still mostly simulation/CPU-PhysX paced rather than pure GPU-saturated training.
- Why it feels slower:
  - This is now the real `100000`-iter job, not the `100`-iter sanity test; the first `off` stage alone is `10000` iter and takes roughly `6+ hours`.
  - RSL-RL iteration display continues from the checkpoint count, so `20717/23864` means progress inside the first 10000-iter resumed stage, not total 100000 pipeline progress.
  - Stage boundaries add non-training overhead: full IsaacSim restart, scene creation, checkpoint loading, per-stage plotting, and CSV/figure writing.
  - Frequent checkpoint/model saves can create short disk stalls; current `off` run directory is already about `1.3G`.
- Current conclusion:
  - Training speed itself is normal.
  - Perceived slowness is mainly due to the much longer schedule and stage-boundary/output overhead.
  - The larger concern remains data quality in late no-attack training, not throughput.

2026-06-29 no-attack stage degradation diagnosis:

- User asked why the current run gets worse before adding attack, even though the previous attack-trained run looked good.
- Current active `off` stage:
  - run: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_11-54-08_paper100k_flatterrain_curriculum_20260629_115353_s_off`
  - last-200 mean at check:
    - `speed_error_abs_mean ~= 0.0664`
    - `gap_error_abs_mean ~= 0.2699`
    - `lateral_error_abs_mean ~= 0.1840`
    - `centerline_error_abs_mean ~= 0.3460`
    - `min_pair_gap_mean ~= 1.1247`
    - `shield_trigger_rate ~= 1.0000`
    - `shield_lateral_rate ~= 0.6908`
    - `critic_grad_norm ~= 161.6`
    - `centerline_robot_1_abs_mean ~= 1.204`
    - `centerline_robot_2_abs_mean ~= 0.410`
    - `lateral_pair_1_abs_mean ~= 0.432`
- Previous good medium-attack run for comparison:
  - run: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000`
  - last-200 mean:
    - `speed_error_abs_mean ~= 0.1149`
    - `gap_error_abs_mean ~= 0.2432`
    - `lateral_error_abs_mean ~= 0.1316`
    - `centerline_error_abs_mean ~= 0.0704`
    - `min_pair_gap_mean ~= 1.3360`
    - `critic_grad_norm ~= 11.36`
    - all per-robot centerline abs means were about `0.05~0.10`.
- Interpretation:
  - The current run is not a clean no-attack evaluation; it is a long no-attack fine-tune of an already attack-trained checkpoint.
  - Turning attack off for `10000` updates changes the state distribution and removes the regularizing pressure that previously kept formation/centerline corrections active.
  - The policy is still optimizing and exploring, so it can drift into a local optimum that preserves speed/time-out while sacrificing leader centerline and pair-1 geometry.
  - The current degradation is mainly leader/first-pair drift: `centerline_robot_1` and `lateral_pair_1` dominate the bad metrics.
  - The previous attack-trained checkpoint looked better in centerline/formation because attack curriculum plus shield pressure acted like robustness regularization; it did not mean a long attack-free fine-tune would remain stable.
- Action recommendation:
  - Do not let this degraded `off` checkpoint become the base for `light/easy/medium/hard`.
  - Restart the paper curriculum from the known good checkpoint with either no `off` stage or a much shorter `off` warmup, e.g. `500~1000` iterations.
  - Add an automatic guard to the pipeline: abort or do not advance stages if last-200 `centerline_error_abs_mean > 0.15`, `lateral_error_abs_mean > 0.15`, or `min_pair_gap_mean < 1.30`.

2026-06-29 training length conclusion:

- User asked whether this means the training does not need to be so long.
- Refined conclusion:
  - A long training schedule is not automatically better once a good checkpoint already exists.
  - The current bad behavior specifically shows that a long `off/no-attack` fine-tune from a good attack-trained checkpoint is harmful; it can cause forgetting and leader/first-pair drift.
  - For paper data, `off` should be treated as a short sanity/baseline stage, not a long optimization stage.
  - Attack curriculum stages may still need longer training/evaluation samples, but they should be advanced only when guard metrics are healthy.
- Practical recommendation:
  - Use `off = 0~1000` iterations from the known good checkpoint.
  - Put most training budget into `light/easy/medium/hard`, with automatic stage guards.
  - Prefer several shorter guarded runs and checkpoint selection over one blind `100000`-iteration chain.

2026-06-29 adaptive curriculum update:

- User clarified that the pipeline should not be constrained by a fixed total iteration count; each attack stage should switch to the next stage once the training effect is good enough.
- Action taken:
  - Stopped the old fixed long run:
    - old PID: `3124577`
    - old run tag: `paper100k_flatterrain_curriculum_20260629_115353`
    - reason: the long `off` stage had already degraded centerline/formation metrics and should not feed `light/easy/medium/hard`.
  - Added new adaptive pipeline:
    - `scripts/tools/run_platoon_adaptive_hard_curriculum.sh`
- New adaptive behavior:
  - Starts from the known good checkpoint by default:
    - `START_RUN=2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000`
    - `START_CHECKPOINT=model_best.pt`
  - Stages remain: `off -> light -> easy -> medium -> hard`.
  - Each stage trains in chunks, then reads the latest `platoon_metrics.csv`.
  - If the last-window metrics satisfy the pass thresholds after the stage minimum iterations, it immediately switches to the next stage.
  - If metrics become critically bad, it stops before passing a bad checkpoint forward.
  - Default adaptive schedule:
    - `min_iters = 500 3000 5000 7000 10000`
    - `chunk_iters = 500 1500 2500 3500 5000`
    - `max_iters = 1000 15000 20000 25000 35000`
  - Pass metrics include speed error, gap error, lateral error, centerline error, min pair gap, reset-on-bad-orientation, and collision rate.
- Validation:
  - The old residual IsaacSim child process was stopped; `ps` no longer shows PIDs `3124577/3124581/3124586`.
  - `bash -n scripts/tools/run_platoon_adaptive_hard_curriculum.sh` passed.
  - `DRY_RUN=1 PLATOON_PIPELINE_LOG_ACTIVE=1 scripts/tools/run_platoon_adaptive_hard_curriculum.sh` passed and printed the expected adaptive stage schedule.
- Recommendation:
  - Use the adaptive script for the next formal paper data run.
  - Do not use `scripts/tools/run_platoon_paper100k_hard_curriculum.sh` for the next run unless a fixed-count baseline is explicitly needed.
- New adaptive run launched:
  - PID: `3127809`
  - RUN_TAG: `paper_adaptive_hard_curriculum_20260629_163210`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_hard_curriculum_20260629_163210.log`
  - package tar after completion: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_adaptive_hard_curriculum_20260629_163210_package.tar.gz`
  - current first stage at launch: `off`, chunk `1`, `500` iterations.

2026-06-29 clean adaptive restart verification:

- User requested: stop every other training process, start one new training run, then verify from a separate log tail that training is truly running.
- Cleanup:
  - Found and stopped the active adaptive run process group for `paper_adaptive_hard_curriculum_20260629_164757`.
  - Confirmed no residual `run_platoon_*curriculum.sh`, `python.sh train.py`, or IsaacSim `python3 train.py` processes remained before restart.
- Important launcher note:
  - Plain `nohup ... &` from this tool was not reliable because child IsaacSim processes could be cleaned up with the command process group.
  - The reliable launcher uses `setsid ... < /dev/null &` so the training pipeline gets its own session/process group.
- Current valid run:
  - PID: `3129013`
  - PGID/SID: `3129013`
  - RUN_TAG: `paper_adaptive_hard_curriculum_20260629_164943`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_hard_curriculum_20260629_164943.log`
  - launcher log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/paper_adaptive_hard_curriculum_20260629_164943_launcher.log`
  - run dir currently being written:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_16-49-58_paper_adaptive_hard_curriculum_20260629_164943_s_off_c1`
  - package tar after completion:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_adaptive_hard_curriculum_20260629_164943_package.tar.gz`
- Verification:
  - Process tree shows exactly one active training pipeline:
    - `bash run_platoon_adaptive_hard_curriculum.sh`
    - `python.sh scripts/reinforcement_learning/rsl_rl/train.py`
    - IsaacSim `kit/python/bin/python3 ... train.py`
  - Log tail from a separate command showed live learning output:
    - current stage: `off`, chunk `1`, `500` iterations
    - `Learning iteration 13910+ / 14364`
    - throughput about `915~935 steps/s`
    - `Episode_Termination/time_out=1.0000`
    - `Episode_Termination/reset_on_bad_ori=0.0000`
    - `Metrics/base_velocity/error_vel_xy` around `0.05~0.09`
  - GPU check showed the RTX 3080 active: about `40%` utilization, `3771/10240 MB` memory, `137 W`.
- Correct log command for the user:
  - `tail -f /home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_hard_curriculum_20260629_164943.log`
- Do not monitor the earlier aborted/stopped logs:
  - `train_paper_adaptive_hard_curriculum_20260629_163210.log`
  - `train_paper_adaptive_hard_curriculum_20260629_164649.log`
  - `train_paper_adaptive_hard_curriculum_20260629_164757.log`

2026-06-29 adaptive run status check and bug fix:

- User asked how the current training is going.
- Status of the validated adaptive run `paper_adaptive_hard_curriculum_20260629_164943`:
  - `off/chunk1` completed and passed:
    - run dir: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_16-49-58_paper_adaptive_hard_curriculum_20260629_164943_s_off_c1`
    - `speed_error_abs_mean=0.0629`
    - `gap_error_abs_mean=0.1855`
    - `lateral_error_abs_mean=0.0130`
    - `centerline_error_abs_mean=0.0180`
    - `min_pair_gap_mean=1.4986`
    - `reset_on_bad_ori=0.0000`
    - `collision_rate=0.0000`
    - assessment: `[ASSESS][off] status=PASS total=500`
  - `light/chunk1` completed:
    - run dir: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1`
    - `speed_error_abs_mean=0.0673`
    - `gap_error_abs_mean=0.2191`
    - `lateral_error_abs_mean=0.0526`
    - `centerline_error_abs_mean=0.0524`
    - `min_pair_gap_mean=1.3688`
    - `reset_on_bad_ori=0.0000`
    - `collision_rate=0.0000`
    - assessment: `[ASSESS][light] status=CONTINUE total=1500`
- Bug found:
  - The training itself did not fail.
  - The adaptive script incorrectly exited when an assessment returned `CONTINUE` code `10`, because `set -e` was restored inside `assess_stage()` before returning a nonzero status.
- Fix:
  - Patched `scripts/tools/run_platoon_adaptive_hard_curriculum.sh` so `CONTINUE` is handled by the caller and does not exit the whole script.
  - Added `START_STAGE_INDEX` support so a run can resume from a later stage instead of always starting from `off`.
  - `bash -n` and dry-run with `START_STAGE_INDEX=1` passed.
- Current valid continuation run:
  - PID: `3131497`
  - RUN_TAG: `paper_adaptive_continue_from_light_20260629_192451`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_continue_from_light_20260629_192451.log`
  - starts from:
    - `START_RUN=2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1`
    - `START_CHECKPOINT=model_final.pt`
    - `START_STAGE_INDEX=1`
  - current stage: `light`, continuing from the previous light checkpoint.
  - early live metrics after restart:
    - rows checked: `42`
    - `speed_error_abs_mean ~= 0.0758`
    - `gap_error_abs_mean ~= 0.2162`
    - `lateral_error_abs_mean ~= 0.0457`
    - `centerline_error_abs_mean ~= 0.0436`
    - `min_pair_gap_mean ~= 1.3963`
    - `reset_on_bad_ori=0.0000`
    - `collision_rate=0.0000`
  - process/GPU check:
    - one active training pipeline is running
    - GPU memory about `3787/10240 MB`
    - throughput in live log about `840~890 steps/s` at the checked moment.
- Correct log command now:
  - `tail -f /home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_continue_from_light_20260629_192451.log`

2026-06-29 adaptive training flow clarification:

- User asked what the current training flow actually is.
- Current intended pipeline:
  - stage order: `off -> light -> easy -> medium -> hard`
  - each stage runs in chunks, not one fixed monolithic iteration count.
  - after every chunk:
    - locate the just-finished run dir
    - plot metrics
    - read the last `ASSESS_WINDOW=200` rows of `platoon_metrics.csv`
    - compare speed, gap, lateral, centerline, min-gap, reset, and collision metrics against stage-specific thresholds
  - stage outcomes:
    - `PASS`: stage minimum iterations reached and metrics pass, so move to the next attack level
    - `CONTINUE`: metrics are not ready or minimum stage iterations are not reached, so continue training the same attack level from that chunk's `model_final.pt`
    - `ABORT`: metrics are critically bad, so stop before passing a bad checkpoint forward
- Default adaptive limits:
  - `off`: min `500`, chunk `500`, max `1000`
  - `light`: min `3000`, chunk `1500`, max `15000`
  - `easy`: min `5000`, chunk `2500`, max `20000`
  - `medium`: min `7000`, chunk `3500`, max `25000`
  - `hard`: min `10000`, chunk `5000`, max `35000`
- Current actual run history:
  - first valid adaptive run: `paper_adaptive_hard_curriculum_20260629_164943`
    - `off/chunk1` passed after `500` iters
    - `light/chunk1` completed `1500` iters and was healthy, but correctly returned `CONTINUE` because light needs at least `3000` iters
    - a script bug caused this `CONTINUE` return code to exit the shell; the training itself was not bad
  - continuation run: `paper_adaptive_continue_from_light_20260629_192451`
    - starts from `2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1/model_final.pt`
    - starts at `START_STAGE_INDEX=1`, so it resumes the `light` stage rather than redoing `off`
    - currently running `light` with attack enabled and `attack_level=light`
- Important nuance:
  - The continuation script does not currently count the already completed `1500` light iterations from the previous run toward its internal `total_iters`.
  - Therefore, it may train light more conservatively than strictly necessary: previous `1500` + at least `3000` continuation iters before it can pass by the internal minimum rule.
  - This is not a correctness problem because metrics are guarded, but it is slightly longer than a perfectly stitched run.

2026-06-29 paper figure data availability:

- User asked whether the overall training data will still be visible later, whether the previously requested paper figures can be drawn, and whether current data are enough.
- Data persistence:
  - Yes, every finished chunk has its own run directory, `platoon_metrics.csv`, checkpoint files, and generated per-run figures.
  - The current data are split because the adaptive script was fixed mid-run, but no data were lost.
  - Current known chunks:
    - `off`: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_16-49-58_paper_adaptive_hard_curriculum_20260629_164943_s_off_c1`
    - `light_part1`: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1`
    - `light_continue`: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Metrics coverage:
  - Each checked CSV has `148` columns.
  - Available fields include:
    - reward/training: `reward_env_mean`, reward components, `policy_loss`, `value_loss`, `actor_grad_norm`, `critic_grad_norm`
    - tracking: speed, gap, lateral, centerline, pair heading, per-pair lateral, per-robot centerline
    - attack: `attack_max_fdi_pos`, `attack_max_fdi_acc`, `attack_max_dos_rate`, `fdi_abs_mean`, `dos_rate`, observation/action attack metrics
    - physical cost: `physical_cost` and component costs
    - shield: trigger/warn/critical/lateral/centerline metrics
    - teacher/meta/local reward: `teacher_shaping_mean`, `teacher_loss`, `teacher_outer_loss`, `teacher_advantage_corr`, `local_reward_shaping_mean`
- Existing paper plotting support:
  - `scripts/tools/plot_platoon_paper_figures.py` can generate:
    - `fig_01_reward_physical.png`
    - `fig_02_tracking_errors.png`
    - `fig_03_attack_strength.png`
    - `fig_04_safety_shield.png`
    - `fig_05_teacher_meta.png`
    - `fig_06_attack_level_bars.png`
    - `fig_07_pair_centerline_bars.png`
    - `fig_08_eval_safety_tracking.png`
    - `fig_09_eval_attack_response.png`
- What current data are enough for:
  - Partial off/light training curves.
  - Reward, physical-cost, tracking-error, shield, attack-signal, and teacher/meta/local-reward plots for the completed stages.
  - Diagnosing whether off/light are healthy.
- What current data are not enough for yet:
  - Complete `off/light/easy/medium/hard` comparison: still need easy, medium, and hard stages to finish.
  - Final robustness under hard attack: need hard-stage training and final eval.
  - Full ablation/baseline figures such as no-shield, no-local-reward, no-curriculum, vanilla HAPPO: these require separate baseline runs and are not provided by the current full-method adaptive run alone.
  - Final city play snapshots/video: need the final selected checkpoint after the adaptive curriculum finishes.
- Follow-up requirement:
  - Because the run was split by a mid-run script fix, final packaging/plotting should manually stitch the known off/light chunks plus future easy/medium/hard chunks.
  - Do not rely only on the continuation run's automatic package if it omits the earlier off/light directories.

2026-06-29 remaining training plan clarification:

- User asked whether the rest of training is `light -> medium -> hard` and ends only after those are trained.
- Current script stage order is:
  - `off -> light -> easy -> medium -> hard`
- Current state:
  - `off` has already passed.
  - `light` is currently running in the continuation run:
    - PID `3131497`
    - log `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_continue_from_light_20260629_192451.log`
  - The process command confirms `env.algorithm.enable_attack=true` and `env.algorithm.attack_level=light`.
- Planned remainder:
  - finish/pass `light`
  - train/pass `easy`
  - train/pass `medium`
  - train/pass `hard`
  - after hard passes, run final eval for all attack levels and generate package/figures
- End condition:
  - The pipeline should not be considered complete merely after a fixed iteration count.
  - It is complete only after `hard` passes guard metrics and final eval/packaging finishes.
  - If a stage returns `CONTINUE`, it keeps training that same stage.
  - If a stage returns `ABORT`, it stops to avoid passing a bad checkpoint forward.

2026-06-29 command formatting preference:

- User requested that commands provided in replies should not be split across multiple lines.
- Future shell commands should be given as single-line commands whenever possible, especially `tail`, `play.py`, `train.py`, and paths under `/home/cnc/SSD_1T/xzw/IsaacLab-main`.
- If a command is too long to read comfortably, prefer one single-line copyable command plus a short explanation, rather than a backslash-continued multi-line block.

2026-06-29 short log command helper:

- User noted that even single-line long commands appear wrapped in their display.
- Added executable helper:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/watchtrain`
- Behavior:
  - reads `/home/cnc/SSD_1T/xzw/IsaacLab-main/.current_adaptive_training.log`
  - runs `tail -f` on the current adaptive training log
  - prints the followed log path first
- Test:
  - `timeout 3s /home/cnc/SSD_1T/xzw/IsaacLab-main/watchtrain | head -20` worked and showed live training log lines.
- Future user-facing command should be the short one:
  - `./watchtrain`

2026-06-29 half-hour training monitor:

- User requested checking the training process every half hour and reporting if there is a problem.
- Added monitor script:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/monitor_current_training.py`
- Monitor behavior:
  - reads current training PID from `.current_adaptive_training.pid`
  - reads current training log path from `.current_adaptive_training.log`
  - checks whether the pipeline PID is alive
  - scans recent log text for common error markers
  - locates the latest adaptive run dir and reads `platoon_metrics.csv`
  - checks key metrics: speed, gap, lateral, centerline, min gap, reset, collision, critic grad
  - appends every result to `train_monitor_current.log`
  - appends abnormal results to `train_monitor_alerts.log`
- Started background monitor loop:
  - PID: `3132805`
  - interval: `1800` seconds
  - loop log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_monitor_loop.log`
  - PID pointer: `/home/cnc/SSD_1T/xzw/IsaacLab-main/.current_training_monitor.pid`
- Immediate monitor result:
  - `status=OK`
  - training PID `3131497` alive
  - current run dir: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
  - GPU active around `30~38%`, memory around `3780/10240 MB`
  - last-window metrics around:
    - `speed_error_abs_mean ~= 0.073`
    - `gap_error_abs_mean ~= 0.246`
    - `lateral_error_abs_mean ~= 0.091`
    - `centerline_error_abs_mean ~= 0.088`
    - `min_pair_gap_mean ~= 1.257`
    - `reset_on_bad_ori = 0`
    - `collision_rate = 0`
- Short commands if needed:
  - `tail -f train_monitor_current.log`
  - `cat train_monitor_alerts.log`

2026-06-29 half-hour monitor follow-up check:

- A manual follow-up check was run after the monitor loop was started.
- Background monitor loop is alive:
  - monitor PID: `3132805`
  - training PID: `3131497`
  - current training stage remains `light`
  - current log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_continue_from_light_20260629_192451.log`
- Latest monitor result:
  - timestamp: `2026-06-29 19:45:22`
  - `status=OK`
  - GPU active: about `38%`, `3788/10240 MB`, `133.58 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0730`
    - `gap_error_abs_mean ~= 0.2469`
    - `lateral_error_abs_mean ~= 0.0929`
    - `centerline_error_abs_mean ~= 0.0895`
    - `min_pair_gap_mean ~= 1.2524`
    - `critic_grad_norm ~= 12.24`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- No alert was written to `train_monitor_alerts.log`.

2026-06-29 half-hour monitor continuation check:

- A further manual continuation check was run while waiting for the next automatic 30-minute monitor tick.
- Background monitor loop is still alive:
  - monitor PID: `3132805`
  - training PID: `3131497`
  - stage still shown as `light` in the active command
- Latest monitor result:
  - timestamp: `2026-06-29 19:46:19`
  - `status=OK`
  - GPU active: about `37%`, `3778/10240 MB`, `134.73 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0734`
    - `gap_error_abs_mean ~= 0.2469`
    - `lateral_error_abs_mean ~= 0.0941`
    - `centerline_error_abs_mean ~= 0.0903`
    - `min_pair_gap_mean ~= 1.2506`
    - `critic_grad_norm ~= 12.21`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- `train_monitor_alerts.log` still has no alert entries.

2026-06-29 half-hour monitor continuation check 2:

- Another manual continuation check was run before the next automatic 30-minute monitor tick.
- Background monitor loop is still alive:
  - monitor PID: `3132805`
  - training PID: `3131497`
  - active stage still `light`
- Latest monitor result:
  - timestamp: `2026-06-29 19:47:06`
  - `status=OK`
  - GPU active: about `28%`, `3790/10240 MB`, `133.57 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0728`
    - `gap_error_abs_mean ~= 0.2468`
    - `lateral_error_abs_mean ~= 0.0944`
    - `centerline_error_abs_mean ~= 0.0905`
    - `min_pair_gap_mean ~= 1.2498`
    - `critic_grad_norm ~= 12.12`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- No alert entries were produced.

2026-06-29 half-hour monitor continuation check 5:

- Another manual continuation check was run.
- Background monitor loop is still alive:
  - monitor PID: `3132805`
  - training PID: `3131497`
  - active stage still `light`
- Latest monitor result:
  - timestamp: `2026-06-29 19:49:25`
  - `status=OK`
  - GPU active: about `39%`, `3778/10240 MB`, `135.60 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0742`
    - `gap_error_abs_mean ~= 0.2473`
    - `lateral_error_abs_mean ~= 0.0961`
    - `centerline_error_abs_mean ~= 0.0914`
    - `min_pair_gap_mean ~= 1.2436`
    - `critic_grad_norm ~= 12.21`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- No alert entries were produced.

2026-06-29 half-hour monitor continuation check 4:

- Another manual continuation check was run.
- Background monitor loop is still alive:
  - monitor PID: `3132805`
  - training PID: `3131497`
  - active stage still `light`
- Latest monitor result:
  - timestamp: `2026-06-29 19:48:36`
  - `status=OK`
  - GPU active: about `27%`, `3784/10240 MB`, `135.65 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0739`
    - `gap_error_abs_mean ~= 0.2470`
    - `lateral_error_abs_mean ~= 0.0958`
    - `centerline_error_abs_mean ~= 0.0914`
    - `min_pair_gap_mean ~= 1.2454`
    - `critic_grad_norm ~= 12.07`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- No alert entries were produced.

2026-06-29 half-hour monitor continuation check 3:

- Another manual continuation check was run.
- Background monitor loop is still alive:
  - monitor PID: `3132805`
  - training PID: `3131497`
  - active stage still `light`
- Latest monitor result:
  - timestamp: `2026-06-29 19:47:53`
  - `status=OK`
  - GPU active: about `35%`, `3778/10240 MB`, `138.91 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0735`
    - `gap_error_abs_mean ~= 0.2470`
    - `lateral_error_abs_mean ~= 0.0951`
    - `centerline_error_abs_mean ~= 0.0910`
    - `min_pair_gap_mean ~= 1.2469`
    - `critic_grad_norm ~= 12.17`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- No alert entries were produced.

2026-06-29 half-hour monitor continuation check 6:

- Another immediate monitor check was run after the user requested half-hourly checks.
- Background half-hour monitor loop is alive:
  - monitor PID: `3132805`
  - command: `monitor_current_training.py` every `1800s`
- Adaptive training process is alive:
  - training PID: `3131497`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage still appears to be `light`
- Latest monitor result:
  - timestamp: `2026-06-29 19:51:23`
  - `status=OK`
  - GPU active: about `39%`, `3778/10240 MB`, `132.30 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0763`
    - `gap_error_abs_mean ~= 0.2478`
    - `lateral_error_abs_mean ~= 0.0981`
    - `centerline_error_abs_mean ~= 0.0926`
    - `min_pair_gap_mean ~= 1.2350`
    - `critic_grad_norm ~= 12.11`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- No alert log exists yet, which means no abnormal monitor event has been recorded so far.
- Note: lateral/centerline error is drifting upward slightly within the light stage, and `min_pair_gap_mean` is slowly decreasing, but all values are still inside the current safety thresholds. Continue watching before intervening.

2026-06-29 half-hour monitor continuation check 7:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `8m19s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:52:21`
  - `status=OK`
  - GPU active: about `39%`, `3774/10240 MB`, `138.30 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0761`
    - `gap_error_abs_mean ~= 0.2476`
    - `lateral_error_abs_mean ~= 0.0983`
    - `centerline_error_abs_mean ~= 0.0928`
    - `min_pair_gap_mean ~= 1.2339`
    - `critic_grad_norm ~= 11.91`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16551/17362`
  - elapsed time around `00:26:41`
  - ETA around `00:31:22`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: training is still healthy. The light-stage lateral/centerline errors remain acceptable but continue a slow upward drift; keep monitoring until the chunk assessment decides whether to pass or continue.

2026-06-29 half-hour monitor continuation check 8:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `9m00s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:53:00`
  - `status=OK`
  - GPU active: about `39%`, `3778/10240 MB`, `145.48 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0766`
    - `gap_error_abs_mean ~= 0.2478`
    - `lateral_error_abs_mean ~= 0.0984`
    - `centerline_error_abs_mean ~= 0.0930`
    - `min_pair_gap_mean ~= 1.2325`
    - `critic_grad_norm ~= 12.04`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16571/17362`
  - elapsed time around `00:27:27`
  - ETA around `00:30:35`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: training remains healthy. The slow upward drift in lateral/centerline error and decreasing `min_pair_gap_mean` continue, but values are still well within current guard thresholds; no intervention yet.

2026-06-29 half-hour monitor continuation check 9:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `9m48s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:53:48`
  - `status=OK`
  - GPU active: about `38%`, `3776/10240 MB`, `144.37 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0763`
    - `gap_error_abs_mean ~= 0.2490`
    - `lateral_error_abs_mean ~= 0.0996`
    - `centerline_error_abs_mean ~= 0.0942`
    - `min_pair_gap_mean ~= 1.2270`
    - `critic_grad_norm ~= 11.91`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16591/17362`
  - elapsed time around `00:28:13`
  - ETA around `00:29:48`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: still healthy. The drift trend remains visible, but current lateral/centerline/gap values are not near the intervention thresholds; continue watching until the chunk assessment is printed.

2026-06-29 half-hour monitor continuation check 10:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `10m31s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:54:32`
  - `status=OK`
  - GPU active: about `40%`, `3778/10240 MB`, `140.05 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0764`
    - `gap_error_abs_mean ~= 0.2501`
    - `lateral_error_abs_mean ~= 0.1003`
    - `centerline_error_abs_mean ~= 0.0954`
    - `min_pair_gap_mean ~= 1.2238`
    - `critic_grad_norm ~= 12.00`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16611/17362`
  - elapsed time around `00:28:58`
  - ETA around `00:29:01`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: training is still healthy. Lateral and centerline errors have crossed about `0.10` and `0.095` respectively, but remain below guard thresholds; `min_pair_gap_mean` is still safely above the hard alert threshold. Keep monitoring until this `light` chunk assessment completes.

2026-06-29 half-hour monitor continuation check 11:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `11m12s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:55:13`
  - `status=OK`
  - GPU active: about `36%`, `3778/10240 MB`, `138.57 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0774`
    - `gap_error_abs_mean ~= 0.2516`
    - `lateral_error_abs_mean ~= 0.1018`
    - `centerline_error_abs_mean ~= 0.0973`
    - `min_pair_gap_mean ~= 1.2180`
    - `critic_grad_norm ~= 12.08`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16627/17362`
  - elapsed time around `00:29:35`
  - ETA around `00:28:23`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: training remains operational and stable by hard safety checks. The light-stage drift trend continues (`lateral_error_abs_mean` now about `0.102`, `centerline_error_abs_mean` about `0.097`, `min_pair_gap_mean` about `1.218`), so the next chunk assessment should be used to decide pass vs. continue rather than manually intervening now.

2026-06-29 half-hour monitor continuation check 12:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `11m52s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:55:54`
  - `status=OK`
  - GPU active: about `25%`, `3778/10240 MB`, `138.73 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0779`
    - `gap_error_abs_mean ~= 0.2525`
    - `lateral_error_abs_mean ~= 0.1030`
    - `centerline_error_abs_mean ~= 0.0987`
    - `min_pair_gap_mean ~= 1.2149`
    - `critic_grad_norm ~= 11.98`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16647/17362`
  - elapsed time around `00:30:20`
  - ETA around `00:27:36`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: training remains healthy by hard checks. The light-stage drift continues gradually, with `lateral_error_abs_mean` near `0.103` and `centerline_error_abs_mean` near `0.099`; still no reset/collision. Continue monitoring until the chunk assessment prints pass/continue/fail.

2026-06-29 half-hour monitor continuation check 13:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `12m44s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:56:44`
  - `status=OK`
  - GPU active: about `39%`, `3776/10240 MB`, `134.81 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0773`
    - `gap_error_abs_mean ~= 0.2537`
    - `lateral_error_abs_mean ~= 0.1041`
    - `centerline_error_abs_mean ~= 0.1004`
    - `min_pair_gap_mean ~= 1.2112`
    - `critic_grad_norm ~= 12.05`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16667/17362`
  - elapsed time around `00:31:06`
  - ETA around `00:26:49`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: still no hard failure. The slow degradation trend is now more visible (`centerline_error_abs_mean` just over `0.10`, `min_pair_gap_mean` about `1.21`) but still below alert thresholds. Wait for the automatic chunk assessment before changing training.

2026-06-29 half-hour monitor continuation check 14:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `13m25s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:57:27`
  - `status=OK`
  - GPU active: about `29%`, `3776/10240 MB`, `137.35 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0764`
    - `gap_error_abs_mean ~= 0.2551`
    - `lateral_error_abs_mean ~= 0.1059`
    - `centerline_error_abs_mean ~= 0.1026`
    - `min_pair_gap_mean ~= 1.2059`
    - `critic_grad_norm ~= 11.93`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16687/17362`
  - elapsed time around `00:31:52`
  - ETA around `00:26:03`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: the run is still alive and safe by the current hard thresholds, but the light-stage quality drift continues (`lateral_error_abs_mean` about `0.106`, `centerline_error_abs_mean` about `0.103`, `min_pair_gap_mean` about `1.206`). Continue monitoring and let the adaptive stage assessment decide whether to continue or reject this chunk.

2026-06-29 half-hour monitor continuation check 15:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `15m29s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 19:58:11`
  - `status=OK`
  - GPU active: about `36%`, `3780/10240 MB`, `140.71 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0757`
    - `gap_error_abs_mean ~= 0.2560`
    - `lateral_error_abs_mean ~= 0.1069`
    - `centerline_error_abs_mean ~= 0.1042`
    - `min_pair_gap_mean ~= 1.2042`
    - `critic_grad_norm ~= 12.01`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16739/17362`
  - elapsed time around `00:33:52`
  - ETA around `00:24:01`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: no process or hard safety failure. However, this `light` continuation chunk is steadily degrading relative to its start; `centerline_error_abs_mean` increased from about `0.0875` to `0.1042`, `lateral_error_abs_mean` from about `0.0906` to `0.1069`, and `min_pair_gap_mean` fell from about `1.2574` to `1.2042`. Continue watching until the adaptive stage assessment decides whether this chunk should be accepted or continued/rejected.

2026-06-29 half-hour monitor continuation check 16:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `16m18s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 20:00:19`
  - `status=OK`
  - GPU active: about `36%`, `3772/10240 MB`, `140.08 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0770`
    - `gap_error_abs_mean ~= 0.2591`
    - `lateral_error_abs_mean ~= 0.1099`
    - `centerline_error_abs_mean ~= 0.1083`
    - `min_pair_gap_mean ~= 1.1963`
    - `critic_grad_norm ~= 12.29`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16759/17362`
  - elapsed time around `00:34:38`
  - ETA around `00:23:15`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- No alert log exists yet, so no abnormal monitor event has been recorded.
- Interpretation: still no hard failure, but the `light` continuation chunk is now clearly worse than its start. Since the adaptive script has not yet reached chunk assessment, do not interrupt manually, but if the assessment accepts this chunk despite these trends, review the pass thresholds before using the resulting checkpoint as a paper-quality stage output.

2026-06-29 half-hour monitor continuation check 17:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `17m04s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 20:01:04`
  - `status=OK` by hard thresholds
  - GPU active: about `26%`, `3784/10240 MB`, `137.82 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0774`
    - `gap_error_abs_mean ~= 0.2601`
    - `lateral_error_abs_mean ~= 0.1117`
    - `centerline_error_abs_mean ~= 0.1105`
    - `min_pair_gap_mean ~= 1.1916`
    - `critic_grad_norm ~= 12.24`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16779/17362`
  - elapsed time around `00:35:24`
  - ETA around `00:22:29`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- The pipeline has not reached the automatic chunk assessment yet:
  - current active launch remains `stage=light chunk=1 chunk_iters=1500`
  - no `PASS`, `CONTINUE`, or `ABORT` assessment line has appeared yet for this chunk.
- A quality-drift warning was written to `train_monitor_alerts.log` because the deterioration is now persistent even though hard safety checks have not failed:
  - `centerline_error_abs_mean` rose from about `0.0875` to `0.1105`
  - `lateral_error_abs_mean` rose from about `0.0906` to `0.1117`
  - `min_pair_gap_mean` fell from about `1.2574` to `1.1916`
- Interpretation: no process failure and no reset/collision yet, but this chunk should be treated as quality-degrading. If the adaptive script accepts it, the pass thresholds should be reviewed before using the resulting checkpoint as a paper-quality result.

2026-06-29 half-hour monitor continuation check 18:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `18m11s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 20:02:10`
  - `status=OK` by hard thresholds
  - GPU active: about `25%`, `3776/10240 MB`, `137.39 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0765`
    - `gap_error_abs_mean ~= 0.2616`
    - `lateral_error_abs_mean ~= 0.1145`
    - `centerline_error_abs_mean ~= 0.1134`
    - `min_pair_gap_mean ~= 1.1858`
    - `critic_grad_norm ~= 12.22`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16807/17362`
  - elapsed time around `00:36:30`
  - ETA around `00:21:25`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- The pipeline still has not reached the automatic chunk assessment:
  - active launch remains `stage=light chunk=1 chunk_iters=1500`
  - no `PASS`, `CONTINUE`, or `ABORT` assessment line has appeared yet.
- A second quality-drift warning was appended to `train_monitor_alerts.log` because deterioration continued:
  - `centerline_error_abs_mean` reached `0.1134`
  - `lateral_error_abs_mean` reached `0.1145`
  - `min_pair_gap_mean` fell to `1.1858`
- Interpretation: hard safety checks still pass, but this `light` chunk is increasingly poor. If it is accepted automatically, the adaptive acceptance thresholds are probably too loose for paper-quality training data.

2026-06-29 half-hour monitor continuation check 19:

- Manual monitor check was run again for the active adaptive curriculum training.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `19m22s`
- Adaptive training process remains alive:
  - training PID: `3131497`
  - active script: `run_platoon_adaptive_hard_curriculum.sh`
  - active run: `paper_adaptive_continue_from_light_20260629_192451`
  - active stage/chunk: `light`, run dir `2026-06-29_19-25-06_paper_adaptive_continue_from_light_20260629_192451_s_light_c1`
- Latest monitor result:
  - timestamp: `2026-06-29 20:03:21`
  - `status=OK` by hard thresholds
  - GPU active: about `41%`, `3780/10240 MB`, `145.15 W`
  - last-window metrics:
    - `speed_error_abs_mean ~= 0.0760`
    - `gap_error_abs_mean ~= 0.2638`
    - `lateral_error_abs_mean ~= 0.1181`
    - `centerline_error_abs_mean ~= 0.1173`
    - `min_pair_gap_mean ~= 1.1769`
    - `critic_grad_norm ~= 12.82`
    - `reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Training log is still advancing:
  - latest observed iteration around `16839/17362`
  - elapsed time around `00:37:45`
  - ETA around `00:20:11`
  - reported `time_out = 1.0000`, `reset_on_bad_ori = 0.0000`
- The pipeline still has not reached the automatic chunk assessment:
  - active launch remains `stage=light chunk=1 chunk_iters=1500`
  - no `PASS`, `CONTINUE`, or `ABORT` assessment line has appeared yet.
- A third quality-drift warning was appended to `train_monitor_alerts.log`:
  - `centerline_error_abs_mean` reached `0.1173`
  - `lateral_error_abs_mean` reached `0.1181`
  - `min_pair_gap_mean` fell to `1.1769`
- Interpretation: process and hard safety are still OK, but this `light` chunk is no longer clean progress. Unless the adaptive assessment rejects/retries it, the resulting checkpoint should be treated as suspect.

2026-06-29 half-hour monitor continuation check 20:

- User asked to continue checking this training every half hour; an immediate check was run.
- Background half-hour monitor loop remains alive:
  - monitor PID: `3132805`
  - elapsed runtime at check: about `2h23m35s`
- Adaptive training process is no longer alive:
  - expected training PID: `3131497`
  - `ps` no longer finds this PID.
- The adaptive pipeline reached automatic assessment and stopped:
  - active run tag: `paper_adaptive_continue_from_light_20260629_192451`
  - `light/chunk1` completed and was assessed as `CONTINUE`:
    - `status=CONTINUE total=1500 speed=0.0836 gap=0.2898 lat=0.1795 center=0.1945 min_gap=1.0942 reset=0.0000 collision=0.0000`
  - `light/chunk2` completed, then was assessed as `ABORT`:
    - `status=ABORT total=3000 speed=0.1047 gap=0.2688 lat=0.2447 center=0.2570 min_gap=1.1183 reset=1.0000 collision=0.0000`
  - pipeline printed:
    - `[ERROR] stage light hit critical bad metrics; stopping before passing a bad checkpoint forward.`
    - `[PIPELINE] exit status=20 at line=1`
- Latest monitor state is an alert:
  - timestamp: `2026-06-29 22:07:34`
  - `status=ALERT`
  - run dir: `2026-06-29_20-23-58_paper_adaptive_continue_from_light_20260629_192451_s_light_c2`
  - metrics:
    - `speed_error_abs_mean ~= 0.1047`
    - `gap_error_abs_mean ~= 0.2688`
    - `lateral_error_abs_mean ~= 0.2447`
    - `centerline_error_abs_mean ~= 0.2570`
    - `min_pair_gap_mean ~= 1.1183`
    - `critic_grad_norm ~= 97.52` at the final monitor window; higher values were seen during the run
    - `reset_on_bad_ori = 1.0000`
    - `collision_rate = 0.0000`
- A new `stage_abort` alert was appended to `train_monitor_alerts.log`.
- Interpretation:
  - The monitor did its job: quality drift became a real failure in `light/chunk2`.
  - The adaptive script correctly stopped before forwarding a bad checkpoint.
  - Do not continue from `2026-06-29_20-23-58_paper_adaptive_continue_from_light_20260629_192451_s_light_c2/model_final.pt`.
  - The last usable point for analysis/restart is before the failed continuation, especially `2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1/model_final.pt` or the earlier stable baseline, depending on the next training plan.

2026-06-29 failure cause analysis for adaptive light continuation:

- The run failed because `light/chunk2` triggered the adaptive script's critical bad-metrics gate, not because Isaac Sim crashed or the shell command failed.
- `light/chunk1` had already degraded and was not good enough:
  - assessment: `CONTINUE total=1500 speed=0.0836 gap=0.2898 lat=0.1795 center=0.1945 min_gap=1.0942 reset=0.0000 collision=0.0000`
  - for `light`, pass thresholds are approximately: `speed<=0.11`, `gap<=0.22`, `lateral<=0.08`, `centerline<=0.08`, `min_gap>=1.35`, `reset<=0.005`, `collision<=0.0001`.
  - so chunk1 already failed gap/lateral/centerline/min_gap, even though reset was still zero.
- Continuing from the degraded chunk1 checkpoint made chunk2 much worse:
  - final assessment: `ABORT total=3000 speed=0.1047 gap=0.2688 lat=0.2447 center=0.2570 min_gap=1.1183 reset=1.0000 collision=0.0000`
  - `reset_on_bad_ori=1.0000` exceeded the critical abort gate (`>0.05`) by a wide margin.
  - episode length dropped to about `527`, `time_out=0.0000`, and `reset_on_bad_ori=1.0000`, which means episodes were ending by bad orientation rather than successful timeout.
- The mechanism looks like policy/critic instability after over-continuing light attack:
  - lateral and centerline errors rose steadily from about `0.09` to `0.25+`.
  - `min_pair_gap_mean` fell from about `1.26` toward `1.12`.
  - `critic_grad_norm` became large (`~97` final window, with higher windows observed around `162+`), while value losses near the end reached tens.
  - `Mean action noise std` rose to about `12.00`, and shield/lateral correction rewards became much more negative, indicating the policy was relying heavily on correction while drifting.
- Practical conclusion:
  - `light/chunk2/model_final.pt` is invalid and should not be used.
  - `light/chunk1/model_final.pt` is also not a clean pass; it only produced `CONTINUE`, so it should not be treated as a stable final light result.
  - restart should use the earlier stable checkpoint before this continuation, then either shorten light training, reduce learning/noise aggressiveness, or tighten early stop criteria so a degraded `CONTINUE` chunk is not used as the next resume point.

2026-06-29 why current light failed despite previous light/easy success:

- Main cause: yes, the `light` stage was effectively trained too much and was continued from a degrading checkpoint.
- Previous successful light/easy runs were short fine-tunes/sanity stages:
  - examples include `platoon5_city_lightattack_from_model700_sanity100`, `platoon5_city_lightattack_from_model700_ft400`, `platoon5_city_easyattack_from_light_sanity100`, and `platoon5_city_easyattack_from_light_ft400`.
  - those runs were much shorter and were not forced through thousands of extra light iterations after quality started degrading.
- The failed adaptive continuation had a bookkeeping mismatch:
  - it started from `2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1/model_final.pt`, which was already a light-trained checkpoint.
  - the continuation script used `START_STAGE_INDEX=1`, but internally began `light` again with `total_before=0`.
  - so the continuation did not count the previous light exposure; it ran another `light/chunk1` and then `light/chunk2`.
  - actual light exposure was therefore higher than the adaptive stage counter reported.
- The first continuation chunk already showed the failure direction:
  - `light/chunk1` assessment was `CONTINUE`, not pass:
    - `gap=0.2898`, `lat=0.1795`, `center=0.1945`, `min_gap=1.0942`
    - these were far outside the light pass thresholds (`gap<=0.22`, `lat<=0.08`, `center<=0.08`, `min_gap>=1.35`).
  - continuing from this degraded `model_final.pt` pushed chunk2 into bad orientation resets.
- The failure is therefore not evidence that light attack is inherently impossible. It means:
  - the policy was already drifting before chunk2;
  - `model_final.pt` was a bad resume point;
  - the adaptive stage should have rolled back or stopped earlier instead of continuing from a degraded checkpoint;
  - light should be a short validation/fine-tune stage in this pipeline, not a multi-thousand-iteration stage after a strong medium-trained baseline.
- Proposed next adjustment:
  - restart from a stable pre-failure checkpoint such as `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt` or another verified play-good checkpoint;
  - if using an already light-trained checkpoint, skip light and move to easy/medium validation instead of rerunning light;
  - reduce adaptive `light` to short chunks such as `300-500` iterations with low max, or treat it as a sanity/evaluation stage;
  - use checkpoint selection/rollback so a `CONTINUE` chunk that worsens lateral/centerline/min_gap does not become the next resume point.

2026-06-29 why HAPPO/meta-learning can still get worse:

- HAPPO/PPO-style updates are not monotonic-improvement algorithms in practice. PPO clipping limits policy update size, but it does not guarantee that evaluation metrics such as centerline error, lateral error, min-gap, or bad-orientation reset will improve after every resume/chunk.
- Meta-learning/adaptation also does not guarantee "cannot get worse"; it can accelerate adaptation, but if the resume point is already biased or the stage is over-trained, it can adapt toward a locally bad behavior.
- In this failed run, the degradation mechanism is visible in the logs:
  - `lateral_error_abs_mean` and `centerline_error_abs_mean` rose steadily.
  - `min_pair_gap_mean` fell steadily.
  - `critic_grad_norm` and value loss grew.
  - `Mean action noise std` increased toward about `12.0`.
  - `reset_on_bad_ori` finally reached `1.0`, meaning the learned behavior became physically unstable.
- The likely practical causes are:
  - continuing too long in `light` after already having a light-trained checkpoint;
  - resuming from `model_final.pt` rather than a validated best/stable checkpoint;
  - using a degraded `CONTINUE` chunk as the next resume source;
  - multi-agent non-stationarity, where all agents keep adapting and can move away from a coordinated equilibrium;
  - reward/shield mismatch, where reward can still look partially acceptable while shield and physical resets reveal unstable behavior.
- Correct expectation: HAPPO + meta-learning can improve sample efficiency and robustness when guarded by validation, early stopping, rollback, and checkpoint selection. It should not be expected to improve monotonically under indefinite continued training.

2026-06-29 adaptive curriculum validation-gated update:

- The adaptive hard-curriculum script was changed so each stage is now controlled by validation metrics rather than fixed-duration continuation.
- Modified file:
  - `scripts/tools/run_platoon_adaptive_hard_curriculum.sh`
- New behavior:
  - `ALLOW_EARLY_PASS=1` lets a stage pass as soon as its validation metrics satisfy that stage's thresholds, even if `min_stage_iters` has not been reached.
  - `RESUME_STAGE_PROGRESS=1` reads an incoming run's `adaptive_assessment.txt` and reuses its `total_stage_iters`, avoiding the previous bookkeeping error where an already light-trained checkpoint restarted light from `total_before=0`.
  - The script seeds each stage with the incoming checkpoint's assessment when the assessment stage matches the active stage.
  - The script computes a stage validation score from speed error, gap error, lateral error, centerline error, min pair gap, reset rate, and collision rate.
  - The script tracks the best checkpoint inside each stage and forwards the best passing checkpoint to the next stage, not blindly the latest `model_final.pt`.
  - `VALIDATION_ROLLBACK=1` with `VALIDATION_WORSE_TOL=0.02` stops the stage when the current chunk is worse than the previous best by more than 2%, preventing a degraded `CONTINUE` checkpoint from being used as the next resume source.
- Validation checks performed:
  - `bash -n scripts/tools/run_platoon_adaptive_hard_curriculum.sh` passed.
  - `DRY_RUN=1 RUN_TAG=adaptive_validation_logic_dryrun scripts/tools/run_platoon_adaptive_hard_curriculum.sh` passed and expanded the expected off/light/easy/medium/hard schedule.
  - Early-pass test with `START_RUN=2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1`, `START_CHECKPOINT=model_final.pt`, `START_STAGE_INDEX=1`, `STAGE_LIMIT=2`, `SKIP_PACKAGE=1` accepted the light stage immediately without launching training:
    - seeded score `3.907079699`
    - accepted `2026-06-29_17-09-12_paper_adaptive_hard_curriculum_20260629_164943_s_light_c1/model_final.pt`
- Practical conclusion:
  - The previous failure mode is now guarded: if light already satisfies validation, the script can move on instead of forcing more light training; if a later chunk worsens, the script stops before feeding the bad checkpoint forward.
  - The next real run should restart from a stable checkpoint and let the validation gate decide stage transitions.

2026-06-29 22:29 current process check:

- Checked active processes for `train.py`, `play.py`, `eval_happo_platoon.py`, `python.sh scripts/reinforcement_learning/rsl_rl/train.py`, and `run_platoon_adaptive_hard_curriculum`.
- No active training, play, or evaluation process is running.
- Stale training PID file:
  - `.current_adaptive_training.pid` still contains `3131497`, but `ps -p 3131497` returns no live process.
- Still-running non-training processes:
  - `3132621`: `tail -f /home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_adaptive_continue_from_light_20260629_192451.log`
  - `3132805`: half-hour monitor loop calling `scripts/tools/monitor_current_training.py`
  - `3136723`: `sleep 1800` child of the monitor loop
- Practical conclusion:
  - There is currently no training consuming GPU for the platoon experiment.
  - The remaining monitor will continue reporting that the previous adaptive pipeline PID is dead unless it is stopped or a new training PID/log is registered.

2026-06-29 22:35 validation-gated hard curriculum restarted:

- Old stale monitor/log-tail processes were stopped before launching the new run.
- Ordinary `nohup ... &` did not persist in the current execution environment: even a short `nohup sleep 120` test exited immediately.
- `setsid ... &` was verified to persist, so the real training and monitor were launched with `setsid`.
- New training:
  - `RUN_TAG=paper_validation_gated_hard_curriculum_20260629_223342`
  - PID: `3137420`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_validation_gated_hard_curriculum_20260629_223342.log`
  - start checkpoint: `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - stage plan: `off -> light -> easy -> medium -> hard`
  - validation gates enabled:
    - `ALLOW_EARLY_PASS=1`
    - `RESUME_STAGE_PROGRESS=1`
    - `VALIDATION_ROLLBACK=1`
    - `VALIDATION_WORSE_TOL=0.02`
- New half-hour monitor:
  - PID: `3137936`
  - loop: `scripts/tools/monitor_current_training.py` every 1800 seconds
  - pointers updated:
    - `.current_adaptive_training.pid`
    - `.current_adaptive_training.log`
    - `.current_adaptive_training.tag`
- Initial verification:
  - The training entered IsaacLab successfully and began `off/chunk1`.
  - Example early log:
    - learning iteration around `13873/14364`
    - speed about `926-933 steps/s`
    - `reset_on_bad_ori=0.0000`
  - Immediate monitor status was `OK`.
  - New current run directory:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_22-33-56_paper_validation_gated_hard_curriculum_20260629_223342_s_off_c1`
  - Early monitor metrics:
    - `centerline_error_abs_mean ~= 0.0069`
    - `lateral_error_abs_mean ~= 0.0060`
    - `gap_error_abs_mean ~= 0.1701`
    - `min_pair_gap_mean ~= 1.4995`
    - `speed_error_abs_mean ~= 0.0684`
    - `termination_reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Practical conclusion:
  - The new validation-gated training is now genuinely running and the monitor is watching the new PID/log, not the previous failed continuation.

2026-06-29 23:20 monitor interval confirmation:

- Current monitor process:
  - PID `3137936`
  - command: `bash -c while true; do /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/monitor_current_training.py || true; sleep 1800; done`
- Therefore automatic checks run every `1800` seconds, i.e. every 30 minutes.
- Latest automatic monitor entry at `2026-06-29 23:04:46` was `status=OK` for PID `3137420`.
- Latest monitored stage/run:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_22-53-14_paper_validation_gated_hard_curriculum_20260629_223342_s_light_c1`
  - metrics were still healthy:
    - `centerline_error_abs_mean ~= 0.0194`
    - `lateral_error_abs_mean ~= 0.0151`
    - `gap_error_abs_mean ~= 0.1680`
    - `min_pair_gap_mean ~= 1.4975`
    - `speed_error_abs_mean ~= 0.0612`
    - `termination_reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`

2026-06-29 23:22 monitor interval changed to 5 minutes:

- User requested changing the automatic training monitor from every 30 minutes to every 5 minutes.
- Old monitor process:
  - PID `3137936`
  - command used `sleep 1800`
  - stopped with its process group.
- New monitor process:
  - PID `3139074`
  - command: `bash -c while true; do /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/monitor_current_training.py || true; sleep 300; done`
  - detection interval is now `300` seconds, i.e. 5 minutes.
- Immediate check after changing the interval:
  - status `OK`
  - training PID `3137420`
  - current run directory:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_22-53-14_paper_validation_gated_hard_curriculum_20260629_223342_s_light_c1`
  - latest metrics around `2026-06-29 23:22:42`:
    - `centerline_error_abs_mean ~= 0.0283`
    - `lateral_error_abs_mean ~= 0.0215`
    - `gap_error_abs_mean ~= 0.1788`
    - `min_pair_gap_mean ~= 1.4890`
    - `speed_error_abs_mean ~= 0.0668`
    - `termination_reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`

2026-06-29 23:59 five-minute monitor visibility check:

- The monitor process is still alive and uses the requested 5-minute interval:
  - PID `3139074`
  - command: `bash -c while true; do /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/monitor_current_training.py || true; sleep 300; done`
  - current child sleep: `sleep 300`
- `train_monitor_current.log` confirms 5-minute writes:
  - `23:27:35`
  - `23:32:35`
  - `23:37:35`
  - `23:42:35`
  - `23:47:35`
  - `23:52:35`
  - `23:57:35`
- The reason this may not be visible while using `./watchtrain` is that `./watchtrain` follows the training log, not the monitor log.
- Added helper command:
  - `./watchmonitor`
  - It follows `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_monitor_current.log`.
- Current training state:
  - training PID `3137420`
  - current stage has moved from `light_c1` to `easy_c1`
  - current easy run directory:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_23-50-33_paper_validation_gated_hard_curriculum_20260629_223342_s_easy_c1`
  - latest immediate monitor at `2026-06-29 23:58:47` was `status=OK`.
  - latest easy metrics:
    - `centerline_error_abs_mean ~= 0.0799`
    - `lateral_error_abs_mean ~= 0.1107`
    - `gap_error_abs_mean ~= 0.2547`
    - `min_pair_gap_mean ~= 1.2737`
    - `speed_error_abs_mean ~= 0.0840`
    - `termination_reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Practical conclusion:
  - The 5-minute monitor is working; monitor output should be viewed with `./watchmonitor`, while training iteration output should be viewed with `./watchtrain`.

2026-06-30 03:08 validation-gated run status:

- Current validation-gated training is no longer running:
  - `.current_adaptive_training.pid` contains `3137420`
  - `ps -p 3137420` returns no live process.
- The monitor process is still alive and will continue reporting alerts until stopped or pointed at a new run:
  - monitor PID `3139074`
  - interval `sleep 300`
- Pipeline:
  - `RUN_TAG=paper_validation_gated_hard_curriculum_20260629_223342`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_validation_gated_hard_curriculum_20260629_223342.log`
- Stage results:
  - `off/chunk1`: PASS
    - run: `2026-06-29_22-33-56_paper_validation_gated_hard_curriculum_20260629_223342_s_off_c1`
    - `speed=0.0629`, `gap=0.1855`, `lat=0.0130`, `center=0.0180`, `min_gap=1.4986`, `reset=0`, `collision=0`
  - `light/chunk1`: PASS
    - run: `2026-06-29_22-53-14_paper_validation_gated_hard_curriculum_20260629_223342_s_light_c1`
    - `speed=0.0673`, `gap=0.2191`, `lat=0.0526`, `center=0.0524`, `min_gap=1.3688`, `reset=0`, `collision=0`
  - `easy/chunk1`: CONTINUE, not pass
    - run: `2026-06-29_23-50-33_paper_validation_gated_hard_curriculum_20260629_223342_s_easy_c1`
    - `speed=0.0949`, `gap=0.2705`, `lat=0.3394`, `center=0.2795`, `min_gap=1.1687`, `reset=0`, `collision=0`
    - failed mainly on lateral/centerline/min-gap versus easy thresholds.
  - `easy/chunk2`: ABORT
    - run: `2026-06-30_01-25-47_paper_validation_gated_hard_curriculum_20260629_223342_s_easy_c2`
    - final assessment: `speed=0.1268`, `gap=0.3302`, `lat=0.2646`, `center=0.2622`, `min_gap=0.9554`, `reset=1.0000`, `collision=0`
    - pipeline stopped with `exit status=20`.
- Effect interpretation:
  - The validation gate worked in the sense that it did not pass the bad `easy/chunk2` checkpoint to medium/hard.
  - The overall curriculum did not succeed beyond light; easy attack training destabilized the policy.
  - Best clean checkpoint from this run is the light pass:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-29_22-53-14_paper_validation_gated_hard_curriculum_20260629_223342_s_light_c1/model_final.pt`
  - `easy/chunk1/model_final.pt` is not a passing checkpoint and should not be treated as final.
  - `easy/chunk2/model_final.pt` is invalid and should not be used.
- Main failure pattern:
  - During easy training, lateral/centerline/gap errors drifted upward.
  - `critic_grad_norm` rose strongly during easy/chunk2, exceeding `300` in monitor windows.
  - `min_pair_gap_mean` fell below `0.90`.
  - `reset_on_bad_ori` rose to `1.0`.
- Next training should not simply continue this run. It should restart from the light pass or the previous stable medium baseline with a gentler easy transition, shorter easy chunks, and an earlier rollback/abort condition for a first non-passing easy chunk.

2026-06-30 clarification of "did not cross easy":

- "Did not cross easy" means the pipeline entered the `easy` attack stage and trained it, but no `easy` checkpoint satisfied the configured validation thresholds required to advance to `medium`.
- It does not mean the `easy` command failed to start.
- Actual sequence:
  - `off` passed.
  - `light` passed.
  - `easy/chunk1` completed but assessment was `CONTINUE`, not `PASS`.
  - `easy/chunk2` completed but assessment was `ABORT` because `reset_on_bad_ori` reached `1.0` and safety/formation metrics degraded.
- Therefore the curriculum stopped at `easy` and never started `medium` or `hard`.
- In this context, "cross easy" means:
  - `easy` assessment status must be `PASS`;
  - then the script forwards the best passing `easy` checkpoint to the next stage, `medium`.

2026-06-30 why easy failed despite previous hard/medium-looking runs:

- The previous "hard can run" evidence and the current validation-gated curriculum are not equivalent.
- Older short tests:
  - `paper100_flatterrain_test_20260629_114627_s_hard` only had about 20 metric rows and used the flatterrain/local-ground test route. It showed the policy could run briefly under `hard`, but it was not the same as a long validation-gated city training stage.
  - Its hard metrics were roughly: `speed=0.2432`, `gap=0.2190`, `lat=0.1112`, `center=0.0420`, `min_gap=1.2751`, `reset=0`, `collision=0`, but only over a very short window.
- Older successful city attack training was also short:
  - `platoon5_city_easyattack_from_light_ft400` had 400 rows and old easy attack values `pos=2.0`, `acc=0.5`, `dos=0.05`, with good final metrics: `speed=0.0998`, `gap=0.2170`, `lat=0.0617`, `center=0.0451`, `min_gap=1.4400`, `reset=0`, `collision=0`.
  - That was a short fine-tune, not a multi-thousand-iteration easy stage after re-running off/light.
- Current validation-gated run:
  - started from `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`;
  - then trained `off` for 500 and `light` for 1500 before easy.
  - `light` did pass, but it was already near several thresholds:
    - `gap=0.2191` vs threshold `0.22`
    - `min_gap=1.3688` vs threshold `1.35`
    - `shield_lateral_rate=0.5411`
  - Compared with old short city light/easy runs, this is not a very strong margin.
- Easy failure mechanism in the current run:
  - `easy/chunk1` did not pass: `lat=0.3394`, `center=0.2795`, `min_gap=1.1687`, even though reset was still 0.
  - The script continued because it was `CONTINUE`, then `easy/chunk2` became unstable:
    - `reset_on_bad_ori=1.0`
    - `min_gap=0.9554`
    - `gap=0.3302`
    - `shield_lateral_rate=0.9704`
    - `critic_grad_norm` rose strongly in monitor windows, exceeding `300` during easy/chunk2.
- Main interpretation:
  - The current long off/light/easy curriculum is causing policy drift. It likely over-adapts away from the previously robust medium-trained behavior before entering easy.
  - The problem is not that easy attack is inherently impossible; prior short easy runs show easy can work.
  - The problem is the current stage schedule/checkpoint selection: long low-attack stages plus long easy chunks are degrading the policy before or during easy.
- Next practical change should be:
  - either start easy directly from a known stable attack-trained checkpoint instead of retraining off/light first;
  - or make off/light pure validation/very-short stages;
  - make easy chunks much shorter, e.g. 100-400 iterations;
  - stop or roll back immediately when easy/chunk1 fails key metrics, rather than allowing a second long chunk from a non-passing checkpoint;
  - consider using `model_best.pt` inside a stage for continuation instead of `model_final.pt` when the final window is drifting.

2026-06-30 hard transition plan and script support:

- User goal: smoothly transition from `easy` through `medium` to `hard`, instead of failing in `easy` or jumping directly to overly strong `hard`.
- Key strategy:
  - Do not re-run long `off` and `light` stages before attack training.
  - Start from the stable attack-trained baseline:
    - `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - Use small attack-strength ladder stages rather than only `easy -> medium -> hard`.
  - Keep each ladder stage short; if a stage does not pass, stop and adjust instead of continuing from a non-passing `model_final.pt`.
- Script changes:
  - Modified `scripts/tools/run_platoon_adaptive_hard_curriculum.sh` to support custom arrays:
    - `STAGE_LABELS_OVERRIDE`
    - `STAGE_ATTACK_LEVELS_OVERRIDE`
    - `STAGE_PASS_PROFILES_OVERRIDE`
    - `STAGE_MAX_FDI_POS_OVERRIDE`
    - `STAGE_MAX_FDI_ACC_OVERRIDE`
    - `STAGE_MAX_DOS_RATE_OVERRIDE`
  - Each stage can now use a custom FDI/DoS strength while still using an existing attack level/profile for logging and pass thresholds.
  - Manifest now records the real stage attack levels and FDI/DoS override arrays.
  - Verified:
    - `bash -n scripts/tools/run_platoon_adaptive_hard_curriculum.sh`
    - default dry-run
    - custom 8-stage ladder dry-run with `STAGE_LIMIT=8`
- Added helper:
  - `./start_hard_ladder`
  - It starts a short-stage hard ladder with 5-minute monitoring and pointer updates.
- Ladder in `./start_hard_ladder`:
  - `easy_a`: pos `1.25`, acc `0.25`, dos `0.05`
  - `easy_b`: pos `1.50`, acc `0.35`, dos `0.07`
  - `med_a`: pos `1.75`, acc `0.45`, dos `0.09`
  - `med_b`: pos `2.00`, acc `0.50`, dos `0.10`
  - `hard_a`: pos `2.50`, acc `0.75`, dos `0.12`
  - `hard_b`: pos `3.50`, acc `1.20`, dos `0.16`
  - `hard_c`: pos `5.00`, acc `2.00`, dos `0.22`
  - `hard`: pos `8.00`, acc `3.00`, dos `0.30`
- Training settings in `./start_hard_ladder`:
  - 8 stages, each `200` iterations only.
  - `ASSESS_WINDOW=100`
  - `PLOT_SMOOTH=50`
  - `VALIDATION_WORSE_TOL=0.005`
  - `ALLOW_EARLY_PASS=1`
  - `VALIDATION_ROLLBACK=1`
- Practical conclusion:
  - The intended path to hard is now a validation ladder:
    - pass a short stage -> forward checkpoint;
    - fail a short stage -> stop immediately and adjust the previous stage/attack increment;
    - never continue for thousands of iterations from a stage that has already failed lateral/centerline/min-gap metrics.

2026-06-30 hard ladder retry policy adjustment:

- User pointed out that "stop if not pass" is too rigid.
- Clarification:
  - The pipeline should not continue increasing attack strength from a non-passing checkpoint.
  - But it also should not necessarily terminate the whole experiment after one short non-passing chunk.
  - Better behavior is: pause escalation, retry the same small attack step for a short budget, continue only if the validation score improves, and stop/rollback if it deteriorates.
- Updated `./start_hard_ladder`:
  - still uses `200`-iteration chunks;
  - now allows multiple short chunks per stage:
    - early/easy/medium ladder stages: max `600` iterations;
    - hard ladder stages: max `800` or `1000` iterations;
  - changed `VALIDATION_WORSE_TOL` from `0.005` to `0.01`.
- New interpretation:
  - Not passing a chunk means "do not go to the next stronger attack yet".
  - If metrics are stable or improving, the script may try another short chunk at the same attack strength.
  - If metrics degrade beyond the validation tolerance or hit critical safety metrics, stop before forwarding the bad checkpoint.
- This is safer than long training from a non-passing final checkpoint, and less brittle than stopping after a single short chunk.

2026-06-30 03:37 hard ladder training started:

- Started a new hard-ladder curriculum with `./start_hard_ladder`.
- New run:
  - `RUN_TAG=paper_hard_ladder_20260630_033651`
  - training PID `3143424`
  - monitor PID `3143425`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_hard_ladder_20260630_033651.log`
- Start checkpoint:
  - `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
- First stage:
  - `easy_a`
  - actual attack override: `attack_level=easy`, `max_fdi_pos=1.25`, `max_fdi_acc=0.25`, `max_dos_rate=0.05`
  - pass profile: `easy`
  - chunk size: `200` iterations
- Verification:
  - IsaacLab started successfully.
  - Training entered learning iterations around `13866/14064`.
  - training speed about `900 steps/s`.
  - `reset_on_bad_ori=0.0000` in early training output.
  - immediate monitor pointed to new run directory:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-30_03-37-07_paper_hard_ladder_20260630_033651_s_easy_a_c1`
  - immediate monitor status: `OK`
  - early metrics:
    - `centerline_error_abs_mean ~= 0.0077`
    - `lateral_error_abs_mean ~= 0.0137`
    - `gap_error_abs_mean ~= 0.2094`
    - `min_pair_gap_mean ~= 1.4649`
    - `speed_error_abs_mean ~= 0.0784`
    - `termination_reset_on_bad_ori = 0.0000`
    - `collision_rate = 0.0000`
- Monitoring:
  - 5-minute monitor is active for PID `3143424`.
  - Use `./watchtrain` for training log and `./watchmonitor` for monitor log.

2026-06-30 03:43 full off-to-hard supervisor started:

- User requested a persistent process with the following stop criterion:
  - continue checking every 5 minutes;
  - if training crashes/stops/fails before a complete `off -> hard` run with full saved data, modify the training strategy and continue;
  - stop only after a complete no-attack-to-hard training round succeeds and saves full data.
- Added automation:
  - `scripts/tools/supervise_hard_ladder.py`
  - `./start_full_hard_supervisor`
- Supervisor behavior:
  - checks every `300` seconds;
  - if current pipeline is still running, logs status;
  - if current pipeline stopped:
    - marks success only when log contains `stage hard passed`, `package tar`, and `exit status=0`;
    - otherwise launches a more conservative full off-to-hard profile from stable baseline.
  - profiles become progressively more conservative:
    - `full_ladder_v1`
    - `full_ladder_v2_finer_easy`
    - `full_ladder_v3_low_action`
  - each profile starts from:
    - `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
  - every profile includes `off` and continues through `hard`, satisfying the user's full-round requirement.
- Pipeline changes supporting this:
  - `scripts/tools/run_platoon_adaptive_hard_curriculum.sh` now accepts `COMMON_OVERRIDES_EXTRA` so rescue profiles can reduce action clip/action scale without editing the script again.
  - `scripts/tools/monitor_current_training.py` now prefers `.current_adaptive_training.tag` when locating the current run directory, avoiding stale old-run alerts during startup.
- Old incomplete run cleanup:
  - The previous `paper_hard_ladder_20260630_033651` run started at `easy_a`, so it did not meet the full `off -> hard` criterion.
  - Its stale Isaac Python child process `3143435` was still running and occupying GPU memory; killed process group `3143424`.
  - After cleanup, only the new full-run training process remained on GPU.
- New supervisor launch:
  - supervisor PID: `3144145`
  - monitor PID: `3144147`
  - current pipeline PID: `3144148`
  - current run tag: `paper_full_hard_auto_a1_20260630_034215`
  - current log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_full_hard_auto_a1_20260630_034215.log`
- Current active stage:
  - `off`
  - run directory:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-30_03-42-31_paper_full_hard_auto_a1_20260630_034215_s_off_c1`
- Immediate monitor after cleanup:
  - status `OK`
  - `centerline_error_abs_mean ~= 0.0057`
  - `lateral_error_abs_mean ~= 0.0052`
  - `gap_error_abs_mean ~= 0.1742`
  - `min_pair_gap_mean ~= 1.4997`
  - `speed_error_abs_mean ~= 0.0669`
  - `termination_reset_on_bad_ori = 0.0000`
  - `collision_rate = 0.0000`
- Practical conclusion:
  - The persistent full off-to-hard supervisor is now active.
  - The current run starts from no attack (`off`) and will attempt to progress to `hard`; if it fails before completing and packaging, the supervisor will launch a modified, more conservative full attempt.

2026-06-30 11:59 full off-to-hard goal status and supervisor fix:

- User asked whether the requested final result has been achieved.
- Current answer: not yet fully achieved.
- First supervised attempt:
  - run tag: `paper_full_hard_auto_a1_20260630_034215`
  - profile: `full_ladder_v1`
  - important positive result: the pipeline successfully progressed from `off` through `light`, `easy_0`, `easy_a`, `easy_b`, `med_a`, `med_b`, `hard_a`, `hard_b`, and `hard_c`.
  - final `hard` stage did start, but did not pass.
- First attempt stage details:
  - `off`: PASS
  - `light`: PASS
  - `easy_0`: PASS
  - `easy_a`: PASS
  - `easy_b`: PASS
  - `med_a`: PASS
  - `med_b`: PASS
  - `hard_a`: PASS
  - `hard_b`: PASS
  - `hard_c`: PASS
  - final `hard/chunk1`: CONTINUE, not pass
    - `speed=0.3054`
    - `gap=0.3021`
    - `lat=0.1598`
    - `center=0.0678`
    - `min_gap=1.0024`
    - `reset=0`
    - `collision=0`
  - final `hard/chunk2`: CONTINUE but validation score worsened
    - `speed=0.3213`
    - `gap=0.3250`
    - `lat=0.1675`
    - `center=0.0618`
    - `min_gap=0.9148`
    - `reset=0`
    - `collision=0`
  - pipeline stopped with `exit status=22`, preserving best checkpoint at:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-30_04-42-30_paper_full_hard_auto_a1_20260630_034215_s_hard_c1`
- This means:
  - The result is close but not complete.
  - It has not met the user's stop criterion because final `hard` did not pass and the package was not produced.
- Supervisor bug found:
  - The pipeline process became a zombie (`STAT=Z`), but the supervisor's previous `pid_alive()` used `os.kill(pid, 0)` only, which treats zombies as alive.
  - Because of that, the supervisor logged `RUNNING` repeatedly and did not start the next recovery attempt.
- Fix applied:
  - `scripts/tools/supervise_hard_ladder.py` now checks `/proc/<pid>/stat` and treats `Z` zombie state as not alive.
  - It also attempts `os.waitpid(pid, os.WNOHANG)` when possible.
  - `./start_full_hard_supervisor` no longer clears `.hard_ladder_supervisor_state.json` unless `RESET_SUPERVISOR_STATE=1`, so retries continue with more conservative profiles instead of restarting at attempt 1.
- Continued training:
  - restarted supervisor with existing state preserved.
  - new supervisor PID: `3157244`
  - new monitor PID: `3157246`
  - new pipeline PID: `3157247`
  - new run tag: `paper_full_hard_auto_a2_20260630_115920`
  - new profile: `full_ladder_v2_finer_easy`
  - current run starts again at `off`, satisfying the full `off -> hard` requirement.
- Practical conclusion:
  - The requested final result has not yet been achieved.
  - The automation to keep trying until a full `off -> hard` pass and package is now corrected and running a second, more conservative attempt.

2026-06-30 12:02 continued supervision after user persistence request:

- User explicitly requested that the conversation/work should not stop until the final objective is achieved:
  - complete full `off/no-attack -> hard` training;
  - save/package complete data;
  - if training stops, crashes, or fails, modify and continue.
- Current status:
  - final objective is still not achieved.
  - supervisor remains active and will continue launching more conservative full attempts until success.
- Additional monitor fix:
  - `scripts/tools/monitor_current_training.py` previously could fall back to unrelated old `paper_*` run directories before the new run directory was created.
  - This caused a misleading startup monitor line using old metrics.
  - The fallback is now removed: the monitor only uses the current `.current_adaptive_training.tag`; if no current run dir exists yet, it reports missing current metrics instead of using stale data.
- Current active attempt:
  - attempt 2
  - profile `full_ladder_v2_finer_easy`
  - run tag `paper_full_hard_auto_a2_20260630_115920`
  - pipeline PID `3157247`
  - supervisor PID `3157244`
  - monitor PID `3157246`
  - current stage at the time of check: `off`
  - current run directory:
    - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-06-30_11-59-35_paper_full_hard_auto_a2_20260630_115920_s_off_c1`
- Immediate monitor after fix:
  - status `OK`
  - `centerline_error_abs_mean ~= 0.0204`
  - `lateral_error_abs_mean ~= 0.0155`
  - `gap_error_abs_mean ~= 0.2114`
  - `min_pair_gap_mean ~= 1.4436`
  - `speed_error_abs_mean ~= 0.0730`
  - `termination_reset_on_bad_ori = 0.0000`
  - `collision_rate = 0.0000`
- Next action:
  - continue monitoring the supervisor and current pipeline.
  - if attempt 2 fails before final hard pass/package, verify supervisor launches attempt 3 (`full_ladder_v3_low_action`) or modify the profile further if needed.

2026-06-30 12:12 hard-ladder recovery update:

- Attempt 2 (`paper_full_hard_auto_a2_20260630_115920`, profile `full_ladder_v2_finer_easy`) failed in the initial `off` stage:
  - `off/chunk1`: `CONTINUE`, `speed=0.0727`, `gap=0.2117`, `lat=0.0157`, `center=0.0207`, `min_gap=1.4432`.
  - `off/chunk2`: still `CONTINUE`, `speed=0.0748`, `gap=0.2169`, `lat=0.0192`, `center=0.0294`, `min_gap=1.4264`.
  - Validation score worsened, so the pipeline correctly stopped before continuing from a degraded checkpoint.
- Interpretation:
  - This was not an attack-stage failure.
  - The likely cause was the v2 profile's global `env.algorithm.happo_action_clip=0.35`, which made the no-attack/off longitudinal gap metric worse than the v1 stable profile.
- Fix applied:
  - `scripts/tools/run_platoon_adaptive_hard_curriculum.sh` now supports per-stage `STAGE_EXTRA_OVERRIDES_<idx>` so final hard can receive extra safety settings without changing off/light/easy.
  - `scripts/tools/supervise_hard_ladder.py` profile 3 was changed from global low-action (`full_ladder_v3_low_action`) to `full_ladder_v3_fine_hard_no_clip`.
  - New profile keeps the v1 stable action scale/clip, adds finer late-hard stages:
    - `hard_d`: `max_fdi_pos=6.75`, `max_fdi_acc=2.50`, `max_dos_rate=0.265`
    - `hard_e`: `max_fdi_pos=7.35`, `max_fdi_acc=2.75`, `max_dos_rate=0.285`
    - final `hard`: `max_fdi_pos=8.00`, `max_fdi_acc=3.00`, `max_dos_rate=0.30`
  - Final hard only now gets additional shield override:
    - `d_drop=1.55`
    - `forward_bias_gain=0.08`
    - `forward_bias_clip=0.03`
    - `forward_bias_min_gap=1.12`
    - `forward_bias_min_command=0.30`
- Process correction:
  - Old supervisor briefly launched attempt 3 with the old low-action profile before the restart.
  - That stale attempt was terminated.
  - Current active run is attempt 4:
    - run tag `paper_full_hard_auto_a4_20260630_121143`
    - profile `full_ladder_v3_fine_hard_no_clip`
    - pipeline PID `3159616`
    - supervisor PID `3159613`
    - monitor PID `3159615`
  - It has restarted from `off`, preserving the required full `off -> hard` training path.

2026-06-30 12:24 attempt 4 early progress:

- Current active run remains:
  - run tag `paper_full_hard_auto_a4_20260630_121143`
  - profile `full_ladder_v3_fine_hard_no_clip`
  - pipeline PID `3159616`
- Stage results so far:
  - `off`: PASS after 100 iterations
    - `speed=0.0646`
    - `gap=0.1690`
    - `lat=0.0108`
    - `center=0.0135`
    - `min_gap=1.4992`
    - `reset=0`
    - `collision=0`
  - `light`: PASS after 100 iterations
    - `speed=0.0647`
    - `gap=0.1815`
    - `lat=0.0097`
    - `center=0.0132`
    - `min_gap=1.4976`
    - `reset=0`
    - `collision=0`
  - Current stage: `easy_0` with `max_fdi_pos=1.00`, `max_fdi_acc=0.18`, `max_dos_rate=0.035`.
- Process hygiene:
  - Old attempt 3 left an orphan Isaac Python child process after its pipeline shell was killed.
  - The stale process had run name `paper_full_hard_auto_a3_20260630_120920_s_off_c1` and was still using GPU.
  - It was terminated; only the current attempt 4 training process remains.
- Supervisor:
  - Ordinary `nohup` supervisor relaunch exited early in this environment.
  - Supervisor was relaunched with `setsid`, matching the previously validated reliable background method.
  - Active supervisor PID after relaunch: `3161828`.

2026-06-30 12:40 attempt 4 easy-stage progress:

- Attempt 4 continues to run under supervisor/monitor:
  - pipeline PID `3159616`
  - supervisor PID `3161828`
  - monitor PID `3159615`
- Additional passed stages:
  - `easy_0`: PASS after 150 iterations
    - `max_fdi_pos=1.00`
    - `max_fdi_acc=0.18`
    - `max_dos_rate=0.035`
    - `speed=0.0732`
    - `gap=0.2106`
    - `lat=0.0160`
    - `center=0.0214`
    - `min_gap=1.4699`
    - `reset=0`
    - `collision=0`
  - `easy_a`: PASS after 150 iterations
    - `max_fdi_pos=1.25`
    - `max_fdi_acc=0.25`
    - `max_dos_rate=0.05`
    - `speed=0.0697`
    - `gap=0.2233`
    - `lat=0.0305`
    - `center=0.0267`
    - `min_gap=1.4298`
    - `reset=0`
    - `collision=0`
- Current active stage:
  - `easy_b`
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`
- Intermediate `easy_b` monitor remains within the easy pass envelope:
  - `speed_error_abs_mean ~= 0.082`
  - `gap_error_abs_mean ~= 0.2265`
  - `lateral_error_abs_mean ~= 0.0523`
  - `min_pair_gap_mean ~= 1.416`
  - `reset_on_bad_ori = 0`
  - `collision_rate = 0`

2026-06-30 12:42 attempt 4 entered medium:

- `easy_b`: PASS after 150 iterations
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`
  - `speed=0.0822`
  - `gap=0.2326`
  - `lat=0.0724`
  - `center=0.0479`
  - `min_gap=1.3931`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `med_a`
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`
- The full ladder has now successfully crossed:
  - `off`
  - `light`
  - `easy_0`
  - `easy_a`
  - `easy_b`
- Remaining stages for final objective:
  - `med_a`
  - `med_b`
  - `hard_a`
  - `hard_b`
  - `hard_c`
  - `hard_d`
  - `hard_e`
  - final `hard`

2026-06-30 12:47 attempt 4 medium progress:

- `med_a`: PASS after 150 iterations
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`
  - `speed=0.0969`
  - `gap=0.2355`
  - `lat=0.1042`
  - `center=0.0658`
  - `min_gap=1.3727`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `med_b`
  - `max_fdi_pos=2.00`
  - `max_fdi_acc=0.50`
  - `max_dos_rate=0.10`
- Remaining stages:
  - `med_b`
  - `hard_a`
  - `hard_b`
  - `hard_c`
  - `hard_d`
  - `hard_e`
  - final `hard`

2026-06-30 12:54 attempt 4 entered hard:

- `med_b`: PASS after 150 iterations
  - `max_fdi_pos=2.00`
  - `max_fdi_acc=0.50`
  - `max_dos_rate=0.10`
  - `speed=0.1069`
  - `gap=0.2389`
  - `lat=0.1141`
  - `center=0.0742`
  - `min_gap=1.3504`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_a`
  - `max_fdi_pos=2.75`
  - `max_fdi_acc=0.75`
  - `max_dos_rate=0.12`
- The full ladder has now crossed `off -> light -> easy -> medium` and entered hard.
- Remaining stages for final objective:
  - `hard_a`
  - `hard_b`
  - `hard_c`
  - `hard_d`
  - `hard_e`
  - final `hard`

2026-06-30 13:00 attempt 4 hard progress:

- `hard_a`: PASS after 150 iterations
  - `max_fdi_pos=2.75`
  - `max_fdi_acc=0.75`
  - `max_dos_rate=0.12`
  - `speed=0.1377`
  - `gap=0.2427`
  - `lat=0.1529`
  - `center=0.1001`
  - `min_gap=1.2987`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_b`
  - `max_fdi_pos=4.00`
  - `max_fdi_acc=1.30`
  - `max_dos_rate=0.18`
- Remaining stages:
  - `hard_b`
  - `hard_c`
  - `hard_d`
  - `hard_e`
  - final `hard`

2026-06-30 13:07 attempt 4 hard_b passed:

- `hard_b`: PASS after 150 iterations
  - `max_fdi_pos=4.00`
  - `max_fdi_acc=1.30`
  - `max_dos_rate=0.18`
  - `speed=0.2006`
  - `gap=0.2446`
  - `lat=0.1807`
  - `center=0.1071`
  - `min_gap=1.2538`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_c`
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
- Remaining stages:
  - `hard_c`
  - `hard_d`
  - `hard_e`
  - final `hard`

2026-06-30 13:13 attempt 4 hard_c passed:

- `hard_c`: PASS after 150 iterations
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
  - `speed=0.2639`
  - `gap=0.2751`
  - `lat=0.1674`
  - `center=0.0929`
  - `min_gap=1.1219`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_d`
  - `max_fdi_pos=6.75`
  - `max_fdi_acc=2.50`
  - `max_dos_rate=0.265`
- This is the first new fine-hard stage inserted between previous passing `hard_c` and final hard.
- Remaining stages:
  - `hard_d`
  - `hard_e`
  - final `hard`

2026-06-30 13:20 attempt 4 hard_d status:

- Active run tag: `paper_full_hard_auto_a4_20260630_121143`
- Active stage:
  - `hard_d`
  - `max_fdi_pos=6.75`
  - `max_fdi_acc=2.50`
  - `max_dos_rate=0.265`
- `hard_d/chunk1`: CONTINUE after 120 iterations
  - `speed=0.2896`
  - `gap=0.2991`
  - `lat=0.1518`
  - `center=0.0731`
  - `min_gap=1.0158`
  - `reset=0`
  - `collision=0`
- Reason for not passing: hard profile requires `min_gap >= 1.05`; chunk1 only reached `1.0158`. Other safety terms are acceptable and no reset/collision occurred.
- Pipeline correctly continued from the best checkpoint into `hard_d/chunk2` instead of forwarding a marginal model.
- Current process state is normal:
  - pipeline PID `3159616`
  - current Isaac train child PID `3169002`
  - training speed around `850-880 steps/s`
- Next action: wait for `hard_d/chunk2` assessment. If `hard_d` cannot pass by max iterations, add a small stage-specific shield/forward-bias adjustment to `hard_d` and continue rather than restarting blindly.

2026-06-30 13:26 attempt 4 stopped and attempt 5 started:

- `hard_d/chunk2` assessment:
  - `status=CONTINUE`
  - `total=240`
  - `speed=0.2973`
  - `gap=0.3080`
  - `lat=0.1554`
  - `center=0.0719`
  - `min_gap=0.9766`
  - `reset=0`
  - `collision=0`
- This was worse than `hard_d/chunk1` (`min_gap=1.0158`), so validation rollback stopped attempt 4 before forwarding a degraded checkpoint.
- Failure mode is mainly hard-stage safe gap erosion, not lateral/centerline collapse or orientation failure.
- Modified `scripts/tools/supervise_hard_ladder.py` by adding profile `full_ladder_v4_hard_gap_guard`:
  - hard tail is now finer: `6.0 -> 6.5 -> 7.0 -> 7.5 -> 8.0`
  - stage-specific hard tail overrides add larger `d_drop`, less aggressive `catchup_action`, catchup lateral/centerline limits, and small gap-gated `forward_bias`.
  - off/light/easy/medium settings are unchanged.
- Restarted supervisor and launched attempt 5:
  - run tag `paper_full_hard_auto_a5_20260630_132638`
  - profile `full_ladder_v4_hard_gap_guard`
  - pipeline PID `3169965`
  - current stage `off`

2026-06-30 13:32 attempt 5 first check:

- `off`: PASS after 100 iterations
  - `speed=0.0646`
  - `gap=0.1690`
  - `lat=0.0108`
  - `center=0.0135`
  - `min_gap=1.4992`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `light`
- Process state:
  - supervisor PID `3169963`
  - pipeline PID `3169965`
  - current Isaac child is running normally.
- No regression from the new hard-tail profile in the no-attack stage.

2026-06-30 13:37 attempt 5 second check:

- `light`: PASS after 100 iterations
  - `speed=0.0647`
  - `gap=0.1815`
  - `lat=0.0097`
  - `center=0.0132`
  - `min_gap=1.4976`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `easy_0`
  - `max_fdi_pos=1.00`
  - `max_fdi_acc=0.18`
  - `max_dos_rate=0.035`
- New profile `full_ladder_v4_hard_gap_guard` has not regressed off/light behavior.

2026-06-30 13:43 attempt 5 easy_0 passed:

- `easy_0`: PASS after 150 iterations
  - `max_fdi_pos=1.00`
  - `max_fdi_acc=0.18`
  - `max_dos_rate=0.035`
  - `speed=0.0732`
  - `gap=0.2106`
  - `lat=0.0160`
  - `center=0.0214`
  - `min_gap=1.4699`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `easy_a`
  - `max_fdi_pos=1.25`
  - `max_fdi_acc=0.25`
  - `max_dos_rate=0.05`

2026-06-30 13:49 attempt 5 easy_a passed:

- `easy_a`: PASS after 150 iterations
  - `max_fdi_pos=1.25`
  - `max_fdi_acc=0.25`
  - `max_dos_rate=0.05`
  - `speed=0.0697`
  - `gap=0.2233`
  - `lat=0.0305`
  - `center=0.0267`
  - `min_gap=1.4298`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `easy_b`
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`

2026-06-30 13:54 attempt 5 easy_b in progress:

- `easy_b` chunk1 is still running at this check.
- Latest monitor snapshot from partial metrics:
  - `speed_error_abs_mean ~= 0.0818`
  - `gap_error_abs_mean ~= 0.2268`
  - `lateral_error_abs_mean ~= 0.0528`
  - `centerline_error_abs_mean ~= 0.0373`
  - `min_pair_gap_mean ~= 1.4149`
  - `reset=0`
  - `collision=0`
- This partial state is healthy; wait for the formal `[ASSESS][easy_b]` before acting.

2026-06-30 13:59 attempt 5 easy_b passed:

- `easy_b`: PASS after 150 iterations
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`
  - `speed=0.0822`
  - `gap=0.2326`
  - `lat=0.0724`
  - `center=0.0479`
  - `min_gap=1.3931`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `med_a`
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`
- Attempt 5 has now crossed off/light/easy successfully.

2026-06-30 14:04 attempt 5 med_a passed:

- `med_a`: PASS after 150 iterations
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`
  - `speed=0.0969`
  - `gap=0.2355`
  - `lat=0.1042`
  - `center=0.0658`
  - `min_gap=1.3727`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `med_b`
  - `max_fdi_pos=2.00`
  - `max_fdi_acc=0.50`
  - `max_dos_rate=0.10`

2026-06-30 14:09 attempt 5 med_b passed:

- `med_b`: PASS after 150 iterations
  - `max_fdi_pos=2.00`
  - `max_fdi_acc=0.50`
  - `max_dos_rate=0.10`
  - `speed=0.1069`
  - `gap=0.2389`
  - `lat=0.1141`
  - `center=0.0742`
  - `min_gap=1.3504`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_a`
  - `max_fdi_pos=2.75`
  - `max_fdi_acc=0.75`
  - `max_dos_rate=0.12`
- Attempt 5 has crossed off/light/easy/medium successfully; remaining work is the hard ladder.

2026-06-30 14:15 attempt 5 hard_a passed:

- `hard_a`: PASS after 150 iterations
  - `max_fdi_pos=2.75`
  - `max_fdi_acc=0.75`
  - `max_dos_rate=0.12`
  - `speed=0.1377`
  - `gap=0.2427`
  - `lat=0.1529`
  - `center=0.1001`
  - `min_gap=1.2987`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_b`
  - `max_fdi_pos=4.00`
  - `max_fdi_acc=1.30`
  - `max_dos_rate=0.18`
- Hard entry remains stable; no safety regression before the modified hard tail.

2026-06-30 14:20 attempt 5 hard_b in progress:

- `hard_b` chunk1 is still running at this check.
- Latest partial monitor values:
  - `speed_error_abs_mean ~= 0.1919`
  - `gap_error_abs_mean ~= 0.2252`
  - `lateral_error_abs_mean ~= 0.1508`
  - `centerline_error_abs_mean ~= 0.0831`
  - `min_pair_gap_mean ~= 1.3174`
  - `reset=0`
  - `collision=0`
- Partial values are well within hard thresholds. Wait for formal `[ASSESS][hard_b]`.

2026-06-30 14:25 attempt 5 hard_b passed:

- `hard_b`: PASS after 150 iterations
  - `max_fdi_pos=4.00`
  - `max_fdi_acc=1.30`
  - `max_dos_rate=0.18`
  - `speed=0.2006`
  - `gap=0.2446`
  - `lat=0.1807`
  - `center=0.1071`
  - `min_gap=1.2538`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_c`
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
- Next critical point is after `hard_c`: new profile enters the modified hard tail (`hard_d=6.5/2.4/0.255` with stage-specific gap guard).

2026-06-30 14:29 attempt 5 hard_c passed and modified hard tail started:

- `hard_c`: PASS after 150 iterations
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
  - `speed=0.2639`
  - `gap=0.2751`
  - `lat=0.1674`
  - `center=0.0929`
  - `min_gap=1.1219`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_d`
  - `max_fdi_pos=6.50`
  - `max_fdi_acc=2.40`
  - `max_dos_rate=0.255`
- Confirmed `hard_d` stage-specific overrides are active:
  - `env.safety_shield.d_drop=1.55`
  - `env.safety_shield.catchup_action=-0.335`
  - `env.safety_shield.catchup_lateral_limit=0.36`
  - `env.safety_shield.catchup_centerline_limit=0.25`
  - `env.safety_shield.forward_bias_gain=0.06`
  - `env.safety_shield.forward_bias_clip=0.020`
  - `env.safety_shield.forward_bias_min_gap=1.16`
  - `env.safety_shield.forward_bias_min_command=0.30`
- This is the first stage testing the new hard-gap-guard profile.

2026-06-30 14:32 attempt 5 hard_d passed:

- `hard_d`: PASS after 100 iterations
  - `max_fdi_pos=6.50`
  - `max_fdi_acc=2.40`
  - `max_dos_rate=0.255`
  - `speed=0.2850`
  - `gap=0.2284`
  - `lat=0.1497`
  - `center=0.0685`
  - `min_gap=1.1056`
  - `reset=0`
  - `collision=0`
- This is the first evidence that the new hard-gap-guard profile is effective:
  - previous attempt at a nearby hard tail failed because `min_gap` fell below `1.05`;
  - new `hard_d` keeps `min_gap=1.1056` while keeping speed error below the hard threshold `0.30`.
- Current active stage:
  - `hard_e`
  - `max_fdi_pos=7.00`
  - `max_fdi_acc=2.60`
  - `max_dos_rate=0.275`
- Confirmed `hard_e` overrides active:
  - `env.safety_shield.d_drop=1.60`
  - `env.safety_shield.catchup_action=-0.325`
  - `env.safety_shield.catchup_lateral_limit=0.34`
  - `env.safety_shield.catchup_centerline_limit=0.24`
  - `env.safety_shield.forward_bias_gain=0.08`
  - `env.safety_shield.forward_bias_clip=0.025`
  - `env.safety_shield.forward_bias_min_gap=1.18`

2026-06-30 14:36 attempt 5 hard_e in progress:

- `hard_e` chunk1 is still running at this check; no formal `[ASSESS][hard_e]` yet.
- Last stable hard-tail result remains `hard_d`:
  - `speed=0.2850`
  - `min_gap=1.1056`
  - `reset=0`
  - `collision=0`
- Continue waiting for formal `hard_e` assessment before changing anything.

2026-06-30 14:39 attempt 5 hard_e chunk1 did not pass:

- `hard_e/chunk1`: CONTINUE after 100 iterations
  - `max_fdi_pos=7.00`
  - `max_fdi_acc=2.60`
  - `max_dos_rate=0.275`
  - `speed=0.3042`
  - `gap=0.2384`
  - `lat=0.1583`
  - `center=0.0623`
  - `min_gap=1.0084`
  - `reset=0`
  - `collision=0`
- Failure is close but real:
  - hard speed threshold is `0.30`; observed `0.3042`
  - hard min-gap threshold is `1.05`; observed `1.0084`
- Pipeline continued into `hard_e/chunk2` from the best checkpoint rather than advancing.
- Do not change parameters until chunk2 assessment; if chunk2 worsens/stops, next adjustment should strengthen `hard_e`/later gap guard slightly and possibly add one more intermediate before `7.0/2.6/0.275`.
2026-06-30 14:48 hard ladder attempt 6 restarted with v5 profile:

- Attempt 5 (`paper_full_hard_auto_a5_20260630_132638`) improved over previous runs and passed through `hard_d`.
- The remaining failure point was `hard_e`:
  - `hard_e/chunk1`: `speed=0.3042`, `min_gap=1.0084`, `reset=0`, `collision=0`.
  - `hard_e/chunk2`: worsened to `speed=0.3155`, `min_gap=0.9439`.
  - Validation rollback correctly stopped instead of forwarding the degraded checkpoint.
- Interpretation: this is not a crash and not a need for longer training; the `6.5 -> 7.0` hard-tail jump was too large and allowed the policy/shield to trade safety gap for speed.
- Modified only the supervisor training ladder:
  - added `full_ladder_v5_hard_e_gap_guard`;
  - changed hard tail from `6.50 -> 7.00 -> 7.50 -> 8.00` to `6.40 -> 6.70 -> 7.00 -> 7.30 -> 7.65 -> 8.00`;
  - added stronger stage-specific `d_drop`, safer `catchup_action`, and larger `forward_bias_min_gap` for the hard tail.
- New active run:
  - tag: `paper_full_hard_auto_a6_20260630_144652`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_full_hard_auto_a6_20260630_144652.log`
  - training pipeline PID: `3180109`
  - supervisor PID: `3180669`
  - monitor loop PID: `3159615`
- Supervisor is now launched with `setsid` so it remains alive for 5-minute checks.

2026-06-30 14:52 attempt 6 off passed:

- `off`: PASS after 100 iterations.
- Metrics:
  - `speed=0.0646`
  - `gap=0.1690`
  - `lat=0.0108`
  - `center=0.0135`
  - `min_gap=1.4992`
  - `reset=0`
  - `collision=0`
- Current active stage is `light/chunk1`, continuing from `off/model_final.pt`.

2026-06-30 14:56 attempt 6 light passed:

- `light`: PASS after 100 iterations.
- Metrics:
  - `speed=0.0647`
  - `gap=0.1815`
  - `lat=0.0097`
  - `center=0.0132`
  - `min_gap=1.4976`
  - `reset=0`
  - `collision=0`
- Current active stage is `easy_0/chunk1`.

2026-06-30 20:50 attempt 12 easy_0 passed:

- `easy_0`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0733`
  - `gap=0.2105`
  - `lat=0.0161`
  - `center=0.0209`
  - `min_gap=1.4698`
  - `reset=0`
  - `collision=0`
- Current active stage is `easy_a/chunk1`.

2026-06-30 20:56 attempt 12 easy_a passed:

- `easy_a`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0700`
  - `gap=0.2229`
  - `lat=0.0300`
  - `center=0.0261`
  - `min_gap=1.4314`
  - `reset=0`
  - `collision=0`
- Current active stage is `easy_b/chunk1`.

2026-06-30 21:03 attempt 12 easy_b passed:

- `easy_b`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0825`
  - `gap=0.2323`
  - `lat=0.0738`
  - `center=0.0483`
  - `min_gap=1.3924`
  - `reset=0`
  - `collision=0`
- Current active stage is `med_a/chunk1`.
- `med_a` attack strength:
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`
- `easy_0` attack strength:
  - `max_fdi_pos=1.00`
  - `max_fdi_acc=0.18`
  - `max_dos_rate=0.035`

2026-06-30 15:04 attempt 6 easy_0 passed:

- `easy_0`: PASS after 150 iterations.
- Attack strength:
  - `max_fdi_pos=1.00`
  - `max_fdi_acc=0.18`
  - `max_dos_rate=0.035`
- Metrics:
  - `speed=0.0732`
  - `gap=0.2106`
  - `lat=0.0160`
  - `center=0.0214`
  - `min_gap=1.4699`
  - `reset=0`
  - `collision=0`
- Current active stage is `easy_a/chunk1`.
- `easy_a` attack strength:
  - `max_fdi_pos=1.25`
  - `max_fdi_acc=0.25`
  - `max_dos_rate=0.05`

2026-06-30 15:09 attempt 6 easy_a passed:

- `easy_a`: PASS after 150 iterations.
- Attack strength:
  - `max_fdi_pos=1.25`
  - `max_fdi_acc=0.25`
  - `max_dos_rate=0.05`
- Metrics:
  - `speed=0.0697`
  - `gap=0.2233`
  - `lat=0.0305`
  - `center=0.0267`
  - `min_gap=1.4298`
  - `reset=0`
  - `collision=0`
- Current active stage is `easy_b/chunk1`.
- `easy_b` attack strength:
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`

2026-06-30 15:15 attempt 6 easy_b passed:

- `easy_b`: PASS after 150 iterations.
- Attack strength:
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`
- Metrics:
  - `speed=0.0822`
  - `gap=0.2326`
  - `lat=0.0724`
  - `center=0.0479`
  - `min_gap=1.3931`
  - `reset=0`
  - `collision=0`
- Current active stage is `med_a/chunk1`.
- `med_a` attack strength:
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`

2026-06-30 21:09 attempt 12 med_a passed:

- `med_a`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0971`
  - `gap=0.2353`
  - `lat=0.1037`
  - `center=0.0657`
  - `min_gap=1.3727`
  - `reset=0`
  - `collision=0`
- Current active stage is `med_b/chunk1`.
- `med_b` attack strength:
  - `max_fdi_pos=2.00`
  - `max_fdi_acc=0.50`
  - `max_dos_rate=0.10`

2026-06-30 20:28 full hard ladder status:

- The full objective is not complete yet.
- `attempt 6` successfully passed:
  - off
  - light
  - easy_0
  - easy_a
  - easy_b
  - med_a
  - med_b
  - hard_a
  - hard_b
  - hard_c
  - hard_d
- `attempt 6` then failed at `hard_e0`:
  - `hard_e0/chunk1`: `speed=0.3131`, `gap=0.2795`, `lat=0.1680`, `center=0.0632`, `min_gap=0.9116`, `reset=0`, `collision=0`
  - `hard_e0/chunk2`: `speed=0.3184`, `gap=0.2848`, `lat=0.1626`, `center=0.0606`, `min_gap=0.8713`, `reset=0`, `collision=0`
  - pipeline stopped with critical bad metrics before forwarding the degraded checkpoint.
- Supervisor continued restarting attempts; current active run is:
  - tag: `paper_full_hard_auto_a10_20260630_194802`
  - pipeline PID: `3218370`
  - supervisor PID: `3180669`
  - current stage at 20:28 is `med_b/chunk1`
  - current live monitor at 20:26: `speed=0.1043`, `gap=0.2328`, `lat=0.1015`, `center=0.0649`, `min_gap=1.3730`, `reset=0`, `collision=0`
- Main remaining blocker: the v5 hard tail still cannot pass `hard_e0=6.70/2.50/0.265`; min-gap collapses below the hard threshold even with the added gap guard.
- Next likely modification if attempt 10 repeats the same failure:
  - reduce/insert another hard-tail stage before `6.70`;
  - make `hard_e0` less aggressive or add stronger gap-protection overrides;
  - avoid continuing chunks when `min_gap` is already below threshold, since chunk2 has repeatedly worsened the checkpoint.

2026-06-30 20:31 v6 profile prepared and v5 attempt stopped:

- Additional evidence from attempts 6/8/9 confirms v5 repeatedly fails at `hard_e0=6.70/2.50/0.265`.
- Root issue in v5:
  - hard pass requires `min_gap >= 1.05`;
  - safety shield default `d_crit=0.50` only brakes when the gap is already far below the validation threshold;
  - hard-tail overrides adjusted `d_drop/catchup/forward_bias`, but did not raise `d_crit`, so the controller could still compress gaps to `0.87~0.92`.
- Added `full_ladder_v6_hard_gap_brake_guard` in `scripts/tools/supervise_hard_ladder.py`.
- v6 changes:
  - hard tail split more finely:
    `6.20 -> 6.40 -> 6.55 -> 6.70 -> 6.90 -> 7.15 -> 7.50 -> 8.00`
  - hard-tail `d_crit` now increases from `0.95` to `1.18`;
  - hard-tail `d_drop` increases from `1.60` to `2.60`;
  - catch-up action is weakened from `-0.325` to `-0.185`;
  - forward speed bias remains available but is gated by larger `forward_bias_min_gap`.
- Stopped current v5 `attempt 10` early at 20:31 to avoid spending more time on the known bad hard tail.
- Supervisor PID `3180669` remains active; next attempt should launch with v6.

2026-06-30 20:34 v6 training started:

- The first supervisor restart after patch still used old in-memory code and launched `attempt 11` with v5.
- Fixed by stopping old supervisor and v5 `attempt 11`, then restarting supervisor with `setsid`.
- New active run:
  - tag: `paper_full_hard_auto_a12_20260630_203425`
  - profile: `full_ladder_v6_hard_gap_brake_guard`
  - pipeline PID: `3224761`
  - supervisor PID: `3224759`
  - current stage: `off/chunk1`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_full_hard_auto_a12_20260630_203425.log`
- This is the first run that actually uses the new hard-tail `d_crit` brake guard.

2026-06-30 20:39 attempt 12 off passed:

- `off`: PASS after 100 iterations.
- Metrics:
  - `speed=0.0636`
  - `gap=0.1687`
  - `lat=0.0120`
  - `center=0.0151`
  - `min_gap=1.4991`
  - `reset=0`
  - `collision=0`
- Current active stage is `light/chunk1`.

2026-06-30 20:43 attempt 12 light passed:

- `light`: PASS after 100 iterations.
- Metrics:
  - `speed=0.0631`
  - `gap=0.1797`
  - `lat=0.0106`
  - `center=0.0144`
  - `min_gap=1.4975`
  - `reset=0`
  - `collision=0`
- Current active stage is `easy_0/chunk1`.

2026-06-30 21:17 attempt 12 progressed into hard ladder:

- Attempt/tag: `paper_full_hard_auto_a12_20260630_203425`.
- Supervisor/monitor are still running:
  - supervisor PID: `3224759`
  - pipeline PID: `3224761`
  - active train stage: `hard_a/chunk1`
- Passed stages so far:
  - `off`: speed `0.0636`, gap `0.1687`, lat `0.0120`, center `0.0151`, min_gap `1.4991`, reset/collision `0`.
  - `light`: speed `0.0631`, gap `0.1797`, lat `0.0106`, center `0.0144`, min_gap `1.4975`, reset/collision `0`.
  - `easy_0`: speed `0.0733`, gap `0.2105`, lat `0.0161`, center `0.0209`, min_gap `1.4698`, reset/collision `0`.
  - `easy_a`: speed `0.0700`, gap `0.2229`, lat `0.0300`, center `0.0261`, min_gap `1.4314`, reset/collision `0`.
  - `easy_b`: speed `0.0825`, gap `0.2323`, lat `0.0738`, center `0.0483`, min_gap `1.3924`, reset/collision `0`.
  - `med_a`: speed `0.0971`, gap `0.2353`, lat `0.1037`, center `0.0657`, min_gap `1.3727`, reset/collision `0`.
  - `med_b`: speed `0.1073`, gap `0.2395`, lat `0.1144`, center `0.0749`, min_gap `1.3483`, reset/collision `0`.
- `hard_a` active attack strength:
  - `attack_level=hard`
  - `max_fdi_pos=2.75`
  - `max_fdi_acc=0.75`
  - `max_dos_rate=0.12`
- `hard_a` live metrics after 45 rows:
  - `speed_error_abs_mean=0.1354`
  - `gap_error_abs_mean=0.2244`
  - `lateral_error_abs_mean=0.1287`
  - `centerline_error_abs_mean=0.0806`
  - `min_pair_gap_mean=1.3697`
  - `reset=0`
  - `collision=0`
  - `critic_grad_norm=18.1589`
- Current conclusion: this v6 run is healthy through `med_b` and early `hard_a`; the real check remains the later hard-tail stages (`hard_d0` onward), where v5 repeatedly failed from min-gap collapse near `hard_e0`.

2026-06-30 21:25 attempt 12 hard_a passed and hard_b started:

- `hard_a`: PASS after 150 iterations.
- Metrics:
  - `speed=0.1394`
  - `gap=0.2439`
  - `lat=0.1508`
  - `center=0.0977`
  - `min_gap=1.2940`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_b/chunk1`
  - `attack_level=hard`
  - `max_fdi_pos=4.00`
  - `max_fdi_acc=1.30`
  - `max_dos_rate=0.18`
- Monitoring status:
  - 5-minute monitor loop is active.
  - supervisor is active and checking every 300 seconds.
- Current conclusion: ladder has entered hard attack successfully; `min_gap` is decreasing as attack increases, so the critical risk remains later hard stages where prior runs collapsed around `hard_e0`.

2026-06-30 21:30 attempt 12 hard_b passed and hard_c started:

- `hard_b`: PASS after 150 iterations.
- Metrics:
  - `speed=0.2045`
  - `gap=0.2461`
  - `lat=0.1800`
  - `center=0.1046`
  - `min_gap=1.2507`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_c/chunk1`
  - `attack_level=hard`
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
- Monitor snapshot during `hard_b` showed `min_pair_gap_mean=1.2813`, `speed_error_abs_mean=0.1925`, no reset/collision.
- Current conclusion: progression is still valid, but `min_gap` has dropped from `1.2940` at `hard_a` to `1.2507` at `hard_b`; the next stages are close to the historical failure region, so hard-tail checks remain critical.

2026-06-30 21:35 attempt 12 hard_c passed and hard_d0 started:

- `hard_c`: PASS after 150 iterations.
- Metrics:
  - `speed=0.2668`
  - `gap=0.2774`
  - `lat=0.1653`
  - `center=0.0913`
  - `min_gap=1.1098`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_d0/chunk1`
  - `attack_level=hard`
  - `max_fdi_pos=6.20`
  - `max_fdi_acc=2.28`
  - `max_dos_rate=0.245`
- `hard_d0` is the first hard-tail stage using additional gap guard overrides:
  - `d_crit=0.95`
  - `d_drop=1.60`
  - `catchup_action=-0.325`
  - `forward_bias_gain=0.07`
  - `forward_bias_clip=0.022`
  - `forward_bias_min_gap=1.25`
- Current conclusion: the ladder has reached the historical failure zone. `hard_c` still passed, but `min_gap=1.1098` is close to the hard validation floor, so `hard_d0/hard_d1/hard_e*` will decide whether v6 guard fixes the earlier collapse.

2026-06-30 21:41 attempt 12 hard_d0 passed and hard_d1 started:

- `hard_d0`: PASS after 80 iterations.
- Metrics:
  - `speed=0.2855`
  - `gap=0.1965`
  - `lat=0.1485`
  - `center=0.0648`
  - `min_gap=1.1697`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_d1/chunk1`
  - `attack_level=hard`
  - `max_fdi_pos=6.40`
  - `max_fdi_acc=2.35`
  - `max_dos_rate=0.250`
- `hard_d1` gap guard overrides:
  - `d_crit=1.00`
  - `d_drop=1.70`
  - `catchup_action=-0.300`
  - `forward_bias_gain=0.09`
  - `forward_bias_clip=0.028`
  - `forward_bias_min_gap=1.32`
- Current conclusion: v6 gap guard appears effective at `hard_d0`; `min_gap` improved from `1.1098` at `hard_c` to `1.1697` while attack increased. Continue watching `hard_d1` and `hard_e0/e1`, which are the previous collapse region.

2026-06-30 21:47 attempt 12 stopped at hard_d1 due conservative rollback logic:

- `hard_d1/chunk1`: `CONTINUE`
  - `speed=0.3077`
  - `gap=0.2209`
  - `lat=0.1506`
  - `center=0.0568`
  - `min_gap=1.1320`
  - `reset=0`
  - `collision=0`
- `hard_d1/chunk2`: `CONTINUE`
  - `speed=0.3175`
  - `gap=0.2196`
  - `lat=0.1557`
  - `center=0.0529`
  - `min_gap=1.1033`
  - `reset=0`
  - `collision=0`
- Pipeline stopped with exit 22 because validation score worsened from `3.355302646` to `3.410968762`.
- Important conclusion:
  - This is not a physical collapse: no reset/collision and `min_gap` remains above `1.05`.
  - It is a flow-control issue: rollback protection stopped the whole ladder instead of reverting to the best checkpoint and continuing.
- Code change applied:
  - File: `scripts/tools/run_platoon_adaptive_hard_curriculum.sh`
  - Behavior changed so degraded `CONTINUE` chunks roll back to the preserved best checkpoint and keep training, instead of `exit 22`.
- Next action:
  - Let supervisor launch the next attempt with the patched pipeline.
  - Continue 5-minute monitoring.

2026-06-30 21:49 attempt 13 launched with patched rollback behavior:

- Supervisor detected attempt 12 stopped unfinished at `hard_d1` and launched attempt 13.
- New attempt/tag:
  - `paper_full_hard_auto_a13_20260630_214925`
  - PID: `3235061`
  - log: `/home/cnc/SSD_1T/xzw/IsaacLab-main/train_paper_full_hard_auto_a13_20260630_214925.log`
- Current stage:
  - `off/chunk1`
  - start checkpoint: `2026-06-26_00-10-21_platoon5_city_newmedium_attack_20000/model_best.pt`
- Patched behavior confirmed in active script:
  - degraded `CONTINUE` chunks now log rollback and continue from the preserved best checkpoint.
  - They no longer exit with status 22.
- Current conclusion: attempt 13 is a clean full-ladder rerun with the corrected rollback behavior, so it should preserve complete stage data while avoiding the premature `hard_d1` stop seen in attempt 12.

2026-06-30 21:55 attempt 13 off passed:

- `off`: PASS after 100 iterations.
- Metrics:
  - `speed=0.0636`
  - `gap=0.1687`
  - `lat=0.0120`
  - `center=0.0151`
  - `min_gap=1.4991`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `light/chunk1`
- Supervisor remains active and checks every 300 seconds.
- Current conclusion: attempt 13 started correctly with the patched pipeline; early baseline behavior matches attempt 12.

2026-06-30 22:00 attempt 13 light passed:

- `light`: PASS after 100 iterations.
- Metrics:
  - `speed=0.0631`
  - `gap=0.1797`
  - `lat=0.0106`
  - `center=0.0144`
  - `min_gap=1.4975`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `easy_0/chunk1`
  - `max_fdi_pos=1.00`
  - `max_fdi_acc=0.18`
  - `max_dos_rate=0.035`
- Current conclusion: attempt 13 remains healthy through light; no evidence of regression from the rollback logic patch.

2026-06-30 22:06 attempt 13 easy_0 passed:

- `easy_0`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0733`
  - `gap=0.2105`
  - `lat=0.0161`
  - `center=0.0209`
  - `min_gap=1.4698`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `easy_a/chunk1`
  - `max_fdi_pos=1.25`
  - `max_fdi_acc=0.25`
  - `max_dos_rate=0.05`
- Current conclusion: attempt 13 continues normally through early easy; no training instability observed.

2026-06-30 22:11 attempt 13 easy_a passed:

- `easy_a`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0700`
  - `gap=0.2229`
  - `lat=0.0300`
  - `center=0.0261`
  - `min_gap=1.4314`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `easy_b/chunk1`
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`
- Current conclusion: attempt 13 remains stable; no rollback events yet.

2026-06-30 22:16 attempt 13 easy_b still running:

- Latest completed stage remains `easy_a`.
- Current active stage:
  - `easy_b/chunk1`
  - `max_fdi_pos=1.50`
  - `max_fdi_acc=0.35`
  - `max_dos_rate=0.07`
- Monitor snapshot for active `easy_b`:
  - `speed_error_abs_mean=0.0816`
  - `gap_error_abs_mean=0.2296`
  - `lateral_error_abs_mean=0.0620`
  - `centerline_error_abs_mean=0.0422`
  - `min_pair_gap_mean=1.4042`
  - `reset=0`
  - `collision=0`
- Current conclusion: `easy_b` is healthy and near completion; wait for next assessment before marking the stage passed.

2026-06-30 22:22 attempt 13 easy_b passed and med_a running:

- `easy_b`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0825`
  - `gap=0.2323`
  - `lat=0.0738`
  - `center=0.0483`
  - `min_gap=1.3924`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `med_a/chunk1`
  - `max_fdi_pos=1.75`
  - `max_fdi_acc=0.45`
  - `max_dos_rate=0.09`
- Monitor snapshot for active `med_a`:
  - `speed_error_abs_mean=0.0953`
  - `gap_error_abs_mean=0.2322`
  - `lateral_error_abs_mean=0.0938`
  - `centerline_error_abs_mean=0.0582`
  - `min_pair_gap_mean=1.3854`
  - `reset=0`
  - `collision=0`
- Current conclusion: attempt 13 has crossed the easy ladder and entered medium cleanly.

2026-06-30 22:27 attempt 13 med_a passed and med_b running:

- `med_a`: PASS after 150 iterations.
- Metrics:
  - `speed=0.0971`
  - `gap=0.2353`
  - `lat=0.1037`
  - `center=0.0657`
  - `min_gap=1.3727`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `med_b/chunk1`
  - `max_fdi_pos=2.00`
  - `max_fdi_acc=0.50`
  - `max_dos_rate=0.10`
- Monitor snapshot for active `med_b`:
  - `speed_error_abs_mean=0.1030`
  - `gap_error_abs_mean=0.2282`
  - `lateral_error_abs_mean=0.0899`
  - `centerline_error_abs_mean=0.0565`
  - `min_pair_gap_mean=1.3906`
  - `reset=0`
  - `collision=0`
- Current conclusion: medium stage remains healthy; next major risk point is the hard ladder.

2026-06-30 22:33 attempt 13 med_b passed and hard_a started:

- `med_b`: PASS after 150 iterations.
- Metrics:
  - `speed=0.1073`
  - `gap=0.2395`
  - `lat=0.1144`
  - `center=0.0749`
  - `min_gap=1.3483`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_a/chunk1`
  - `max_fdi_pos=2.75`
  - `max_fdi_acc=0.75`
  - `max_dos_rate=0.12`
- Early monitor snapshot for `hard_a`:
  - `speed_error_abs_mean=0.1218`
  - `gap_error_abs_mean=0.2107`
  - `lateral_error_abs_mean=0.0497`
  - `centerline_error_abs_mean=0.0179`
  - `min_pair_gap_mean=1.4292`
  - `reset=0`
  - `collision=0`
- Current conclusion: hard ladder has started cleanly. Continue watching progression toward `hard_d1`, where attempt 12 stopped due workflow rollback rather than physical collapse.

2026-06-30 22:38 attempt 13 hard_a passed and hard_b running:

- `hard_a`: PASS after 150 iterations.
- Metrics:
  - `speed=0.1394`
  - `gap=0.2439`
  - `lat=0.1508`
  - `center=0.0977`
  - `min_gap=1.2940`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_b/chunk1`
  - `max_fdi_pos=4.00`
  - `max_fdi_acc=1.30`
  - `max_dos_rate=0.18`
- Monitor snapshot for late `hard_a`:
  - `speed_error_abs_mean=0.1357`
  - `gap_error_abs_mean=0.2357`
  - `lateral_error_abs_mean=0.1455`
  - `centerline_error_abs_mean=0.0942`
  - `min_pair_gap_mean=1.3261`
  - `reset=0`
  - `collision=0`
- Current conclusion: attempt 13 hard entry is healthy and consistent with attempt 12.

2026-06-30 22:44 attempt 13 hard_b passed and hard_c started:

- `hard_b`: PASS after 150 iterations.
- Metrics:
  - `speed=0.2045`
  - `gap=0.2461`
  - `lat=0.1800`
  - `center=0.1046`
  - `min_gap=1.2507`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_c/chunk1`
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
- Current conclusion: hard progression remains valid; the next stage (`hard_c`) is the last high-strength stage before the added hard-tail gap-guard overrides begin.

2026-06-30 22:50 attempt 13 hard_c still running:

- Latest completed hard stage remains `hard_b`.
- Current active stage:
  - `hard_c/chunk1`
  - `max_fdi_pos=6.00`
  - `max_fdi_acc=2.20`
  - `max_dos_rate=0.24`
- Monitor snapshot for active `hard_c`:
  - `speed_error_abs_mean=0.2484`
  - `gap_error_abs_mean=0.2449`
  - `lateral_error_abs_mean=0.1432`
  - `centerline_error_abs_mean=0.0750`
  - `min_pair_gap_mean=1.2242`
  - `reset=0`
  - `collision=0`
- Current conclusion: `hard_c` appears healthy and above the safety gap floor. Wait for the next assessment; if passed, `hard_d0` will activate the added hard-tail gap guard.

2026-06-30 22:56 attempt 13 hard_c and hard_d0 passed; hard_d1 running:

- `hard_c`: PASS after 150 iterations.
- Metrics:
  - `speed=0.2668`
  - `gap=0.2774`
  - `lat=0.1653`
  - `center=0.0913`
  - `min_gap=1.1098`
  - `reset=0`
  - `collision=0`
- `hard_d0`: PASS after 80 iterations.
- Metrics:
  - `speed=0.2855`
  - `gap=0.1965`
  - `lat=0.1485`
  - `center=0.0648`
  - `min_gap=1.1697`
  - `reset=0`
  - `collision=0`
- Current active stage:
  - `hard_d1/chunk1`
  - `max_fdi_pos=6.40`
  - `max_fdi_acc=2.35`
  - `max_dos_rate=0.250`
  - gap guard: `d_crit=1.00`, `d_drop=1.70`, `catchup_action=-0.300`, `forward_bias_gain=0.09`
- Current conclusion: attempt 13 has reached the exact stage where attempt 12 hit the rollback-flow issue. The pipeline is now patched, so a degraded `CONTINUE` should roll back and continue rather than exit 22.

2026-06-30 23:01 attempt 13 hard_d1 chunk2 running:

- `hard_d1/chunk1`: `CONTINUE`, not a collapse.
- Metrics:
  - `speed=0.3077`
  - `gap=0.2209`
  - `lat=0.1506`
  - `center=0.0568`
  - `min_gap=1.1320`
  - `reset=0`
  - `collision=0`
- Reason for not passing:
  - Speed error is slightly above hard pass threshold (`0.30`).
  - Safety and formation metrics are still acceptable.
- Current active run:
  - `hard_d1/chunk2`
  - continuing from `hard_d1/chunk1/model_final.pt`.
- Current conclusion: this reproduces the attempt 12 hard_d1 behavior before the former exit-22 point. The next check should show whether the patched rollback logic avoids stopping the full ladder.

2026-06-30 23:04 attempt 13 patched rollback confirmed at hard_d1:

- `hard_d1/chunk2`: `CONTINUE`.
- Metrics:
  - `speed=0.3175`
  - `gap=0.2196`
  - `lat=0.1557`
  - `center=0.0529`
  - `min_gap=1.1033`
  - `reset=0`
  - `collision=0`
- Score:
  - best after chunk1: `3.355302646`
  - chunk2: `3.410968762`
- Patched behavior worked:
  - The pipeline logged rollback to preserved best checkpoint.
  - It did not exit with status 22.
  - It launched `hard_d1/chunk3` from `hard_d1/chunk1/model_final.pt`.
- Current active stage:
  - `hard_d1/chunk3`
- Current conclusion:
  - Flow-control issue is fixed.
  - Remaining issue is that `hard_d1` speed error is slightly above the hard pass threshold while safety metrics remain acceptable.
  - If additional chunks cannot reduce speed below threshold before max iters, next modification should target hard-tail speed/safety tradeoff rather than rollback logic.

2026-07-01 01:31 current hard attack upper bound:

- Active supervisor profile: `full_ladder_v7_hard_speed_gap_balance`.
- The final hard upper bound used by the current ladder is:
  - `max_fdi_pos = 8.00`
  - `max_fdi_acc = 3.00`
  - `max_dos_rate = 0.30`
- Intermediate hard stages are still gradual:
  - position: `2.75 -> 4.00 -> 6.00 -> 6.20 -> 6.40 -> 6.55 -> 6.70 -> 6.90 -> 7.15 -> 7.50 -> 8.00`
  - acceleration: `0.75 -> 1.30 -> 2.20 -> 2.28 -> 2.35 -> 2.42 -> 2.50 -> 2.58 -> 2.67 -> 2.84 -> 3.00`
  - DoS rate: `0.12 -> 0.18 -> 0.24 -> 0.245 -> 0.250 -> 0.258 -> 0.265 -> 0.272 -> 0.280 -> 0.290 -> 0.30`
- Current run tag: `paper_full_hard_auto_a18_20260701_012842`.

2026-07-01 01:36 hard stage boundaries and prior failure points:

- Active v7 hard ladder:
  - `hard_a`: pos `2.75`, acc `0.75`, dos `0.12`
  - `hard_b`: pos `4.00`, acc `1.30`, dos `0.18`
  - `hard_c`: pos `6.00`, acc `2.20`, dos `0.24`
  - `hard_d0`: pos `6.20`, acc `2.28`, dos `0.245`
  - `hard_d1`: pos `6.40`, acc `2.35`, dos `0.250`
  - `hard_e0`: pos `6.55`, acc `2.42`, dos `0.258`
  - `hard_e1`: pos `6.70`, acc `2.50`, dos `0.265`
  - `hard_f0`: pos `6.90`, acc `2.58`, dos `0.272`
  - `hard_f1`: pos `7.15`, acc `2.67`, dos `0.280`
  - `hard_g`: pos `7.50`, acc `2.84`, dos `0.290`
  - `hard`: pos `8.00`, acc `3.00`, dos `0.30`
- Hard pass thresholds are: speed error `<=0.30`, gap error `<=0.45`, lateral error `<=0.24`, centerline error `<=0.20`, min gap `>=1.05`, bad-orientation reset `<=0.02`, collision near `0`.
- Prior true instability/unsafe failure:
  - v5 at `hard_e0` (`pos=6.70`, `acc=2.50`, `dos=0.265` in that profile) degraded to `min_gap=0.8713`, below the critical abort floor `0.90`; this was a real safety-gap collapse risk.
- Prior non-unsafe gate failure:
  - v6/v13 reached `hard_d1` (`pos=6.40`, `acc=2.35`, `dos=0.250`) with `speed=0.3175`, `min_gap=1.1033`, reset/collision `0`; it failed the speed threshold but did not physically collapse.
- Current v7 modification specifically targets the `hard_d1` speed/safety tradeoff while preserving the improved hard-tail gap guard.

2026-07-01 01:51 hard target redefined to former hard_b:

- User decided the deeper hard ladder is not necessary; final hard should stop at the former `hard_b` strength.
- New final hard benchmark:
  - `max_fdi_pos = 4.00`
  - `max_fdi_acc = 1.30`
  - `max_dos_rate = 0.18`
- `scripts/tools/supervise_hard_ladder.py` now has `full_ladder_v8_hard_b_final`, whose final stage is named `hard` but uses the former `hard_b` attack strength. This keeps pipeline completion detection (`stage hard passed`) compatible.
- The previously running long v7 ladder (`paper_full_hard_auto_a18_20260701_012842`) was stopped because it was no longer needed under the new benchmark.
- Existing `paper_full_hard_auto_a13_20260630_214925` data is sufficient under this new standard:
  - off/light/easy_0/easy_a/easy_b/med_a/med_b/hard_a/hard all PASS.
  - Final hard result from former `hard_b`: `speed=0.204473`, `gap=0.246123`, `lateral=0.179973`, `centerline=0.104597`, `min_gap=1.250713`, `reset=0`, `collision=0`.
- Packaged data and figures:
  - package root: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package`
  - tarball: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package.tar.gz`
  - generated paper figures: `fig_01_reward_physical.png` through `fig_07_pair_centerline_bars.png`.

2026-07-01 01:58 separated reward-curve figures:

- Added separate algorithm and meta-learning reward plots under:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/fig_08_algorithm_reward_curve.png`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/fig_09_meta_learning_reward_curve.png`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/reward_curves_summary.csv`
- `fig_08` contains HAPPO/environment reward signals and reward/penalty components.
- `fig_09` contains teacher/meta shaping signals plus teacher loss/correlation diagnostics.
- The tar package was rebuilt after adding these files.

2026-07-01 reward curve interpretation:

- A standard RL reward curve is expected to improve mainly when the MDP/task distribution is fixed.
- The current paper package concatenates curriculum stages with increasing attack strength, so the raw reward curve is not expected to be globally monotonic.
- In this run, stage difficulty increases from off/light/easy/medium to hard (`max_fdi_pos=4.0`, `max_fdi_acc=1.3`, `dos=0.18`), so lower raw reward at later stages can be normal even when the controller remains robust.
- For paper clarity, training reward should be described as curriculum-stage performance, and learning progress should be supported by either within-stage curves or fixed-attack evaluation curves using the same evaluation condition across checkpoints.

2026-07-01 02:03 within-stage total reward subplot:

- Added requested multi-subplot figure using only total reward (`reward_env_mean`) for each curriculum stage:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/fig_10_stage_internal_total_reward.png`
  - summary CSV: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/stage_internal_total_reward_summary.csv`
- The package tarball was rebuilt after adding the new figure.
- Initial read: within-stage total reward is mostly flat-to-decreasing in these short gated chunks, especially in `hard`; this suggests the staged data is better interpreted as short robustness adaptation/validation under increasing attack, not as a long fixed-MDP learning curve.

2026-07-01 02:07 reward curve appears visually flat:

- Checked `reward_env_mean` ranges per stage. The field does change, but it is a per-step mean reward with a narrow numeric range, not a cumulative episode return.
- Existing subplot figure used a shared y-axis, which visually compressed the within-stage variation.
- Added a zoomed per-stage figure with independent y-axes:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/fig_11_stage_internal_total_reward_zoomed.png`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/paper_hardb_final_from_a13_package/paper_figures/stage_internal_total_reward_zoomed_summary.csv`
- Important interpretation: these chunks start from an already trained checkpoint and run only 100-150 updates per stage while attack difficulty increases. Therefore they do not show the classic random-policy-to-converged reward climb. Most within-stage smoothed total rewards are flat-to-decreasing; the hard stage drops most clearly.

2026-07-01 why within-stage reward does not improve:

- `reward_env_mean` in `platoon_metrics.csv` is written in `router._collect_platoon_metrics()` as `float(rewards_tensor.mean().item())`; it is an instantaneous per-step mean reward after local shaping, not a cumulative episode return.
- The plotted rows are on-policy rollout samples, not fixed-condition evaluation points. They are affected by episode phase, exploration noise, random commands/attacks, and accumulated tracking error.
- Evidence from the current package: within a stage, `termination_time_out` often goes from about `0.0156` at the first row to `1.0` at the last row, so early rows are near episode start while later rows reflect much later episode states.
- As the rollout advances, shield and penalties grow. Example final hard stage:
  - `reward_env_mean`: `0.2371 -> 0.1046`
  - `speed_error_abs_mean`: `0.1104 -> 0.2108`
  - `lateral_error_abs_mean`: `0.0206 -> 0.1695`
  - `centerline_error_abs_mean`: `0.0017 -> 0.1021`
  - `shield_lateral_rate`: `0.1069 -> 0.9527`
  - `local_reward_shaping_mean`: `-0.0077 -> -0.1181`
- Therefore the current staged CSV is valid for robustness/threshold evidence but is not ideal to show a classic monotonic learning curve. To show learning, run fixed-condition evaluation after periodic checkpoints within each stage and plot cumulative episode return under the same attack setting.

2026-07-01 02:27 fixed-medium three-algorithm comparison setup:

- User requested a fixed `medium` attack comparison under identical eval conditions for:
  - MAPPO baseline;
  - HAPPO without meta-learning;
  - current HAPPO with meta-learning preserved unchanged as the third method.
- Before any comparison edits, current code/assets were backed up locally to:
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/backups/pre_medium_compare_code_assets_20260701_022215.tar`
- The pre-comparison state was committed and pushed to GitHub:
  - branch `freeze/cagan-step3-dualchannel-logging`
  - commit `79b8290 Backup hard-b benchmark and plotting tools before medium comparison`
- Added a MAPPO-like baseline switch by allowing the task-local HAPPO runner to disable HAPPO's sequential importance factor while keeping the same centralized critic/per-agent actor structure.
- Added/updated scripts for fixed-medium comparison:
  - `scripts/tools/run_medium_algorithm_comparison.sh`
  - `scripts/tools/plot_medium_algorithm_comparison.py`
  - `scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py`
- Static checks passed:
  - `python3 -m py_compile ...`
  - `bash -n scripts/tools/run_medium_algorithm_comparison.sh`
- No active training/eval process was detected before starting the new sanity run.
- First tiny sanity showed that MAPPO and HAPPO-no-meta branches can train/save/evaluate, but eval return printed as `0.000` because `reward_env_mean` is zero in this task-local HAPPO eval path while reward-manager terms are populated.
- Fixed `scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py` so fixed-medium episode return is computed from the logged `reward_*` reward-manager terms (`eval_total_reward_mean`) instead of relying only on outer `reward_env_mean`.
- Re-ran static checks after the fix; `py_compile` and `bash -n` still pass.
- Second tiny sanity passed end-to-end:
  - package root: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/medium_compare_sanity2_20260701_023212_package`
  - tarball: `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/medium_compare_sanity2_20260701_023212_package.tar.gz`
  - generated fixed-medium comparison figures: `fig_01_fixed_medium_episode_return.png`, `fig_02_fixed_medium_eval_metrics.png`, `fig_03_fixed_medium_final_bars.png`
  - all three eval summaries show `attack_enabled=1`, `attack_max_fdi_acc=0.5`, `attack_max_dos_rate=0.1`, and nonzero `episode_return_mean`.
- Added a final plotting fix so `model_final.pt` is placed at the true `MAX_ITERATIONS` coordinate rather than reusing the last selected intermediate checkpoint coordinate.
- The fixed-medium comparison pipeline code was committed and pushed:
  - branch `freeze/cagan-step3-dualchannel-logging`
  - commit `888bcf9 Add fixed-medium MAPPO HAPPO comparison pipeline`
