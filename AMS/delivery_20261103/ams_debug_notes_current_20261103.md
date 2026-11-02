# AMS Integration Debug Notes

## 2026-08-09 下一步实施状态（23:27）

- 复核 BC checkpoint：其 runner 配置本来就是 `share_actor=true`，5 个 actor state 完全相同；此前“BC 种子与正式 shared actor 架构不同”的推断已纠正。
- 找到训练退化的直接实现问题：retention BC/KL 原先在每个 PPO minibatch 内执行；shared-actor HAPPO 又按 5 个 agent 顺序更新，同一个 actor 一轮可能执行约 80 次 retention optimizer step。现改为每次 runner update、每个唯一 actor 只执行一次 retention，并把诊断回填到对应 agent info。
- 修复 clean 启动语义：`run_medium_algorithm_comparison.sh` 新增 `ENABLE_ATTACK`，为 false 时既设置 `enable_attack=false`，也不传 `--enable_attack_eval`。`run_ackermann_progressive_clean.sh` 现在显式 `ENABLE_ATTACK=false`，AMS warmup/ramp 默认 10/40，加入真实轨迹 retention，clean gate 默认要求 100%。
- clean gate 已改成严格联合条件：全车瞬时完成峰值=1、持续完成峰值=1、最终 ever completion=1、collision=0、road_boundary_any=0、road-boundary termination=0、attack_enabled=0。
- 原始 BC clean 基线两次启动均因隔离 CA-GAN 工作流随后开启同 GPU Isaac 而主动中止，未采信中断数据。中断前 440 步显示 BC leader 速度约 0.36–0.42 m/s，接近专家，支持“后续更新导致退化”。
- 当前外部 M5 64-env/300-update 正式训练占用 Isaac。已启用 `ackermann-ams-clean-idle.timer`：每 10 分钟检查，必须连续两次无任何 Isaac train/play/eval 才运行 `run_ackermann_bc_clean_gate_when_idle.sh` 的 3500-step、16-env、无攻击/无专家 BC clean 基线；不会自动进入攻击阶段。
- 新改动已通过 `py_compile`、`bash -n`、`git diff --check` 和单步 retention 数值测试。

## 2026-08-09 AMS 未通过的根因复核

- 首要结论：8 月 8 日 50-update retention recovery 不能作为“AMS 算法失败”的证据。虽然 `ams_enabled=1` 且辅助 Q loss 从约 0.086 降至 0.010，但全部 50 行的 `ams_advantage_weight` 都是 0；配置恰为 warmup=50、ramp=100，所以 AMS 局部 Q 优势从未进入 actor 更新。
- warm-start 架构不一致：2-env 冒烟使用 separate actors 时首轮 BC loss 约 0.000636；正式恢复使用 Meta 默认 shared actor，首轮 BC loss 约 0.0727，最终仍约 0.0292，reference KL 从约 0.0518 升到 0.2483。说明 BC seed 对 shared actor 并非可直接保持的同一策略起点，恢复阶段先丢失了专家前进能力。
- 专家数据本身可完成任务：44 秒轨迹中五车最终 reference_progress 都是 1.0，平均 vx 约 0.378–0.379 m/s，纵向 executed action 恒为 0.384。失败候选评估平均 leader/platoon speed 仅 0.155/0.171 m/s，远低于专家，导致进度从首车到末车递减，末车几乎为 0。
- 训练顺序不符合既定门控：该恢复块一开始就是 50% clean + 50% light，并在 50 轮内把攻击爬到 full light；clean 严格全程评估未先通过（且后续 clean eval 因包装脚本 127 提前中止而未执行）。因此基础控制尚未确认就加入了攻击。
- 评估还存在累计边界重置：`termination_road_boundary_exceeded` 汇总约 0.792，反复 reset 会清空空间路径进度；瞬时 `road_boundary_any_rate` 均值虽仅约 0.0076，不能据此认为整段没有越界失败。
- 正确判断：当前失败是“AMS 实际权重为零 + shared-actor warm-start 不匹配 + 未先通过 clean + 低速/边界重置”的组合问题，不是经过有效 AMS 训练后仍无法变道。

## 2026-08-09 实验归属澄清

- 昨晚 00:50–00:57 的 `m4_progressive_cagan_pilot_20260809_retry1` **不是 AMS 实验**。
- 实际配置证据：六阶段均为 `algorithm: happo`；非 off 阶段 `attack_mode: cagan`、`enable_attack: true`，环境配置中没有 `happo_ams_enabled` 字段；报告记录分支为隔离的 `agent/cagan-progressive-hard-v2`。
- 该实验验证的是渐进 CA-GAN 攻击器从 off 到 Hard-B 的状态继承、冻结预算和顺序独立性。不能把它写成 AMS 训练结果，也不能用于回答 AMS 是否完成全车变道或是否优于六算法。
- 真正的 AMS/Ackermann 最新实验仍是 2026-08-08 的 retention recovery，其中 AMS 指标启用，但严格 light 全车完成率为 0，尚未通过门控。

## 2026-08-09 当前状态复核

- 当前没有 Isaac 训练/评估进程，也没有每小时监控 timer 在运行。
- Ackermann 全车变道主线没有新晋级模型：最新严格 light 3500 步候选仍是 2026-08-08 恢复模型，全车完成率 0，六算法全攻击矩阵仍未获准启动。
- 另一个隔离的 CA-GAN 重构工作流昨晚完成了 M4 六阶段端到端 pilot：off→light→easy→medium→hard-A→hard-B，每阶段仅 1 环境×4 iterations，总计 24 iterations；Hard-B 冻结攻击器的预算、状态继承和 AB/BA 顺序独立性审计通过。
- M4 明确只是攻击管线集成证据，不是车辆控制性能结果：冻结审计仅 40 steps，不能证明完整变道，也不能作为六算法科学对比。其下一模块是 M5 多种子因果消融，但在 Ackermann clean/light 全程门控未通过前，不把它等同于用户要求的全攻击强度完整实验。

## 2026-08-08 当前状态复核

- 当前没有 Isaac 训练、评估或 Ackermann 恢复脚本在运行；六算法攻击矩阵仍被门控阻止。
- 最新有效候选仍是 `ackermann_retention_recovery_20260808_1658` 的 `model_final.pt`。light 3500 步结果：return 35935.947、collision 0、road_boundary_any 0.007607，但全车完成率与持续完成率均为 0，末车最终路径进度为 0。
- 分车进度均值呈明显递减：robot 1–5 分别约 0.259、0.150、0.0566、0.0113、0.00001；说明空间路径传播/纵向跟随没有把同一路径可靠传递到末车，当前不是“时间不够”，而是后车基本没有进入变道路径。
- 代码修复与训练/评估产物均保留在工作区，尚未提交；下一项实施工作仍是修复共享 actor 下的前进动作方向和队尾路径推进，再先跑 clean 全程门控。

## 2026-08-08 17:14 — 策略一致性修复、保留训练加入及首轮门控结果

- 已停止旧的 `ackermann-bc-six-monitor-20260804.timer`，旧结果保留；在 clean/light 门控通过前不恢复六算法攻击矩阵。
- 根因修复：策略用受攻击观测采样动作与 old log-prob，但 rollout 曾存 clean 观测，导致 PPO/HAPPO 更新时在不同观测上重算 log-prob。现缓存并写入“实际采样所用观测”，并对 transition 做逐元素一致性断言；raw policy action 与其 old log-prob 继续成对保存，攻击/安全盾作为环境侧变换。
- 加入真实轨迹保留训练：从 `logs/ackermann_expert_smoke/evidence/trajectory.csv` 重建 10995 条 11D 观测/2D Ackermann 动作，训练中加入 BC + 冻结参考策略 KL；检查点保存参考 actor，旧 BC 检查点加载后自动建立参考。
- 加入渐进攻击：固定比例 clean 并行环境、攻击强度爬坡。修复爬坡误用低频 attacker 更新计数的问题，改按 student update；修复无 attacker-update 时 clean fraction/strength 日志回退。
- 离线验证通过：观测缓存与 old/new log-prob 一致（1e-6），clean_fraction=1 时扰动严格为零，保留更新使人工 MSE 从 0.015347 降到 0.012375；`py_compile`、`bash -n`、`git diff --check` 通过。
- 真实 Isaac 2 轮回归通过：`2026-08-08_16-57-00_ackermann_retention_ramp_smoke_20260808`；strength 0.5→1.0，clean fraction=0.5，ratio 1.000893→0.999431，BC/KL 有限，生成 `model_final.pt`。
- 50 轮恢复块：`logs/ackermann_retention_recovery/ackermann_retention_recovery_20260808_1658`。训练完成且数值稳定；最终 light 完整 3500 步评估：return=35935.947、collision=0、road_out=0.008，但 `whole_done=0`、`min_progress=0`，未达到完成门槛，因此检查点不晋级、不进入六算法矩阵。失败已保留 CSV 与 3 张比较图。
- 新增阶段选择器 `scripts/tools/select_ackermann_stage_checkpoint.py` 与恢复入口 `scripts/tools/run_ackermann_retention_recovery.sh`。修复比较脚本把 `model_0` 当周期检查点的问题（以后排除 iteration=0），绘图改用 host `python3`，避免 Isaac launcher 在图已生成后返回 127。
- 下一步：针对“前进/路径进度不足”调整 warm-start 的共享 actor/动作方向与推进保持；先重新做 clean 全程评估，clean 完成率和持续完成率均为 1 且零碰撞/零越界后，再做 light 恢复门控。

## Workstream Boundary

- This file is the only progress log for the Action Manifold Smoothing (AMS) integration and its new 3000-iteration experiments.
- Meta Reward Teacher work continues to use `/home/cnc/SSD_1T/xzw/IsaacLab-main/debug_notes.md`.
- Do not write AMS conclusions into `debug_notes.md`, and do not write Meta-only conclusions into this file.

## Current Goal

Integrate the useful ideas from "Action Manifold Smoothing: A Lipschitz Pathway Perspective on High-Dimensional Reinforcement Learning" into the current IsaacLab platoon simulation, train a new 3000-iteration candidate, repeatedly evaluate it under the same fixed protocol, and determine whether it outperforms the existing HAPPO+meta reference.

## Source Material

- Poster image: `/home/cnc/Desktop/微信图片_20260622110047_596_28.jpg`
- Poster title: `Action Manifold Smoothing: A Lipschitz Pathway Perspective on High-Dimensional Reinforcement Learning`
- Poster project URL: `https://github.com/Lin-Zhihao98/AMS`
- Main visible idea: replace pointwise TD targets with averages over orthogonally sampled action-neighborhood directions, using QR-derived directions, neighborhood radius epsilon, and K samples to smooth the learned action-value landscape.

## Baseline To Beat

- Existing comparison target: the completed HAPPO+meta 3000-iteration/final result recorded in `IsaacLab-main/debug_notes.md`.
- Medium fixed-evaluation reference final return: `18407.0382`, with speed error `0.04401`, gap error `0.17268`, centerline error `0.00759`, lateral error `0.00642`, minimum gap `1.51020`, collision `0`, and bad-orientation reset `0`.
- A fair AMS comparison must use the same attack profile, command distribution, checkpoint order, evaluation length, seed policy, and shield configuration as the selected reference protocol.
- Primary comparison is now fixed to the hard-b protocol used by the completed 3000-iteration HAPPO+meta run, not the Medium reference above. Its final 1000-step return is `18262.624`, and its peak is `18682.905` at iteration 600. The Medium numbers remain historical context only and must not be used for the final AMS-vs-Meta claim.

## Initial Constraints

- Back up the current working state before AMS edits.
- Publish the intended current code state to GitHub with an explicit commit scope; do not add logs, model checkpoints, simulator assets, or unrelated untracked files.
- Preserve current Meta behavior by keeping AMS default-disabled and activating it only in the new experiment configuration.
- Avoid concurrent Isaac training/evaluation processes because prior concurrent runs caused allocator corruption.

## Next Steps

1. Inspect the official AMS paper/repository and map its exact target-smoothing algorithm onto the task-local HAPPO critic update.
2. Audit the current git diff and create a reproducible pre-AMS backup and GitHub commit.
3. Implement AMS behind default-neutral configuration flags, with diagnostics and tests.
4. Run syntax/unit/smoke checks, then launch a 3000-iteration AMS experiment.
5. Evaluate checkpoints sequentially under a protocol directly comparable with HAPPO+meta.
6. Iterate only when evaluation evidence identifies a concrete failure; record every important conclusion here.

## 2026-07-16 Initial Status

- The poster was located and visually inspected successfully.
- The repository is currently on `freeze/cagan-step3-dualchannel-logging` at `b6a2c5f`, tracking the same remote branch.
- GitHub CLI is authenticated as `hs997`; origin is `git@github.com:hs997/densact.git`.
- The worktree contains many existing tracked modifications and many unrelated untracked repository/assets files, so backup/publish must use an explicit allowlist rather than `git add -A`.

## 2026-07-16 Pre-AMS Backup And GitHub Baseline

- Created and verified three pre-AMS backups under `IsaacLab-main/backups/pre_ams_20260716_053942_*`:
  - complete Git history bundle (`9.9 MB`), verified with `git bundle verify`;
  - binary tracked-diff patch (`349 KB`);
  - selected current-worktree snapshot (`213 KB`), including this AMS log.
- SHA-256 verification passed for all three backup artifacts; checksums are stored in `pre_ams_20260716_053942_SHA256SUMS.txt`.
- Python compilation passed for the selected current source/tools, and `bash -n` passed for the comparison/evaluation shell scripts.
- Published the pre-AMS baseline to GitHub commit `4c404ae` (`Back up completed HAPPO meta work before AMS`) on `freeze/cagan-step3-dualchannel-logging`.
- Existing GitHub PR `hs997/densact#1` now contains the baseline commit. Unrelated untracked files, logs, simulator assets, model checkpoints, ROS tools, and backup artifacts were not staged.

## 2026-07-16 Official AMS Audit And Adaptation Decision

- Inspected official repository `Lin-Zhihao98/AMS` at commit `125f74f811f8f36c9fc6841c9d833db2cf256c29`.
- Official AMS uses an action-conditioned Q critic. It generates a random orthonormal action basis using `torch.linalg.qr`, samples K directions, evaluates bounded action neighbors around the target-policy action, and averages their target Q values in the Bellman backup. The SAC implementation uses explicit symmetric `+epsilon/-epsilon` pairs.
- Current task-local HAPPO uses a centralized state-value critic `V(share_obs)` with GAE; it has no action input. Directly pasting the TD3/SAC target code would therefore be mathematically invalid and would not influence the HAPPO actor.
- Planned on-policy adaptation: add an optional centralized action-conditioned auxiliary Q critic over `(share_obs, joint_action)`, train it with symmetric QR action-neighborhood Bellman targets and a Polyak target network, then blend its normalized local action advantage with the existing normalized GAE advantage. Keep all AMS behavior default-disabled so old Meta checkpoints/configurations remain compatible.
- Required AMS diagnostics: auxiliary Q loss/gradient norm, anchor and neighborhood Q means, neighborhood variance, local Q advantage mean/std, effective K/radius, and final blended-advantage weight.

## 2026-07-16 Core Integration In Progress

- Added a default-disabled twin auxiliary Q branch over centralized state and the clipped 20-dimensional joint action. The initial 10-dimensional estimate assumed two controls per vehicle; the real Isaac task reports four wheel-action dimensions for each of five vehicles, and Smoke 1 confirmed `act_dim=4`, `joint_action_dim=20`.
- The auxiliary Bellman target averages symmetric `+epsilon/-epsilon` neighbors along QR-orthogonal directions; the actor receives the normalized joint Q advantage only after a configurable warm-up and ramp, blended with the original normalized GAE.
- Stored policy actions are clipped to the wrapper action limit before Q training, so the auxiliary critic learns the bounded commands proposed by the policy rather than unbounded Gaussian samples. Attack and Shield transformations remain part of the environment transition; these Q inputs are not claimed to be the final post-Shield wheel commands.
- AMS network, optimizer, target networks, QR basis/call count, and runner update count are included in checkpoints. AMS diagnostics are preserved through the Student summary path.
- Added default-neutral environment/Hydra fields for all AMS hyperparameters. Router CSV output now exposes enable state, Q loss/gradient norm, anchor/target/neighborhood Q, neighborhood variance, local advantage statistics, effective K/radius, and the ramped actor-advantage weight.
- Added the `happo_meta_ams` pipeline label and a dedicated `scripts/tools/run_platoon_ams_hardb_3000.sh` launcher. It pins the attack, Meta Teacher, reward, action, command, and Shield values to the completed hard-b HAPPO+meta manifest and runs training/evaluation serially; the only experimental addition is AMS.

## 2026-07-16 Static And Numerical Verification

- Python compilation passed for AMS, runner, Student, router, both task config definitions, and the comparison plotter. `bash -n` passed for the generic pipeline and dedicated hard-b launcher; `git diff --check` is clean.
- A CPU numerical test passed for QR orthogonality, a complete rollout/Q/HAPPO update, finite AMS diagnostics, effective advantage blending, AMS checkpoint round-trip including the QR basis, and the default-disabled checkpoint path.
- The first numerical attempt used a two-dimensional reward fixture while the real buffer requires `[env, agent, 1]`; this was a test-fixture error, not an implementation failure. Re-running with the real shape passed.

## 2026-07-16 IsaacLab Smoke 1

- Completed a real GPU smoke run with 4 environments and 3 training iterations under the full hard-b Meta+AMS configuration: `logs/rsl_rl/platoon_happo/2026-07-16_05-53-30_ams_smoke_20260716_1_happo_meta_ams`.
- Hydra accepted every new AMS override. Meta Teacher, hard-b attacker, safety shield, HAPPO and AMS ran together without a CUDA error or non-finite update.
- All 12 AMS columns were written to `platoon_metrics.csv`. Across the three updates, Q loss was `0.0696`, `0.0719`, `0.0433`; local-advantage standard deviation stayed near `0.017-0.018`; neighborhood variance was small but non-zero; configured `K=8`, radius `0.25`, and smoke-test weight `0.10` were recorded correctly.
- AMS learning took about `0.10 s` inside each roughly `2.3 s` training iteration after startup. The package and tar archive were produced successfully and contain `ams_debug_notes.md`, not the Meta-only log.
- This first smoke intentionally skipped evaluation. A second end-to-end smoke must verify loading the AMS checkpoint in a fresh simulator process and plotting the new label.

## 2026-07-16 IsaacLab Smoke 2

- Completed a fresh-process train-load-evaluate-package smoke at `logs/rsl_rl/platoon_happo/ams_smoke_20260716_2_package` with exit status 0.
- The evaluator rebuilt the 20-dimensional HAPPO+AMS runner, loaded `model_0.pt` and `model_final.pt` without missing-state or shape errors, and evaluated both under enabled hard-b attack (`FDI acc=1.3`, `DoS=0.18`).
- It produced both step CSVs, `eval_summary.csv`, a combined summary, and all three comparison figures. The plotting path recognizes `happo_meta_ams`, and the package again contains `ams_debug_notes.md`.
- The one-iteration return is only a load-path smoke result and is not evidence for or against AMS performance. No performance comparison will be made before the full 3000-iteration fixed-protocol evaluation.

## 2026-07-16 Pre-Launch Review

- Code-path review confirmed that AMS should smooth the bounded policy command while the attack/Shield projection remains part of the transition dynamics. Replacing AMS inputs with post-Shield commands would make the auxiliary action inconsistent with the actor output being optimized.
- Unified the default and experiment AMS advantage weight at the conservative value `0.10`; the previous `0.15` code default was inactive but inconsistent with the launcher.
- Added validation for dimensions, action limit, radius, discount, Polyak coefficient, learning rate, and target clip. Runner startup now prints AMS enable state, K, radius, and weight for auditability.

## 2026-07-16 Evaluation Protocol Correction

- A manifest/CSV audit found that the selected Meta reference (`final=18262.624`, `peak=18682.905`) is the `latpair_v3` deployment-Shield re-evaluation, not the original hard-b Shield evaluation (`final=18286.980`, `peak=18698.026`). The trained checkpoint is the same, but deployment Shield gains differ.
- The AMS training launcher already matches all 30 pinned attack, command, Meta Teacher, reward, action and base-Shield fields from the original hard-b manifest. For a fair comparison against the user's selected `latpair_v3` numbers, AMS checkpoints must also be evaluated with the exact `latpair_v3` deployment override.
- Based on the Meta log, the candidate override is: lateral gain `0.44`, first-follower lateral scale `0.80`, pair-2 lateral scale `1.25`, centerline gain `1.15`, and centerline clip `0.24`, with other base parameters unchanged. This must be reproduced on the old Meta final checkpoint before it is accepted as the AMS evaluation protocol.
- Evaluating only the old `model_final.pt` from a fresh environment produced `10914.704`, confirming that it cannot be compared with the published sequence-final row: the historical protocol used `FRESH_ENV_PER_CHECKPOINT=0`, so the final checkpoint inherits simulator/command state after ten earlier checkpoint evaluations.
- A fresh `model_0.pt` evaluation under the inferred v3 settings reproduced the sequence's first point closely (`2857.767` versus `2860.183`, a `-0.084%` difference). Leader and command metrics matched, while small follower lateral differences suggest v3 also inherited v1's `lateral_turn_clip=0.09` and `lateral_velocity_gain=0.12`. One more model-0 reproduction with those values is required.
- Adding the two inherited v1 values improved reproduction to `2858.652` versus `2860.183` (`-0.054%`), centerline absolute-error difference `2.9e-5`, lateral absolute-error difference `9.2e-5`, and Shield lateral-rate difference `4.3e-4`. This close match confirms the full v3 override; residual drift is consistent with the Linux kernel/Isaac physics runtime having changed since July 3.
- The dedicated launcher now applies the complete v3 override only through `HAPPO_META_EVAL_EXTRA_OVERRIDES`. Training retains the original hard-b base Shield, exactly matching the Meta checkpoint's training protocol.

## 2026-07-16 Pre-Commit Verification

- Re-ran Python compilation, shell syntax checks, and `git diff --check`; all passed after the v3 protocol correction.
- Re-ran the numerical test with explicit invalid-config rejection, QR orthogonality, one full Q/HAPPO update, finite diagnostics, weight activation, and checkpoint round-trip; all passed.
- No Isaac training or evaluation process remains. The implementation is ready for an explicit allowlist commit and GitHub push before the long run.

## 2026-07-16 AMS GitHub Snapshot

- Committed the 10-file AMS implementation and experiment pipeline as `a9ba8be` (`Integrate action manifold smoothing into HAPPO`) on `agent/ams-platoon-smoothing`.
- Pushed the branch to `origin/agent/ams-platoon-smoothing` and opened draft PR `hs997/densact#2`: `https://github.com/hs997/densact/pull/2`.
- The commit uses an explicit allowlist. Smoke logs, model checkpoints, simulator assets, backups, adjacent progress notes, and unrelated untracked files were not added to Git.

## 2026-07-16 Formal 3000-Iteration Run

- Launched tag `hardb_ams_20260716_060808`; training directory is `logs/rsl_rl/platoon_happo/2026-07-16_06-08-41_hardb_ams_20260716_060808_happo_meta_ams` and pipeline log is `train_hardb_ams_20260716_060808.log`.
- Startup audit prints `ams=1`, `K=8`, radius `0.25`, weight `0.10`, 64 environments, 20-dimensional joint action, and hard-b attack values.
- First four updates are healthy: AMS Q losses are `0.0546`, `0.0701`, `0.0511`, `0.0418`; no non-finite/CUDA error is present. Effective AMS actor weight correctly remains zero during the first 50-update warm-up.
- Post-startup iterations take about `2.5 s`, giving a current training ETA near 2 hours 10 minutes before sequential evaluation.
- Warm-up/ramp audit passed in the formal run. The effective actor weight stayed `0.000` through update 50, then increased exactly by `0.001` per update (`0.004` at update 54 through `0.015` at update 65), targeting `0.10` at update 150. All AMS columns remain finite; Q loss is around `0.031-0.036` and reset-on-bad-orientation remains zero at this stage.
- At update 122 the ramped weight is `0.072`, Q loss `0.0432`, local-advantage std `0.0113`, neighborhood variance `1.79e-6`, and every AMS value is finite. Bad-orientation reset remains zero; no runtime error is present.
- Full AMS weight `0.10` was reached at update 150 without a discontinuity. Updates 156-167 keep finite Q loss (`0.0439-0.0537`), local-advantage std (`0.0090-0.0122`), and small non-zero neighborhood variance; bad-orientation reset remains zero.
- Checkpoint `model_300.pt` was saved successfully. Around update 320, auxiliary anchor/target Q means match closely (`0.640/0.637`), Q loss is `0.0927`, local-advantage std `0.0116`, all AMS values are finite, and reset-on-bad-orientation remains zero.
- The latest 20 training rows show higher environment reward and lower speed/lateral error than the corresponding early Meta rows, but higher gap/centerline error and slightly higher physical cost. These are on-policy training diagnostics under a newer runtime, not fixed evaluation, so no superiority claim is made from them.
- Checkpoint `model_600.pt` was saved. At update 624, anchor/target Q means remain aligned (`1.236/1.241`), local-advantage std is `0.0082`, neighborhood variance `1.02e-5`, and no AMS value is non-finite. Raw Q loss rose with Q scale to `0.169`, while gradients are clipped by the configured `0.5` threshold.
- Window 581-600 versus the historical Meta training log: AMS has higher reward (`0.2375` vs `0.2219`) and lower speed error (`0.1748` vs `0.1899`), but higher gap/lateral/centerline errors and physical cost. Continue without intervention because Q target tracking and runtime stability remain healthy; fixed checkpoint evaluation is still decisive.
- Checkpoint `model_900.pt` was saved. At update 927, anchor/target Q are still tightly aligned (`1.834/1.838`); local-advantage std is stable at `0.0083`, neighborhood variance `1.05e-5`, and all values remain finite. Raw Q loss is `0.233` as the learned Q scale grows, with no evidence of target separation.
- Window 881-900 preserves the prior tradeoff relative to Meta: reward and speed error are better, while gap/lateral/centerline and total physical cost are modestly worse. No safety reset occurred, so the same run continues to checkpoint 1200.
- Checkpoint `model_1200.pt` was saved. At update 1232, Q anchor/target remain aligned (`2.452/2.448`), local-advantage std `0.0087`, neighborhood variance `1.38e-5`, and no non-finite or safety-reset event occurred.
- In window 1181-1200, AMS versus historical Meta has reward `0.2311 vs 0.2221`, speed error `0.1839 vs 0.1908`, lateral error `0.0611 vs 0.0690`, and nearly identical centerline error, while gap error and total physical cost remain modestly higher. The earlier lateral disadvantage has reversed at this checkpoint, but fixed evaluation remains necessary.
- Checkpoint `model_1500.pt` was saved. At update 1536, Q anchor/target are `2.996/2.986`, local-advantage std `0.0106`, and all diagnostics remain finite with zero bad-orientation reset.
- Window 1481-1500 continues to improve reward (`0.2360 vs 0.2179`), speed (`0.1804 vs 0.1947`) and lateral error (`0.0644 vs 0.0663`) relative to Meta training, with centerline essentially tied. Gap error (`0.1119 vs 0.1015`) and total physical cost remain worse. The run remains numerically healthy, so preserve the fixed experiment instead of changing weight mid-run.
- Checkpoint `model_1800.pt` was saved. At update 1839, Q anchor/target are `3.455/3.465`, local-advantage std is `0.0154`, and neighborhood variance remains small; no NaN, runtime error, or bad-orientation reset occurred.
- Window 1781-1800 again has higher reward and lower speed/lateral error than Meta, but the persistent gap and physical-cost disadvantage remains. Because the Q target is still tracking rather than diverging, continue unchanged to the late checkpoints.
- Checkpoint `model_2100.pt` was saved. At update 2143, Q anchor/target are `3.844/3.866`, local-advantage std `0.0146`, every AMS value is finite, and safety resets remain zero.
- Window 2081-2100 improves over Meta in reward (`0.2384 vs 0.2145`), speed, lateral, centerline, and now total physical cost (`0.0765 vs 0.0770`); gap error alone remains worse (`0.1144 vs 0.1023`). This is the strongest late training window so far, but final fixed evaluation is still required.
- Checkpoint `model_2400.pt` was saved. At update 2445, Q anchor/target are `4.499/4.484`, local-advantage std `0.0094`, all values are finite, and reset-on-bad-orientation remains zero.
- Window 2381-2400 still improves reward and speed over Meta, but physical cost, gap, lateral and centerline are worse in this window. The change from checkpoint 2100 shows training-window physical metrics fluctuate enough that the sequential fixed evaluation is essential.
- Checkpoint `model_2700.pt` was saved. At update 2741, Q anchor/target are `5.019/5.008`, local-advantage std `0.0113`, and the complete AMS log remains finite with zero bad-orientation reset.
- Window 2681-2700 is again better than Meta in reward, speed, lateral, and centerline, while gap and total physical cost are worse. Training is stable through the last scheduled intermediate checkpoint; only the final 300 updates remain.
- Formal training completed all 3000 updates in `7633.41 s` (about 2 h 7 min) and saved `model_final.pt`. No AMS value became non-finite and no bad-orientation reset occurred during the final row.
- Final update Q anchor/target are `5.206/5.215`, Q loss `0.435`, local-advantage std `0.0115`, and neighborhood variance `1.21e-5`. The last-100 training-window reward is `0.2323`; these remain diagnostic only.
- Sequential 11-checkpoint evaluation has started in a fresh 32-environment process using the complete latpair-v3 override and `FRESH_ENV_PER_CHECKPOINT=0`, matching the historical checkpoint-order protocol.
- First evaluated AMS checkpoints: `model_0=2906.345`, `model_300=18556.893`, `model_600=18580.415`. All have zero collision and bad-orientation reset.
- At iteration 300, AMS is `+604.704` return above the Meta v3 iteration-300 result (`17952.189`), indicating substantially faster entry into the high-return region. At iteration 600, AMS is `-102.490` below the Meta v3 peak (`18682.905`), so peak/final superiority is not yet established.
- Additional AMS returns: iteration 900 `18286.523`, 1200 `18289.869`, 1500 `18546.133`, 1800 `18180.866`; all retain zero collision/reset and low centerline/lateral errors.
- Compared checkpoint-for-checkpoint with Meta v3, AMS is below Meta from 600 through 1800 (return gaps about `68-169` after iteration 900), except for its large gain at iteration 300. Current evidence supports faster early learning but not a higher peak; late/final checkpoints remain pending.
- Late AMS returns are iteration 2100 `18390.217`, 2400 `18375.215`, and 2700 `17994.664`, with zero collision/reset throughout. Their gaps to Meta narrow to roughly `37-47` return points, but remain negative. Only the final checkpoint is pending.
- The complete first evaluation finished with final AMS return `18220.040` versus historical Meta v3 `18262.624` (`-0.233%`), and AMS peak `18580.415` versus `18682.905` (`-0.549%`). Thus AMS does not beat Meta on peak/final return in this first matched-protocol comparison.
- Final AMS improves centerline error by `7.60%`, lateral error by `14.25%`, heading error by `5.63%`, pair-heading error by `12.90%`, and critical lateral Shield rate by `83.06%`. It is slightly worse in speed error (`+0.29%`), gap error (`+0.45%`), and minimum gap (`-0.066%`); collision and reset remain tied at zero.
- AMS has a large iteration-300 return gain and slightly lower late return variance, but lower late mean return. This is a mixed result: faster early learning and better lateral/orientation quality, not an overall return win.
- Because the historical Meta CSV was generated under an older Linux/Isaac runtime and the model-0 protocol reproduction drifted by `0.054%`, the existing Meta checkpoint sequence must now be re-evaluated under the current runtime before the final verdict. This is more defensible than tuning Shield parameters after seeing the test result.
- A current-runtime Meta `latpair-v3` re-evaluation is now running over the same ordered 11-checkpoint sequence, with 32 environments, seed 42, 1000 evaluation steps, and `FRESH_ENV_PER_CHECKPOINT=0`. Its output will be used for the final apples-to-apples AMS comparison.
- The current-runtime Meta sequence reproduced `model_0=2858.652`, exactly matching the prior protocol-reconstruction check. This confirms that the final comparison is executing the intended full v3 settings; the remaining ten checkpoints are still in progress.
- Current-runtime Meta checkpoint results now include iteration 300 `17952.678` and iteration 600 `18680.429`. AMS remains much better at iteration 300 (`+604.214`) but is `-100.014` below Meta at iteration 600. The early-learning advantage and lower peak trend therefore persist after removing runtime drift.
- While the remaining Meta checkpoints run, a reusable comparison tool will be added to generate checkpoint-level and final-metric CSVs, a concise Markdown report, and return/metric figures from the two `eval_summary.csv` files. This keeps the final verdict reproducible rather than relying on hand calculations.
- Current-runtime Meta returns through iteration 1800 are: 900 `18452.431`, 1200 `18456.832`, 1500 `18613.839`, and 1800 `18344.174`. AMS remains lower by about `68-167` points at these checkpoints, so the mid-training tradeoff also persists under the unified runtime.
- Added `scripts/tools/compare_ams_vs_meta.py`. Once the Meta summary closes, it will emit matched checkpoint/final-metric CSVs, a Markdown report, and return/metric figures, including fixed-seed and direction-of-improvement safeguards.
- The comparison tool passes Python compilation and an end-to-end smoke against the historical summary, producing all five expected artifacts and reproducing the prior `-42.584` historical-runtime final-return delta.
- Current-runtime Meta iteration 2100 is `18426.635`; AMS is `-36.418` at the same checkpoint. The gap narrows late in training but has not reversed.
- Current-runtime Meta iterations 2400 and 2700 are `18419.789` and `18044.201`; AMS differences are `-44.574` and `-49.537`. The full intermediate sequence is now consistent: AMS wins clearly only at iteration 300, while its later return is slightly lower. The final checkpoint is still running.

## 2026-07-16 Final Current-Runtime Comparison

- The current-runtime Meta re-evaluation completed all 11 ordered checkpoints successfully with exit status 0. Its final return is `18262.254`; AMS final is `18220.040`, a difference of `-42.214` (`-0.231%`).
- Meta peak is `18680.429` at iteration 600; AMS peak is `18580.415` at iteration 600, `-100.014` (`-0.535%`) lower. From iteration 1200 onward, Meta/AMS mean returns are `18366.818/18285.286`, so AMS is lower by `81.531`; population standard deviations are `165.199/163.350`, giving AMS slightly lower late fluctuation.
- At iteration 300, AMS is `18556.893` versus Meta `17952.678`, a `+604.214` (`+3.366%`) early-learning gain. The robust conclusion is faster early learning, not a higher final or peak return.
- On the final checkpoint, AMS improves centerline error by `8.871%`, lateral error by `14.183%`, heading error by `7.031%`, pair-heading error by `12.692%`, ordinary lateral Shield rate by `3.352%`, and critical lateral Shield rate by `83.120%`. It is worse on speed error by `0.320%`, gap error by `0.468%`, and mean minimum-pair-gap by `0.070%`; collision and bad-orientation reset remain tied at zero.
- Final verdict for this fixed seed and protocol: **AMS is not globally better than Meta on the primary return criterion. It is a tradeoff: much faster at iteration 300 and materially better in lateral/orientation control and critical Shield interventions, with a small final-return penalty.** A statistical superiority claim still requires multiple independent training seeds.
- Reproducible artifacts are under `logs/rsl_rl/platoon_happo/hardb_ams_20260716_060808_package/comparison_vs_meta/`: `checkpoint_comparison.csv`, `final_metric_comparison.csv`, `comparison_report.md`, `return_comparison.png`, and `final_metric_improvements.png`. The raw current-runtime Meta step CSVs and `eval_summary.csv` are preserved in its `meta_v3_current_runtime/` subdirectory.
- Committed the reusable comparison generator as `e3902e8` (`Add reproducible AMS versus Meta report`) and pushed it to `origin/agent/ams-platoon-smoothing`. Draft PR `hs997/densact#2` now contains both AMS implementation commit `a9ba8be` and report commit `e3902e8`.
- Posted the fixed-seed 3000-iteration result and its mixed verdict to PR #2: `https://github.com/hs997/densact/pull/2#issuecomment-4986787092`.

## 2026-07-16 Final Packaging

- Refreshed the formal package with the final `ams_debug_notes.md`, current-runtime Meta raw evaluation data, reproducible comparison CSV/report/figures, AMS training metrics, `model_best.pt`, `model_final.pt`, pinned environment/agent configs, pipeline log, manifest, and the original poster image under `reference/`.
- Generated package-relative `SHA256SUMS.txt` and verified every listed file successfully.
- Rebuilt both archives at `logs/rsl_rl/platoon_happo/hardb_ams_20260716_060808_package.tar.gz` and `logs/rsl_rl/platoon_happo/hardb_ams_20260716_060808_package.zip` (about `34 MB` each).
- Copied the ZIP to `/home/cnc/Desktop/hardb_ams_20260716_060808_package.zip`; `unzip -tq` reports no compressed-data errors.
- Final sanity check: the relevant tracked AMS implementation/launcher/report files are clean, `HEAD` and `origin/agent/ams-platoon-smoothing` both point to `e3902e8`, the comparison script compiles and executes successfully, all package checksums pass, and no formal training or evaluation process remains running.

## 2026-07-20 New Codex Session Handoff

- The AMS stream should be resumed from a new Codex session by reading this file first and using `/home/cnc/SSD_1T/xzw/IsaacLab-main` as the working root.
- The new session must keep the work streams separate: AMS conclusions go only to `/home/cnc/SSD_1T/xzw/ams_debug_notes.md`; Meta/base-hard-b conclusions go only to the repository `debug_notes.md`.
- The original poster may be attached from `/home/cnc/Desktop/微信图片_20260622110047_596_28.jpg`.
- The handoff prompt should require a git/active-process/artifact audit before further edits or training, preserve existing results, create new tags/backups for new experiments, avoid concurrent Isaac train/eval jobs, and update this Markdown file before every response.

## 2026-07-20 Resume Audit In Progress

- Re-read this 200-line AMS log completely and visually re-checked the attached AMS poster. The poster confirms the already implemented core idea: replace pointwise TD targets with averages over QR-orthogonal action-neighborhood samples; its stated robust setting is approximately `epsilon=0.2`, `K=8`.
- Repository audit starts from `agent/ams-platoon-smoothing` at `e3902e8`, exactly matching `origin/agent/ams-platoon-smoothing`. The only tracked worktree modification shown by Git is the pre-existing Meta-only `debug_notes.md`; it will not be touched by the AMS stream.
- No active Isaac training, evaluation, AMS, HAPPO, or Meta process was found in the first process scan. Therefore there is no concurrent job to preserve or wait for before completing the artifact audit.
- The external AMS log states that implementation, smoke tests, the formal 3000-iteration run, matched current-runtime Meta re-evaluation, reproducible comparison, GitHub snapshot, and final package were already completed. Those experiments must not be repeated unless the remaining audit finds a missing or corrupt artifact.
- The expected package directory, repository TAR/ZIP, and desktop ZIP all exist; both ZIP copies are about 34 MB and the desktop ZIP passes `unzip -tq`.
- Git inspection confirms the AMS implementation commit `a9ba8be` contains the intended 10-file/602-line integration, while `e3902e8` adds only the 243-line reproducible comparison tool. No tracked AMS source or launcher has a new worktree modification.
- The first package checksum command was invoked from the repository root even though `SHA256SUMS.txt` uses package-relative `./...` paths; its “file not found” output is a command working-directory error, not evidence of artifact corruption. Verification must be rerun from inside the package directory.
- Re-running `sha256sum -c SHA256SUMS.txt` from inside the package succeeded for all 45 listed artifacts, including both model files, all AMS/Meta step CSVs, summaries, figures, configs, pipeline log, poster, and packaged notes.
- Static verification remains clean on the resumed session: AMS/comparison Python files compile, both shell launchers pass `bash -n`, and `git diff --check` reports no whitespace error.
- The repository ZIP and desktop ZIP are byte-identical with SHA-256 `44c97c61f403c00c6496a8c534a8c07abbfad2fed30949657472a77a0ff194e7`; the TAR SHA-256 is `8a897fc26b3fd6b5c371e272b57b50ac04a13fee182528bc09de06b20c9b0668`.
- Raw packaged AMS and current-runtime Meta summaries still contain the recorded 11-checkpoint fixed sequence. The pipeline log confirms final checkpoint save and final AMS evaluation return `18220.040` with zero collision and zero bad-orientation reset.

## 2026-07-20 Resume Audit Final

- Independently re-ran `scripts/tools/compare_ams_vs_meta.py` into `/tmp/ams_resume_audit_20260720`. The generated checkpoint CSV, final-metric CSV, and Markdown report are byte-identical to the packaged versions; the regenerated result is again final AMS-minus-Meta `-42.214` (`-0.231%`).
- Validated both raw summary files and all 22 referenced step CSVs. Each method has the exact ordered sequence `model_0, model_300, ..., model_2700, model_final`; every checkpoint has 1000 evaluation rows; numeric summary fields are finite; hard-b attack fields are consistently enabled with FDI acceleration `1.3` and DoS `0.18`; collision and bad-orientation reset are zero at every checkpoint.
- The manifest confirms tag `hardb_ams_20260716_060808`, `K=8`, neighborhood radius `0.25`, AMS advantage weight `0.10`, warm-up/ramp `50/100`, and the ordered seed-42, 32-environment, 1000-step latpair-v3 comparison protocol.
- A final process scan again found no Isaac training or evaluation process. Git remains on `agent/ams-platoon-smoothing` at pushed commit `e3902e8`; the only tracked modification remains the pre-existing Meta-only `debug_notes.md`, which this session did not edit.
- Resume verdict: the requested AMS implementation, tests, 3000-iteration training, fixed-protocol evaluation, current-runtime Meta comparison, reproducible report, backup/push, and verified packaging are already complete and internally consistent. Starting another identical run would duplicate completed work, so no Isaac experiment was launched in this session.
- Scientifically, the completed single-seed result remains a tradeoff rather than a return win: AMS is much faster at iteration 300 and improves lateral/orientation quality and critical Shield intervention rate, but its final return is `42.214` lower and peak return `100.014` lower than Meta. The next non-duplicative research step would be a predeclared multi-seed study, which is outside the completed single-candidate goal and should receive fresh experiment tags/backups before launch.

## 2026-07-21 Vehicle-Model Dimensionality Check

- The experimental vehicle is not a one-dimensional physical model. Isaac/PhysX loads the unfixed `senior_4wd_bs_robot.urdf` as a 3D articulated rigid-body model with chassis mass/inertia, collision meshes, and four continuous wheel joints per vehicle.
- Each of the five agents outputs four wheel-velocity actions (`act_dim=4`, joint action dimension 20). Differential left/right wheel commands produce yaw and lateral correction; observations explicitly include longitudinal and lateral relative position/velocity, body `vx/vy`, yaw rate, centerline offset, heading, and pair-heading error.
- The experiment is intentionally close to a planar/straight-road platoon task: the reference command varies only longitudinal speed (`lin_vel_x=0.6..0.8`) while commanded lateral velocity and yaw rate are zero. Therefore the high-level command is one-dimensional, but the simulated vehicle dynamics and controller are not; they retain lateral, yaw, wheel, contact, and full 3D rigid-body effects.

## 2026-07-21 20-Second Lane-Change Experiment Start

- New requested experiment: replace the fixed straight reference with one smooth 20-second lane change on the current map, then train the existing HAPPO+Meta+AMS controller using the full four-wheel dynamics and local observations.
- Map audit confirms two usable straight lane centers at `y=0.0` and `y=-1.20 m`, separated by the dashed divider at `y=-0.60 m`. The environment episode is exactly 20 seconds (`dt=0.005`, decimation 4, control step `0.02 s`), making one complete lane change per episode a reproducible protocol.
- The planned reference is a zero-endpoint-velocity/acceleration quintic transition from `y=0.0` to `y=-1.20 m` over the full episode. Its tangent will define consistent world-frame lateral velocity, desired heading, and yaw-rate commands while the existing longitudinal speed remains sampled from the pinned hard-b range.
- Existing fixed-centerline terms occur in dense rewards, observations, Meta Teacher physical costs, task-local reward shaping, Shield correction, and metrics. All must use trajectory-relative errors; changing only the visual/reference command would leave several controllers fighting the lane change.
- Initial audit found no active Isaac train/eval process. Git remains at pushed AMS commit `e3902e8`; the pre-existing tracked modification is only Meta-only `debug_notes.md`, which will remain untouched.
- Created and verified the pre-edit backup at `backups/20260721_lanechange_preedit/`: a complete 10 MB Git bundle, binary tracked-worktree patch, selected-source archive, and SHA-256 manifest. Bundle history and every checksum pass.
- Implemented a default-disabled `LaneChangeVelocityCommand` so prior straight experiments remain reproducible. When enabled, it samples the pinned longitudinal speed and produces a quintic `y=0 -> -1.20 m` reference together with trajectory-tangent body velocity and yaw-rate commands.
- Converted fixed-centerline assumptions to trajectory-relative quantities in the policy observation (without changing its 11-dimensional layout), dense centerline/lateral-velocity/heading rewards, Meta Teacher physical cost, HAPPO local shaping, safety Shield centerline correction, and training/evaluation metrics.
- Added lane-change diagnostics for commanded lateral speed, yaw rate, reference progress, and reference `y`; extended the evaluator summary; and added the dedicated launcher `scripts/tools/run_platoon_ams_lanechange20s_3000.sh` with a fresh `hardb_ams_lanechange20s_*` tag.
- Python compilation, launcher syntax, and `git diff --check` currently pass. No existing log, checkpoint, or package has been overwritten.
- Smoke tag `hardb_ams_lanechange20s_smoke_20260721_1` successfully completed one real four-environment HAPPO+Meta+AMS update: Hydra accepted all new overrides, the policy retained `obs_dim=11`, each agent retained four wheel actions, joint action dimension remained 20, AMS produced finite Q loss `0.5048`, and the final checkpoint was saved.
- The initial untrained checkpoint completed a full 1000-step lane-change evaluation and wrote all step metrics. Its poor return/control is expected and is not a performance result; it confirms the end-to-end trajectory command, dynamic reward/Teacher/Shield paths, checkpoint load, and evaluator execute without a Python/CUDA/non-finite error.
- Smoke 1 used `FRESH_ENV_PER_CHECKPOINT=1`; after `model_0` completed, rebuilding a second Isaac environment inside the same evaluator process remained CPU-active without producing the second result, so it was interrupted instead of risking an allocator failure. This is an evaluator lifecycle issue, not a lane-change/AMS exception; the formal protocol will use the already established ordered single-environment evaluation (`FRESH_ENV_PER_CHECKPOINT=0`).
- The first command version expressed the world tangent in the robot's instantaneous body frame, which could request infeasible lateral body velocity when an untrained vehicle had large heading error. It was corrected to the non-holonomic command `[tangent speed, 0, trajectory yaw rate]`; spatial tracking remains closed by trajectory-relative `y` and heading observations/rewards.
- Smoke 2 (`hardb_ams_lanechange20s_smoke_20260721_2`) passed with exit status 0. The logged body lateral command is exactly zero, yaw-rate command is active, trajectory progress/reference fields evolve, reset-on-bad-orientation is zero in the training row, and every AMS/trajectory diagnostic is finite (`Q loss=0.5047`, neighborhood variance `4e-6`, `K=8`, radius `0.25`).
- Static checks were repeated successfully, and the exact eight-file lane-change implementation was committed locally as `0bf1566` (`Add 20-second lane-change AMS experiment`). The unrelated Meta-only `debug_notes.md` modification and unrelated untracked files were excluded.

## 2026-07-21 Formal 20-Second Lane-Change AMS Run

- Launched unique tag `hardb_ams_lanechange20s_20260721_172500`; training directory is `logs/rsl_rl/platoon_happo/2026-07-21_17-20-35_hardb_ams_lanechange20s_20260721_172500_happo_meta_ams` and pipeline log is `train_hardb_ams_lanechange20s_20260721_172500.log`.
- Startup audit confirms 64 environments, 5 agents, unchanged `obs_dim=11`, `act_dim=4`, 20-dimensional joint action, Meta Teacher, hard-b attack, dynamic-reference Shield, and AMS with `K=8`, radius `0.25`, target weight `0.10`.
- The first 12 updates are finite with no Python/CUDA/non-finite error. AMS Q loss falls from `0.6798` to the `0.05-0.09` range during warm-up, neighborhood variance is small and non-zero, and the AMS actor weight correctly remains zero before update 50.
- The initial formal attempt was stopped shortly after update 51 because it showed a concrete curriculum failure before AMS actor blending: bad-orientation reset rose monotonically from zero to `0.94`, last-10 reward was `-0.0983`, centerline error stayed near `0.503 m`, speed error near `0.367 m/s`, and critic loss kept increasing. Continuing 3000 updates under hard-b attack from a random lane-change policy would waste compute rather than test AMS.
- Root-cause evidence points to task difficulty, not AMS numerics: the collapse occurred during the first 50-update AMS warm-up while AMS actor weight was zero, whereas Q loss/target diagnostics stayed finite. The revised training protocol will first learn the 20-second maneuver with attack level off while retaining the full dynamics, Meta Teacher, Shield, and AMS; hard-b robustness can be evaluated only after a viable maneuver policy exists.
- The stopped attempt and its partial metrics/checkpoints are preserved under tag `hardb_ams_lanechange20s_20260721_172500`; they will not be overwritten or presented as a completed result.
- After stopping that training subprocess, its pipeline had already advanced to a hard-b checkpoint evaluation; this owned evaluation was explicitly terminated and no AMS Isaac process remains. A separate pre-existing 2000-step ROS replay evaluation (`hardb_latpair_v3_model_final_40s_20260721`) was then discovered running under another parent process. The planned clean lane-change smoke was aborted before startup to honor the no-concurrent-Isaac rule, and will wait for that external evaluation to finish.
- Metric-semantics correction: training CSV field `termination_reset_on_bad_ori` reads Isaac's per-environment `_last_episode_dones` classification and remains true after an environment's latest completed episode ended for that reason; it is not the instantaneous reset rate. In the clean smoke it reached `1.0` while the actual `done_rate` was usually zero, proving the earlier `0.94` interpretation overstated active collapse. The hard-b attempt was still legitimately poor in reward/speed/trajectory tracking, but its stop decision must not be described as a 94% per-step reset rate.
- Once the unrelated ROS replay process exited, the guarded launcher verified that no train/eval process remained and started `ams_lanechange20s_clean_smoke100_20260721_1`. This clean run is continuing to 100 updates; early dynamic-reference centerline error fell as low as `0.115 m` after episode resets, all AMS diagnostics remain finite, and actual `done_rate` is normally zero with only sparse reset steps.
- The 100-update clean smoke completed successfully in `258.23 s` with pipeline exit status 0. Over the final 20 updates, mean reward is positive (`0.0472`), actual done rate is `0.00156`, speed error improved to `0.2929`, gap error is `0.0678`, centerline error is phase-dependent at `0.3083`, and heading error is `0.1831`. At update 100 AMS remains finite (`Q loss=0.0682`, neighborhood variance `1.4e-5`, local-advantage std `0.0312`) and its ramped weight is exactly `0.05`.
- Apparent non-finite values in a whole-CSV blanket check are only two intentionally blank/inapplicable columns while attack is off (`attack_ref_template_id`, `physical_centerline_cost`); all trajectory, optimization, AMS, reward, safety, and motion diagnostics are finite.
- Clean training therefore passes the go/no-go gate for a new 3000-update run. It shows learning progress and sparse actual terminations, unlike the poor hard-b start, while retaining the requested dynamics, observation layout, Meta Teacher, Shield, and AMS controller.

## 2026-07-21 Formal Clean Lane-Change Training

- Committed the clean-first protocol adjustment as `c262642` (`Train lane change before attack evaluation`), after the main implementation commit `0bf1566`.
- Guarded launch confirmed no Isaac train/eval process was active, then started formal tag `ams_lanechange20s_clean_20260721_173000`. Training directory is `logs/rsl_rl/platoon_happo/2026-07-21_17-32-08_ams_lanechange20s_clean_20260721_173000_happo_meta_ams`; pipeline log is `train_ams_lanechange20s_clean_20260721_173000.log`.
- Startup confirms the intended 64-env, 5-agent, 11-observation, 4-wheel-action, 20-joint-action HAPPO+Meta+Shield+AMS configuration with attack disabled for maneuver acquisition. The first updates are finite with zero actual done rate and AMS warm-up weight zero.
- The launcher will train 3000 updates, save checkpoints every 300 updates, then evaluate all 11 checkpoints serially with `FRESH_ENV_PER_CHECKPOINT=0`. No other Isaac job may be started while this pipeline is active.

## 2026-07-31 Session Resume And Completion Audit

- Re-read the complete AMS handoff log before taking action and audited the repository, active processes, formal pipeline log, package, and evaluation summary. No IsaacLab training/evaluation or lane-change pipeline process is currently active.
- The clean lane-change run `ams_lanechange20s_clean_20260721_173000` did complete all 3000 updates, saved `model_final.pt`, evaluated the ordered 11-checkpoint sequence (`model_0`, every 300 updates through `model_2700`, then `model_final`) with 1000 rows per checkpoint and `FRESH_ENV_PER_CHECKPOINT=0`, generated figures, built the package TAR, and exited with status 0.
- Final-update diagnostics remained finite: policy loss `-0.0041`, value loss `1.9039`, AMS Q loss `0.1138`, and full AMS advantage weight `0.10`. Training and evaluation used attack disabled, as intended for maneuver acquisition; therefore this run establishes clean lane-change learning only and is not evidence of hard-b robustness.
- Fixed clean-evaluation return rises from `-437.507` at `model_0` to a peak of `11014.061` at iteration 600. The final return is `10340.263`; final speed error is `0.1231`, trajectory-relative centerline error `0.2240`, lateral pair error `0.4306`, heading error `0.2028`, minimum pair gap `1.5792`, collision rate `0`, and bad-orientation reset `0`.
- The final checkpoint is not the best clean checkpoint by return, and Shield intervention remains high at the final checkpoint (`lateral_rate=0.8871`, `critical_rate=0.4846`). The experiment demonstrates acquisition of the 20-second maneuver but also indicates incomplete lateral/platoon quality and late-training regression; it should not yet be described as a solved robust lane change.
- Artifacts are preserved at `IsaacLab-main/logs/rsl_rl/platoon_happo/ams_lanechange20s_clean_20260721_173000_package/` and its `27 MB` TAR archive. The pipeline log is `IsaacLab-main/train_ams_lanechange20s_clean_20260721_173000.log`.
- Git is on `agent/ams-platoon-smoothing` at local commit `c262642`, two commits ahead of `origin/agent/ams-platoon-smoothing`: `0bf1566` adds the lane-change implementation and `c262642` switches to clean-first training. The only tracked worktree modification is the pre-existing Meta-only `debug_notes.md`; unrelated untracked files remain and must continue to be excluded by an explicit allowlist.
- The next non-duplicative work is to select/evaluate the clean checkpoint under a predeclared hard-b protocol (preferably comparing the peak-return iteration-600 checkpoint with the final checkpoint), then decide from that evidence whether robust fine-tuning is warranted. Before any new Isaac run, recheck for concurrent processes; before publication, refresh/verify packaging and push the two local lane-change commits explicitly.

## 2026-07-31 Lane-Change Data Visualization Audit

- Audited the formal package for trajectory, reward, and maneuver visualizations. The package originally contained only checkpoint return, aggregate metric, and final-bar plots; their inherited `fixed-medium` titles are misleading for this clean lane-change run. It contained no true vehicle trajectory plot, reward-component plot, rendered frame, or video.
- The evaluator CSVs do not preserve per-environment longitudinal/planar poses, so a true single-rollout world-frame `x/y` path cannot be reconstructed after the fact. They do preserve the quintic reference and each vehicle's signed trajectory-relative lateral error averaged over 32 environments; therefore mean lateral position can be reconstructed exactly as `reference_y + signed_error` without inventing data.
- Added the reproducible generator `scripts/tools/plot_lanechange_run_report.py` and created three diagnostics under the package's `figures/lanechange_diagnostics/`: smoothed training reward/components, iteration-600 mean lane-change tracking, and final-checkpoint mean lane-change tracking.
- The reward curve rises rapidly from negative values and plateaus near `0.22-0.24` by roughly update 250; it does not show continued improvement after iteration 600. Logged leader-motion and formation terms dominate the raw reward scale, while trajectory-lateral and local-shaping terms remain penalties.
- The tracking plots materially qualify the aggregate return result. Both iteration 600 and the final checkpoint initiate the lane change, but the 32-environment mean vehicle paths lag the quintic reference, spread substantially across the five vehicles, and do not all settle at the target lane center `y=-1.20 m` by 20 seconds. Some vehicles overshoot while the last vehicle undershoots; pair lateral error peaks near `0.9 m` late in the maneuver.
- The current defensible status is therefore: the policy learned a lane-change tendency with zero evaluated collision, but coordinated trajectory tracking is not yet satisfactory. Hard-b robustness evaluation should not be interpreted as the only remaining issue; clean tracking/reward balance should be improved first.
- No simulator-rendered lane-change image/video exists in the current package. Producing one requires a fresh camera-enabled evaluation and must follow the no-concurrent-Isaac guard; raw per-environment `x/y/yaw` logging should be added at the same time if a publication-quality plan-view trajectory is required.

## 2026-07-31 Real Video And World-Frame Trajectory

- In response to the request for a real lane-change video and top-down 2D trajectory, verified that no Isaac process was active and ran one exclusive, camera-enabled, deterministic replay of the best-return clean checkpoint `model_600.pt` under the clean lane-change deployment-Shield protocol.
- The first sandboxed launch could not see the NVIDIA driver and stopped before environment creation; it produced no scientific data. The approved GPU launch then completed normally with exit status 0. No Isaac process remains.
- The replay used one environment, seed 42, five vehicles, four wheel actions per vehicle, task-local HAPPO+Meta+AMS loaded strictly, attack disabled, `dt=0.02 s`, and the 20-second quintic `y=0.0 -> -1.20 m` command. A fixed overhead camera rendered the same rollout whose env-0 poses were exported.
- Generated an H.264 overhead video at `1280x720`, 50 fps, 999 frames, and `19.98 s`: `logs/rsl_rl/platoon_happo/lanechange_model600_real_video_20260731/lanechange_model600_overhead_19p98s.mp4`. Its SHA-256 is `8585e98a1804b1c8e38b14088c0cf369ee99c7c827f1ecdf70d51c929f25344a`.
- Exported `trajectory_env0.csv` with exactly 5000 rows (`1000 steps x 5 vehicles`) and real world-frame `x/y/yaw`, velocities, commands, and raw/executed actions. The integrity checker passed field presence, robot IDs per step, monotonic time, and required-value finiteness; only the intentionally inapplicable steering command is NaN.
- Added `scripts/tools/plot_ros_replay_topdown.py` and generated `topdown_true_xy_model600.png`. The automatic reset pose at step 1000 is detected and excluded, so the figure uses steps 1-999 through `19.98 s` and contains no reset line artifact.
- At `19.98 s`, vehicle endpoint `y` values are `[-1.780, -1.609, -1.282, -1.241, -1.099] m` for vehicles 1-5, respectively, versus target `-1.20 m`; endpoint lateral span is `0.681 m`. The real single-rollout trajectory confirms that vehicles 1-2 overshoot while vehicle 5 undershoots and the formation is laterally dispersed.
- Added a six-phase overhead contact sheet, a self-contained README with protocol/checksums/endpoints, and a reproduction script under `logs/rsl_rl/platoon_happo/lanechange_model600_real_video_20260731/`. The result is a faithful diagnostic presentation of the current best-return policy, not a claim of satisfactory lane-change control.
- Opened the completed real-video artifact directory in the desktop file manager for direct inspection: `IsaacLab-main/logs/rsl_rl/platoon_happo/lanechange_model600_real_video_20260731/`.

## 2026-07-31 Lane Geometry And Reference-Logic Re-Audit

- Re-audited the map geometry, URDF footprint, lane-change command, observations, dense rewards, local shaping, Meta Teacher, Shield, terminations, metrics, and the real model-600 trajectory after the video made the outer-lane behavior look like an off-road departure.
- Visible lane geometry is: near outer edge-line center `y=0.47 m`, dashed divider `y=-0.60 m`, and far outer edge-line center `y=-1.72 m`. Using the inside edges of the 2.6 cm outer lines gives a usable visual-road envelope of approximately `[-1.707, 0.457] m`; lane widths are about `1.07 m` and `1.12 m`.
- The exact visual lane midpoints are approximately `-0.065 m` and `-1.160 m`, not the configured `0.0 m` and `-1.20 m`. The configured centers are offset by only `6.5 cm` and `4 cm`, respectively, so this alignment error should be corrected but does not by itself explain the large departure.
- URDF mesh/axle audit gives an overall wheel-envelope width of about `0.385 m` and length of about `0.465 m`. A straight vehicle centered at the configured target `y=-1.20 m` retains roughly `0.315 m` clearance to the usable far outer edge; the quintic reference itself also stays safely inside the road. Therefore the lane is not intrinsically too narrow for this robot.
- The real model-600 rollout does leave the visible road. Using a yaw-aware rectangular footprint (`0.465 x 0.385 m`), vehicle 1 is outside for 139 of 999 physical steps starting at step 861 and reaches a maximum `0.345 m` footprint exceedance; vehicle 2 is outside for 104 steps starting at step 896 with maximum `0.160 m` exceedance. Vehicles 3-5 remain inside. Center-only checking would detect only 33 vehicle-1 steps and miss most of the footprint violation.
- The simulation has no physical road-edge semantics: the global 2000 m terrain plane supports vehicles everywhere, and the invisible `HighwayCollision` slab itself spans approximately `[-2.025, 0.925] m`, wider than the painted road. The outer lines/curbs are effectively visual dressing for control purposes.
- No observation, reward, Meta Teacher cost, Shield projection, termination, or published metric uses road boundaries or a yaw-aware vehicle footprint. `collision_rate` measures only inter-vehicle proximity, so `collision=0` does not mean the platoon stayed on the road. This is the main reason an apparent departure can still score as collision-free.
- The 20-second maneuver duration equals the 20-second episode length and command-resampling time. The quintic reference reaches its target and zero tangent yaw only at the timeout; there is no pre-maneuver hold or target-lane settling phase. In the real rollout the vehicles have begun recovering positive yaw rate late, but still finish with large negative yaw, so the episode ends before recovery can be rewarded or failure exposed.
- There is also a reference/formation inconsistency. All five vehicles receive the same time-based `reference_y(t)` and heading, which asks them to have equal world `y` simultaneously. Formation reward, local shaping, Teacher lateral cost, and Shield pair correction instead measure predecessor-relative lateral offset in the predecessor body frame and ask every follower to remain directly behind. At the quintic peak lateral speed, reference heading is roughly `14-21 deg`; a nominal `1.5 m` longitudinal separation then produces about `0.36-0.53 m` body-frame lateral offset even when both vehicles exactly satisfy the common `y_ref(t)`, exceeding the `0.25 m` hard pair threshold. The two objectives therefore fight each other during the curve.
- Recommended redesign is a spatial same-path platoon lane change: define one road-frame path `y_ref(x)` between corrected lane centers, evaluate separate reference `y/heading/curvature` at each vehicle's own longitudinal position, and allow followers to reach each path phase later. This preserves the existing body-frame predecessor geometry and represents vehicles following the same physical path. It requires an episode long enough for the last vehicle to complete the path plus a target-lane dwell.
- If a synchronized, parallel 20-second whole-platoon shift is preferred instead, the common `y_ref(t)` may remain, but formation/gap/lateral objectives must be changed consistently to road-frame/world-axis geometry during the maneuver. Mixing common-time lateral reference with predecessor-body-frame collinearity must not continue.
- In either redesign, add one authoritative `LaneGeometry` definition shared by rendering and control, yaw-aware four-corner road-envelope metrics, a soft boundary penalty/Shield margin, an off-road termination with persistence/tolerance, pre-change and post-change dwell phases, raw per-vehicle world-pose evaluation, and explicit boundary-violation rate/max-exceedance reporting.
- No core training/control code was changed during this audit because choosing spatial leader-path following versus synchronized parallel shifting is a material experiment-definition decision. Existing lane-change results remain valid as diagnostics of the old protocol but must not be presented as successful in-lane platoon lane changing.
## 2026-07-31 Spatial Same-Path Lane Change and Full Evidence Pipeline

## 2026-08-01 Ackermann/Whole-Platoon Redesign And Six-Algorithm Launch

- Re-audited the prior spatial-path evidence before accepting it. In the nominal off rollout, final per-vehicle path progress was approximately `[0.992, 0.725, 0.522, 0.267, 0.016]`; therefore only the leader completed the path and the old result cannot satisfy the newly required whole-platoon maneuver.
- Replaced the skid-steer `senior_4wd_bs_robot.urdf` vehicle with the real small Ackermann asset `mini_akm_robot.urdf`. Its controlled joints are rear drive `lb_joint/rb_joint`, front steering knuckles `left_joint/right_joint`, and passive front wheel-spin joints `lf_point/rf_joint`. Rear wheels have no steering degree of freedom.
- Added a two-dimensional per-vehicle Ackermann action `[normalized_speed, normalized_center_steering]`, giving a 10-dimensional five-vehicle joint action instead of the old 20 wheel-speed actions. The action term computes left/right Ackermann steering angles and rear differential wheel speeds; the old four-wheel sign adapter is disabled.
- Real Isaac construction smoke passed with five active 2-D action terms, `act_dim=2`, and `joint_action_dim=10`; one HAPPO+Meta+AMS update completed and checkpointed without shape/load/runtime failure.
- Deterministic physics smoke exposed and corrected the mini URDF rear-axis sign. After correction, a positive straight command moved the leader about `+1.007 m` in 3 seconds with positive body-forward velocity. A positive turn command produced unequal front targets `left=-0.1342 rad`, `right=+0.1223 rad` (URDF joint-axis convention) and a real forward/left curved trajectory. Rear joints received velocity targets only. The command speed scale is now `1.2 m/s`, so the deployment action clip `0.4` covers the full `0.30-0.45 m/s` command band.
- Updated the yaw-aware road footprint to the mini vehicle's conservative `0.240 x 0.200 m`, retained the corrected lane centers `-0.065 -> -1.160 m`, and extended the episode/evaluation/video horizon from 50 seconds/2500 steps to 70 seconds/3500 steps so the fifth vehicle can traverse the same spatial path and dwell after completion.
- Added a strict whole-platoon completion state: all five vehicles must reach path progress `>=0.99`, meet centerline/heading/CTH-gap/pair-lateral/collision/road-envelope tolerances simultaneously, and remain valid for 2 seconds. Added persistent per-episode `platoon_completion_ever_rate`; the hard-b acceptance gate is at least `0.50` of evaluation environments completing the full five-vehicle maneuver.
- Expanded the attack ladder to nine declared levels: `off`, `light`, `easy_0`, `easy_a`, `easy_b`, `med_a`, `med_b`, `hard_a`, `hard_b`. Hard-b is exactly FDI position `4.0`, FDI acceleration `1.30`, and DoS `0.18`.
- Defined the six-way comparison as `mappo`, `happo_no_meta`, `happo_meta_ams` (ours), `harl_mappo_shared`, `harl_haa2c`, and `harl_hatrpo`. All six receive the identical clean foundation and monotonically increasing attack curriculum. If ours misses the hard-b 50% gate, every method receives another identical hard-b round, up to five rounds, preserving training-budget fairness.
- The final evidence pipeline will produce, for every algorithm and every attack level, aggregate evaluation CSV, env-0 world-frame trajectory CSV, real renderer MP4, true top-down trajectory image, reward/attack/safety plot, plus cross-algorithm completion/return heatmaps and checksums.
- Created and verified the pre-edit backup at `backups/20260801_ackermann_preedit/`; Python compilation, shell syntax, `git diff --check`, real environment construction, action topology, and deterministic physical motion checks passed.
- Formal exclusive serial run launched as `ackermann_six_full_20260801_165233` (supervisor PID `36143`) under `logs/rsl_rl/platoon_happo/ackermann_six_full_20260801_165233/`. An initial launch was stopped before training because baseline algorithms did not inherit the spatial overrides; the launcher was corrected to apply common path/road/horizon parameters to all six methods, and the audited restart command now contains those parameters. Current first stage is clean/off MAPPO foundation training; no concurrent Isaac job is allowed.

- User confirmed the required geometry: all five vehicles must follow the same spatial path in sequence, and after the clean foundation the attack tests must run serially from low to high through `hard`. Every attack level must retain numeric data, reward/attack plots, a true top-down trajectory plot, and a real rendered lane-change video.
- Before editing, preserved a recovery snapshot under `IsaacLab-main/backups/20260731_spatial_path_preedit/`: `repo.bundle`, `tracked_worktree.diff`, and `selected_sources.tar.gz`, with SHA-256 hashes recorded in that directory. Existing unrelated worktree changes were not overwritten.
- Reworked `source/my_exts/marl_platoon/trajectory.py` to add the geometric quintic path `y=f(x)`. In spatial mode every vehicle evaluates the same path using its own environment-local `x`; therefore path progress, reference `y`, heading, lateral velocity, and yaw rate are vehicle-specific instead of sharing one time-indexed lateral reference.
- Unified the per-vehicle reference through policy observations, centerline/lateral/heading rewards, local reward shaping, Meta Teacher physical cost, Safety Shield, evaluator metrics, and ROS replay export. The legacy temporal behavior remains default-neutral (`lane_change_spatial=false`) for reproducibility.
- Added yaw-aware footprint road semantics using the measured road/vehicle geometry: road line centers `y=0.47/-1.72`, line width `0.026 m`, vehicle length/width `0.465/0.385 m`. New diagnostics include clearance/exceedance per step; new opt-in reward, Shield boundary correction, and material-exceedance termination are enabled only by the spatial protocol.
- The spatial protocol uses visible lane centers `-0.065 m` and `-1.160 m`, path start `x=-7.75 m`, path length `6.0 m`, initial five-car centers `y=-0.065 m`, and a 50-second episode/resampling window. At the minimum configured speed, the fifth vehicle can enter and finish the same path and still leave a settling interval.
- Extended deterministic playback with `--enable_attack_eval` and `--video_folder`; attack videos now retain the configured attack instead of silently disabling it. Extended ROS CSV with per-vehicle reference progress/y/heading/yaw-rate and footprint clearance/exceedance.
- Added reproducible tooling:
  - `scripts/tools/run_platoon_ams_spatial_lanechange_3000.sh` for the clean 3000-update foundation.
  - `scripts/tools/run_platoon_spatial_attack_ladder.sh` for serial `off/light/easy/medium/hard` evaluation, env-0 CSV, real renderer video, reward/attack/safety plot, true top-down plot, manifest, and checksums at every level.
  - `scripts/tools/run_platoon_spatial_complete_pipeline.sh` to train, select the maximum clean `episode_return_mean` checkpoint, then automatically run the five-level artifact ladder.
  - `scripts/tools/plot_spatial_attack_level.py` plus an upgraded `plot_ros_replay_topdown.py` that draws real road boundaries and marks any footprint departure red.
- Verification passed: all modified Python modules compile, all new shell launchers pass `bash -n`, and `git diff --check` is clean. A direct numeric test produced spatial progress `[0,0,0.5,1,1]` at positions before/start/mid/end/after the path, exact endpoint lane centers, and finite heading/yaw-rate.
- Real Isaac smoke training passed with one environment and one update. The command, 16 rewards, three terminations (including road boundary), HAPPO, Teacher, AMS, and Shield all constructed and stepped without runtime errors; final checkpoint was saved. The initially observed `0.798 m/s` command was traced to that isolated smoke omitting the formal speed override, not to spatial-reference corruption; the formal protocol resolves `0.30–0.45 m/s`.
- A separate 500-step/10-second real Isaac fixed-checkpoint evaluation exported 2500 robot rows under `/tmp/ams_spatial_smoke_eval_20260731/`. At step 500, per-vehicle path progress was `[0.40095, 0.08948, 0, 0, 0]`: the leader was on the curve, vehicle 2 had just entered, and vehicles 3–5 remained on the start segment. This directly verifies sequential same-path timing. Road violation rate and maximum exceedance were both zero; minimum footprint clearance remained positive. The ROS checker passed row/robot/time integrity. Smoke plots are `topdown_true_xy.png` and `reward_attack_safety.png` in that directory.
- The formal guarded run is active under tag `ams_spatial_complete_20260731_235300`. Parent scheduler PID `16527` remains attached under the user systemd session, with Isaac Python PID `16545`; therefore completion will automatically continue into clean checkpoint evaluation and the five attack levels. Training run: `IsaacLab-main/logs/rsl_rl/platoon_happo/2026-07-31_23-52-37_ams_spatial_complete_20260731_235300_foundation_happo_meta_ams`; live log: `IsaacLab-main/train_ams_spatial_complete_20260731_235300_foundation.log`.
- Early formal status near update 79/3000: about `718 steps/s`, ETA about `2 h 17 min`, finite AMS loss, `reset_on_bad_ori=0`, `road_boundary_exceeded=0`, and road-boundary reward penalty `0`. A second attempted guarded launch detected this active process and exited before creating a concurrent Isaac job.

## 2026-08-01 Video Folder Opened

- Confirmed the complete spatial-path attack ladder has produced real renderer videos for all five requested levels: `off`, `light`, `easy`, `medium`, and `hard`. Each video is stored as `<level>/video/rl-video-step-0.mp4` under `logs/rsl_rl/platoon_happo/ams_spatial_complete_20260731_235300_attack_ladder/`.
- Confirmed no Isaac training, evaluation, or playback process remains active, then opened the attack-ladder root directory in the desktop file manager so all five level folders, data, plots, and videos can be inspected directly.

## 2026-08-03 Ackermann Six-Algorithm Run Status Audit

- The supervisor `ackermann_six_full_20260801_165233` is still alive, but the experiment has **not** advanced beyond the clean/off stage. All six clean foundations completed 3000 updates and saved `model_final.pt`: MAPPO, HAPPO without Meta, HAPPO+Meta+AMS, shared MAPPO, HAA2C, and HATRPO.
- None of the six clean training runs ever achieved the strict whole-platoon completion condition. Maximum recorded `platoon_completion_ever_rate` is `0.0` for every method. Maximum mean path progress ranged from `0.0028` (HATRPO) to `0.8353` (HAA2C), while maximum minimum-vehicle progress ranged from `0.0` to `0.5700`; therefore the last vehicle never completed the path in any method.
- The clean/off evaluator is stalled on the first algorithm. `model_0` produced all 3500 requested rows by `2026-08-02 08:10:45`, but the evaluator then failed to create/evaluate the fresh environment for `model_final.pt`. The same process has consumed one CPU core for over 27 hours with no log or artifact modification, low GPU utilization, and no second CSV. This is a no-progress hang, not a long valid evaluation.
- The likely immediate hang mechanism is `--fresh_env_per_checkpoint` recreating an Isaac environment inside the same Python/SimulationApp process. The first checkpoint CSV is written only after the first environment closes; the process then spins while trying to construct the second environment. Future evaluation must use one environment per process or reuse a single environment, not recreate SimulationContext inside one process.
- The completed clean training itself is also not acceptable as a maneuver foundation: all algorithms have zero whole-platoon completion. Consequently no light/easy/medium/hard stage has started, no hard-b gate has been attempted, and the promised final videos/trajectory/comparison matrix do not yet exist.
- The stuck evaluator remains active at audit time because this turn asked only for status; no process or source was mutated during the audit. Before resuming useful experiments, stop the stalled process, correct the evaluator lifecycle, and diagnose/rebalance clean maneuver acquisition rather than continuing the attack ladder from failed foundations.

## 2026-08-03 Why The Whole Platoon Did Not Complete

- Episode duration is not the fundamental constraint. Vehicle 5 starts about 13 m before the path end; at the minimum command speed `0.30 m/s`, ideal traversal takes about `43.3 s`, leaving about `26.7 s` inside the configured 70-second episode for settling and the 2-second completion dwell.
- The dominant failure is early episode reset caused by poor path/road tracking. Observed mean `done_rate` is about `0.0015-0.0066` per control step, much larger than the `1/3500 = 0.000286` rate expected from 70-second timeout alone. For our HAPPO+Meta+AMS clean run it is `0.00476`, corresponding to a rough reset interval near 210 steps or `4.2 s`; the leader reaches the path entrance after roughly 2-3 seconds, so many environments reset shortly after steering begins.
- Bad-heading termination is essentially zero, while road-boundary violation is non-zero and material-departure termination is the remaining early-reset mechanism. Our run has mean any-vehicle boundary violation about `13.8%` and reaches about `59.4%` in some logged states; its centerline and pair-lateral tracking errors also become large. The apparent logged max exceedance just below `0.10 m` is consistent with the environment resetting immediately once the termination threshold is crossed and metrics then observing reset state.
- The failure therefore occurs before the strict dwell/gap gate matters. Across all 3000 updates, our maximum minimum-vehicle progress is only `0.242`; at its best-progress row the per-vehicle mean progress is approximately `[0.569, 0.462, 0.388, 0.310, 0.242]`, with centerline error about `0.531 m` and boundary violation in about half the environments. No vehicle-5 cohort reaches progress `0.99`, so increasing dwell time or merely extending the episode would not solve the current behavior.
- Corrective priority is: fix/soften the clean acquisition reset curriculum so vehicles can learn the full curve, validate Ackermann path tracking and Shield steering on a deterministic controller, then reinstate strict road termination and the whole-platoon dwell gate. Only after clean completion should attack training resume.

## 2026-08-03 Progressive Clean Curriculum Launched

- User authorized a clean-only progressive curriculum and explicitly deferred every attack stage until the whole platoon can complete the maneuver.
- Stopped the stalled `ackermann_six_full_20260801_165233` supervisor and its separately re-parented evaluator. All six failed clean checkpoints and the completed model-0 diagnostic CSV remain preserved; no attack stage from that invalid run was accepted.
- Created a new recovery snapshot at `backups/20260803_progressive_clean_preedit/`. SHA-256 is `b73ca9c...` for the tracked worktree diff and `6815cb6a...` for the selected-source archive.
- Added `scripts/tools/run_ackermann_progressive_clean.sh`. It trains only `happo_meta_ams` with attack exactly off and uses three monotonic road-termination phases: `acquire_relaxed` threshold `10.0 m`, `consolidate` threshold `0.30 m`, and final `strict` threshold `0.10 m`. Training budgets are `3000 + 1200 + 1200` updates with checkpoint continuation.
- The spatial path, 70-second horizon, five-car geometry, and final acceptance criteria remain unchanged. Training guidance was strengthened without weakening evaluation: centerline penalty `-2.0`, heading penalty `-1.5`, road-envelope penalty `-8.0`, and sparse whole-platoon success reward `+2.0`. Strict completion still requires all five progress `>=0.99`, center/heading/gap/pair/collision/road tolerances, and a 2-second dwell.
- Evaluation now reuses one Isaac environment for the two selected checkpoints (`FRESH_ENV_PER_CHECKPOINT=0`) instead of recreating SimulationContext inside one process, avoiding the prior fresh-environment hang. The gate uses peak per-episode-ever completion rate, so a success just before the 70-second timeout is not erased by the terminal reset row.
- Formal clean-only tag `ackermann_progressive_clean_20260803_115200` launched under supervisor PID `57257`; active Isaac PID was `57274`. Startup audit confirms 64 environments, five agents, `act_dim=2`, joint action dimension 10, AMS enabled, attack off, and relaxed boundary termination `10.0 m`. The first three updates and metrics are finite. Attack training and attack videos remain disabled until `clean_gate.txt` reports at least `0.50` strict whole-platoon completion.

## 2026-08-03 Hourly Monitoring Enabled

- Created an active recurring monitoring goal at the user's request. Approximately once per hour it will check process liveness, stage/update progress, minimum five-vehicle path progress, strict completion/dwell rate, road violation/reset behavior, numerical/runtime errors, and evaluator forward progress. Every material result will be appended to this file; attacks remain prohibited before the clean gate passes.
- Initial monitor at `2026-08-03 11:55:46 +08:00`: supervisor `57257` and Isaac process `57274` are healthy; acquisition is at update `82/3000`, about `3.3 s/update`, ETA roughly `2 h 39 min`. No Traceback/CUDA/non-finite error is present.
- Early acquisition has already reached mean path progress `0.585` and minimum-vehicle progress `0.264`, substantially beyond the prior AMS late-training collapse, but strict completion remains `0`. The relaxed phase is behaving as designed: road termination is zero while the learner experiences/reduces boundary error; current any-vehicle road violation is `0.391` and max footprint exceedance `0.271 m`, so this is promising acquisition evidence but not yet an acceptable in-lane maneuver.
- Immediate scheduler confirmation at `11:56` (not counted as the next hourly sample): update advanced from `82` to `91`; mean/minimum-vehicle path progress rose to `0.647/0.397`. Processes and metric timestamps are advancing, AMS values are finite, and no runtime error appeared. Boundary exceedance temporarily increased to `0.357 m`, which is permitted only in acquisition; strict completion remains zero. Next material sample is due around `12:55 +08:00`.
- Installed and activated a real wall-clock user-level systemd timer, `ackermann-progressive-clean-monitor.timer`, so monitoring does not depend on conversation timing. It first triggers at `2026-08-03 12:57:36 +08:00` and repeats every hour. Raw read-only snapshots are appended to `logs/rsl_rl/platoon_happo/ackermann_progressive_clean_20260803_115200/hourly_monitor.jsonl`; each later review will append its analyzed conclusion here.
- Manual timer validation snapshot at `11:57:36`: acquisition reached update `115/3000`; process and metrics are live, minimum-vehicle/mean path progress are `0.613/0.748`, and AMS Q loss is finite. Strict instant/ever/dwell completion remain `0`. No road termination, traceback, CUDA error, or non-finite failure was detected. The maximum observed footprint exceedance has risen to `0.673 m` in the intentionally relaxed acquisition phase; this is not accepted as success and its hourly trend must improve before consolidation/strict gating.
- Pre-hour scheduler audit at `11:58:35` confirmed the timer remains `active (waiting)`, its one-shot service exited normally after writing exactly one validation snapshot, and the next trigger is still `12:57:36 +08:00`. Only one minute had elapsed, so no duplicate trend sample or intervention was recorded.
- Strengthened the monitor before its first scheduled hour: every timer invocation now records both the launcher/supervisor and Isaac worker state and appends a concise Chinese health line directly to this notebook in addition to the full JSONL snapshot. A missing process, metrics older than 15 minutes, or a matched runtime error is labeled `警告`; each line includes stage/update, five-car minimum/mean progress, all three strict completion rates, boundary/reset metrics, and clean-gate state. File locking and `fsync` protect notebook appends. Python compilation, summary formatting, and the transient systemd service's exact script target were verified; the test did not create a duplicate snapshot.
- Added explicit CSV numerical-integrity monitoring before the first scheduled hour. Every tracked metric is now checked for unparsable, `NaN`, and `Inf` values; findings are retained in JSON and force the notebook status to `警告`. Compilation passed, and a direct scan of the live acquisition CSV at update/row `165` found zero non-finite values. The timer remains active with its first scheduled trigger at `12:57:36 +08:00`.
- Refined forward-progress/stall monitoring to be phase-aware. The monitor now classifies the live Isaac work as `training`, `evaluation`, `playback`, or `none`, records core-process age, and tracks the newest file under the active phase's evaluation directory. Training is considered stalled only when its metrics are older than 15 minutes; evaluation is considered stalled only when it has produced no artifact within 15 minutes of launch or its newest artifact has stopped changing for 15 minutes. This avoids treating a valid evaluator as stale merely because training metrics are intentionally frozen. Compilation and live helper checks passed; current mode is `training`, the supervisor is present, and no evaluation artifact is expected yet.
- Added an explicit clean-isolation invariant to every hourly snapshot. Until `clean_gate.txt` contains `status=passed`, the monitor now flags (1) any Isaac train/play/evaluator not belonging to this tag, (2) any active command with an attack level other than `off` or non-zero FDI/DoS maxima, and (3) any attack-launcher shell process. Any occurrence forces `警告` and the three counts are included in the notebook line. Compilation and positive/negative parser tests passed. The live audit finds two expected wrapper/Isaac processes for this tag, zero foreign Isaac processes, zero non-zero attack configurations, and zero attack pipelines.
- 自动小时监测 `2026-08-03 12:06:31 +0800` [健康]：阶段 `acquire_bounded`，update `115`，模式 `training`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `1.1/n/a s`，停滞 `False`；逐车路径进度 `[0.342/0.120/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.092`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.000/0.000/0.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.309/0.222/0.298/0.143/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，clean gate `未生成（禁止攻击实验）`。
- Recovery intervention at approximately `12:04-12:06`: the original `10.0 m` relaxed phase was stopped because it had entered a sustained unsafe divergence, not because of a runtime crash. From updates `161 -> 181 -> 201 -> 222`, maximum footprint exceedance grew `0.98 -> 1.81 -> 3.52 -> 6.20 m`; any-vehicle violation reached `0.969`, center error reached `1.32 m`, and AMS Q loss rose to `9.48`, while strict completion remained zero. Sending SIGINT caused the generic pipeline to enter evaluation, so the exact process group `57257` was then terminated; all checkpoints and logs remain preserved.
- Selected the last useful pre-divergence checkpoint `model_100.pt` (SHA-256 `73feb62d99caa65a5ee93e15e48d3b8b6af479f6bd59391fffcaa521f275c9ed`) as an immutable symlinked recovery seed. Added opt-in `RECOVERY_MODE=1` to the clean launcher: it resumes through a new `acquire_bounded` phase for 2000 additional updates with road termination tightened to `0.75 m`, followed by the unchanged `0.30 m` consolidate and `0.10 m` strict phases. Normal non-recovery behavior is unchanged.
- Recovery service `ackermann-progressive-clean-recovery.service` started under the original experiment root and resumed at iteration 101. Post-start validation at update 115 shows live supervisor/Isaac processes, finite AMS values, no errors, no foreign or attack process, and zero boundary violation/exceedance/road termination in the first 14 recovery rows. These rows cover only the beginning of a rollout, so they prove a healthy restart but not yet whole-platoon acquisition; the scheduled hourly monitor remains authoritative for the trend.
- Short post-recovery safety check at `12:07:14` (not an hourly trend sample): update `127` has sequential per-car progress `[0.753, 0.530, 0.323, 0.128, 0.000]`. Current/max boundary exceedance is only `0.116 m`, current any-car violation is `0.031`, and mean violation over all 26 recovery rows is `0.0024`; road termination and total reset remain zero. AMS Q loss is finite (`0.0194` current). Thus the `0.75 m` bound is presently preventing the former multi-meter departure without immediately causing reset churn. Strict completion remains zero because vehicle 5 has not yet reached the path entrance; the first scheduled hourly sample is still required before accepting the recovery trend.
- Scheduler-only check at `12:07:38`: the bounded-recovery service remains active and the monitor timer remains active/waiting for `12:57:36`. The JSONL contains the initial validation and post-recovery validation only; no duplicate sub-hour trend sample or training intervention was made.
- Added behavioral-divergence alarms to the hourly monitor so `健康` no longer means only process/numerical health. It now warns on multi-meter boundary departure, bounded-phase exceedance beyond its expected envelope, persistent widespread boundary violation, AMS Q-loss spikes, high reset/road-termination rates, and high collision rate. Regression against the stopped relaxed CSV correctly reports `multi_meter_boundary_departure=8.902m`, persistent any-vehicle violation (tail mean `0.878`), and AMS Q-loss spike (`8.115`). The live bounded-recovery CSV at update `150` triggers none of these alerts. Python compilation passed; no extra snapshot was written for this unit-style test.
- First recovery check after vehicle 5 entered the spatial path, at `12:08:56` / update `157` / simulated rollout time `35.84 s`: per-car progress is `[0.825, 0.768, 0.725, 0.676, 0.612]`, with strict completion still zero. The bound prevents renewed runaway but tracking is not yet acceptable: over the last 20 updates, maximum exceedance averages `0.514 m` and peaks at `0.746 m` (just below the `0.75 m` termination threshold), any-car violation averages `0.486`, and road-termination/reset rates average `0.00234/0.00156`. Current AMS Q loss is finite (`0.301`) and collision is zero. Interpretation: the bound is clipping unsafe rollouts as intended, but the policy has not yet learned in-lane completion; no further intervention is justified from this short window, and the scheduled hourly trend must show whether these errors fall.
- Follow-up at `12:09:34`, update `168`, simulated time `42.88 s` (still not a full 70-second terminal cycle): per-car progress is `[0.840, 0.767, 0.711, 0.660, 0.615]`, all completion rates remain zero, and the all-time minimum-vehicle progress peak is `0.634`. Over the last 20 rows, any-car boundary violation is `0.530`, max exceedance is `0.580 m`, and road-termination rate has risen to `0.01094`, just above the monitor's behavioral warning threshold; total reset mean remains only `0.00078`, while AMS Q loss is finite (`0.199` current, `0.224` tail mean). This is a yellow signal that the bound may be clipping too many rollouts and progress is plateauing, but the window is too short for another rollback; the hourly sample must determine persistence.
- Scheduler-only check at `12:10:13`: bounded recovery and the monitor timer remain active; next timer trigger is `12:57:36`. Less than one minute had elapsed since the yellow-signal analysis, so no duplicate metric sample or intervention was made.
- Scheduler-only check at `12:10:48`: the recurring monitoring goal is active, bounded recovery remains active, and the timer is waiting normally with 46 minutes to the first formal hourly trigger. No new trend conclusion was possible within this 35-second interval, so no metric resampling or training mutation was performed.
- Waiting-state audit at `12:11:06`: after three consecutive sub-minute continuations with no new hourly evidence, the agent-side recurring goal is marked blocked on the required external-state change (real time reaching the next hourly trigger), rather than continuing statistically meaningless second-scale sampling. This does **not** stop the experiment or monitoring: `ackermann-progressive-clean-recovery.service` remains active, `ackermann-progressive-clean-monitor.timer` remains active/waiting for `12:57:36`, and the timer will autonomously append both JSON and a readable notebook conclusion. Resume agent analysis after that snapshot appears.
- 自动监测 `2026-08-03 12:57:56 +0800` [警告]：阶段 `acquire_bounded`，update `1034`，模式 `training`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `2.1/n/a s`，停滞 `False`；逐车路径进度 `[0.065/0.007/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.014`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.156/0.583/0.078/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.569/0.588/0.300/0.401/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=0.0569`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 13:58:56 +0800` [警告]：阶段 `acquire_bounded`，update `2101`，模式 `evaluation`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `80.0/n/a s`，停滞 `False`；逐车路径进度 `[0.797/0.588/0.406/0.228/0.103]`，五车最低/平均路径进度 `0.103/0.424`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.281/0.619/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.692/0.347/0.441/0.478/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 14:59:56 +0800` [警告]：阶段 `acquire_bounded`，update `2101`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `3740.0/3107.4 s`，停滞 `False`；逐车路径进度 `[0.797/0.588/0.406/0.228/0.103]`，五车最低/平均路径进度 `0.103/0.424`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.281/0.619/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.692/0.347/0.441/0.478/0.000`；运行错误/非有限指标 `2/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 16:00:56 +0800` [警告]：阶段 `acquire_bounded`，update `2101`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `7400.0/6767.4 s`，停滞 `False`；逐车路径进度 `[0.797/0.588/0.406/0.228/0.103]`，五车最低/平均路径进度 `0.103/0.424`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.281/0.619/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.692/0.347/0.441/0.478/0.000`；运行错误/非有限指标 `2/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 17:01:56 +0800` [警告]：阶段 `acquire_bounded`，update `2101`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `11060.0/10427.4 s`，停滞 `False`；逐车路径进度 `[0.797/0.588/0.406/0.228/0.103]`，五车最低/平均路径进度 `0.103/0.424`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.281/0.619/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.692/0.347/0.441/0.478/0.000`；运行错误/非有限指标 `2/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 18:02:56 +0800` [警告]：阶段 `acquire_bounded`，update `2101`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `14720.0/14087.4 s`，停滞 `False`；逐车路径进度 `[0.797/0.588/0.406/0.228/0.103]`，五车最低/平均路径进度 `0.103/0.424`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.281/0.619/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.692/0.347/0.441/0.478/0.000`；运行错误/非有限指标 `2/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。

## 2026-08-03 Hourly Visibility Failure, Missed Alert Review, and Recovery

- The hourly timer did run successfully at `12:57`, `13:58`, `14:59`, `16:00`, `17:01`, and `18:02`; all six snapshots were written to `hourly_monitor.jsonl` and readable warning lines were appended above. They were not visible in chat because a local systemd timer can write local files but cannot initiate a new assistant message in an idle conversation. In addition, the agent-side goal was incorrectly marked blocked while waiting for wall-clock time, so no agent turn surfaced the later warnings. This was a delivery-design failure, not a missing timer execution.
- The missed warnings were material. At `12:57`, bounded training at update `1034` had collapsed to minimum/mean progress `0.000/0.014` with road-termination tail alarm `0.0569`. At `13:58`, training had reached update `2101` and evaluation was active, but strict completion remained zero. By `14:59`, both supervisor and Isaac were dead, and subsequent hours correctly repeated the stopped-state warning.
- Root cause of the pipeline exit was identified precisely: both selected 3500-step clean evaluations completed and `eval_summary.csv` was written, after which `plot_medium_algorithm_comparison.py` failed under the ambient `/home/cnc/Downloads/ENTER/bin/python3` with `ModuleNotFoundError: No module named 'matplotlib'`. The final bounded evaluations were invalid as a clean gate: completion peak `0`, final minimum progress `0`, road-boundary termination `0.864` for `model_2000.pt` and `1.000` for `model_final.pt`.
- Fixed the comparison launcher to execute plotting with Isaac Python, which has `matplotlib 3.10.3`. Reused the completed evaluations to generate all three comparison figures and `combined_eval_summary.csv`, then preserved the phase as a `24 MB` `acquire_bounded.tar.gz`; no training or evaluation was rerun.
- Added `START_PHASE` support to the progressive-clean launcher and resumed the original clean lineage directly at `consolidate`, using the bounded phase's existing `model_final.pt`. Service `ackermann-progressive-clean-consolidate.service` started at `18:44:54`, threshold is `0.30 m`, budget is 1200 updates, and attack remains exactly off/zero. Startup validation found live supervisor/Isaac processes, update `2103`, finite AMS Q loss `0.0248`, and no startup error.
- Corrected visibility for the local workstation: the hourly monitor now sends a desktop notification in addition to JSON and notebook output. The user systemd environment has a valid DBus session and `DISPLAY=:1`; a test notification was sent successfully. The next automatic trigger is `19:02:56`. This provides visible local alerts, but local systemd still cannot autonomously post into an idle chat; chat summaries occur when the conversation is active.
- 自动监测 `2026-08-03 19:03:56 +0800` [警告]：阶段 `consolidate`，update `2434`，模式 `training`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `1.0/n/a s`，停滞 `False`；逐车路径进度 `[0.030/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.006`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.359/0.297/1.000/0.016`，完成约束中心/航向/间距/横向/碰撞 `0.268/0.365/0.260/0.305/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 19:45:37 +0800` [警告]：阶段 `consolidate`，update `3181`，模式 `training`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `2.3/n/a s`，停滞 `False`；逐车路径进度 `[0.074/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.015`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.312/0.286/1.000/0.016`，完成约束中心/航向/间距/横向/碰撞 `0.269/0.299/0.386/0.420/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- Current-status interpretation at approximately `19:45`: consolidate is live and not stalled, at iteration `3180/3299`, about `3.36 s/update`, with training ETA approximately `6m34s` before its clean evaluation. Numerical values remain finite and there are no runtime/CUDA errors or attack processes. Behavior is nevertheless failing: the last 50 rows have minimum-vehicle progress `0`, strict completion `0`, and road-boundary termination `1.0`; current per-car progress is approximately `[0.074,0,0,0,0]`. Because only about 120 updates remain, the stage is allowed to finish so its fixed 3500-step evaluation can provide authoritative evidence; it is not accepted as successful and attacks remain prohibited.
- 自动监测 `2026-08-03 20:04:56 +0800` [警告]：阶段 `strict`，update `3327`，模式 `training`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `3.0/n/a s`，停滞 `False`；逐车路径进度 `[0.002/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.000`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.000/0.000/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.115/0.271/0.316/0.360/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=0.7843`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 21:05:56 +0800` [警告]：阶段 `strict`，update `4426`，模式 `training`，监督/Isaac存活 `True/True`，训练指标/评估产物延迟 `0.7/n/a s`，停滞 `False`；逐车路径进度 `[0.089/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.018`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.000/0.000/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.329/0.390/0.451/0.511/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `未生成（禁止攻击实验）`。
- 自动监测 `2026-08-03 22:06:56 +0800` [警告]：阶段 `strict`，update `4501`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `3413.9/2779.2 s`，停滞 `False`；逐车路径进度 `[0.133/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.027`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.047/0.088/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.587/0.621/0.664/0.866/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `status=failed`。
- 自动监测 `2026-08-03 23:07:56 +0800` [警告]：阶段 `strict`，update `4501`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `7073.9/6439.2 s`，停滞 `False`；逐车路径进度 `[0.133/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.027`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.047/0.088/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.587/0.621/0.664/0.866/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `status=failed`。
- 自动监测 `2026-08-03 23:12:22 +0800` [警告]：阶段 `strict`，update `4501`，模式 `none`，监督/Isaac存活 `False/False`，训练指标/评估产物延迟 `7340.2/6705.5 s`，停滞 `False`；逐车路径进度 `[0.133/0.000/0.000/0.000/0.000]`，五车最低/平均路径进度 `0.000/0.027`，严格瞬时/曾完成/dwell率 `0.000/0.000/0.000`；道路任一车越界率/最大超界/道路终止/总重置率 `0.047/0.088/1.000/0.000`，完成约束中心/航向/间距/横向/碰撞 `0.587/0.621/0.664/0.866/0.000`；运行错误/非有限指标 `0/0`，外来Isaac/非零攻击/攻击流水线 `0/0/0`，行为警报 `high_road_termination_rate=1.0000`，clean gate `status=failed`。

## 2026-08-03 Progressive Clean Final Result

- The complete recovered clean curriculum finished normally through `acquire_bounded`, `consolidate`, and `strict`; there is no active supervisor, Isaac trainer, or evaluator, and `clean_gate.txt` is `status=failed`. This is a behavioral/experiment failure, not a process hang or numerical crash. All stage logs, checkpoints, two-checkpoint 3500-step evaluations, figures, and tar archives are preserved.
- Consolidate formal evaluation: both `model_2400.pt` and `model_final.pt` have strict whole-platoon completion peak `0`, minimum-vehicle progress mean/final `0`, and trajectory progress mean only `0.00674/0.00623`. Road-boundary termination is `0.936/1.000`, despite zero collisions.
- Strict formal evaluation: both `model_3600.pt` and `model_final.pt` again have completion peak `0` and minimum-vehicle progress `0`. Mean trajectory progress is only `0.00259/0.02479`; road-boundary termination is `0.951/1.000`. The final checkpoint has center/heading/gap errors `0.485/0.507/0.537`, so it is not close to the gate.
- Interpretation: tightening termination from `0.75 -> 0.30 -> 0.10 m` did constrain maximum reported exceedance to each threshold, but converted most exploratory lane-change rollouts into early road-boundary termination. The policy collapsed toward near-start behavior instead of learning the complete five-car path. More iterations of this same termination curriculum are not justified; the next clean attempt requires a different acquisition mechanism rather than proceeding to attacks.
- Because training and evaluation are complete and the final gate failed, the hourly timer was stopped after this final status was recorded. Attacks remain prohibited.

## 2026-08-03 Root-Cause Diagnosis After Progressive Clean Failure

- The failure is not primarily a road-width, Ackermann-joint, runtime, or numerical problem. The corrected road corridor contains both requested lane centers and the real Ackermann smoke previously demonstrated feasible forward/curved motion. The spatial reference, policy observation, reward centerline term, and Shield `_env_local_y()` all use each vehicle's own `actual_y - reference_y`; an initial suspicion that the Shield pulled toward global `y=0` was checked and rejected.
- The dominant failure is **episode-survival starvation caused by all-vehicle hard road termination**. Formal strict evaluation has per-step `done_rate` `0.00569` for `model_3600` and `0.00240` for `model_final`, corresponding roughly to only `176-416` steps or `3.5-8.3 s` between resets. Vehicle 5 starts about `7 m` before the path entrance and `13 m` before the path end; at approximately `0.38 m/s` it needs about `18 s` merely to enter and over `34 s` to finish. Thus the full platoon is reset long before vehicle 5 can generate a completion sample.
- The reset hazard is coupled across five cars: `termination_road_boundary_exceeded()` takes the maximum footprint exceedance over all five and resets the entire environment when any one crosses the threshold. Even moderate per-car tracking error therefore makes survival to the fifth vehicle's path phase exponentially harder. The evaluation field `termination_road_boundary_exceeded=0.95-1.0` is the reason stored for nearly every environment's most recently completed episode; it is not a literal per-step reset probability. `done_rate` is the correct quantity for estimating reset frequency.
- The curriculum advances in the wrong direction without phase gates. `run_ackermann_progressive_clean.sh` proceeds from `0.75 -> 0.30 -> 0.10 m` whenever the clean completion gate is missed; it does not require a minimum survival time, vehicle-5 progress, or boundary-safe completion before tightening. Therefore a policy with zero completion at `0.75 m` is made strictly harder instead of remaining at the acquisition phase or rolling back.
- Reward/selection is also misaligned with the terminal task. Strict `model_3600` obtains evaluation return about `65185` while mean path progress is only `0.00259` and completion is zero, showing that the fixed-step accumulated dense return can prefer repeated short forward/reset cycles. `reward_true_collision_fail()` penalizes only inter-vehicle collision and gives no sparse failure cost for road termination; `true_success` remains zero because the five-car completion state is never reached. Consequently frequent road resets are not represented as the decisive outer failure they should be.
- The required correction is not “more iterations” of the same schedule. Acquisition must first guarantee long-horizon survival and expose vehicle 5 to the full path: use a path-capable deterministic/expert or imitation warm start (or progressively introduce vehicles/path length), keep hard road termination disabled during this warm start while retaining soft boundary cost/Shield, then gate every tightening step on measured survival horizon, vehicle-5 progress, and strict evaluation. Add an explicit road-termination terminal cost and select checkpoints by completion/survival/safety rather than fixed-step return. Only after a checkpoint completes the clean maneuver should `0.30/0.10 m` fine-tuning and attacks be considered.

## 2026-08-03 Deterministic Five-car Ackermann Clean Completion

- Implemented an opt-in deterministic clean-acquisition controller in `algorithms/shield.py`, routed through environment `SafetyShieldCfg`. Each of the five cars queries the same quintic spatial path at its own x position and receives a 2-D Ackermann action `[rear-wheel speed, front-knuckle steering]`; rear steering is never commanded. The feature is off by default, so existing learned-policy experiments are unchanged unless `env.safety_shield.expert_path_tracking_enabled=true` is supplied.
- Ran a real Isaac single-environment 3500-step/70 s strict clean evaluation with attacks disabled, road-boundary enforcement enabled, and the road termination exceedance fixed at `0.10 m`. Strict whole-platoon instantaneous completion first occurred at step `1717` (`34.34 s`); the required 2 s dwell completion first occurred at step `1816` (`36.32 s`), with `platoon_completion_dwell_rate=1.0`.
- At dwell completion, the five-car minimum progress was `1.000`; max center/heading/gap/pair-lateral errors were `0.00561 m / 0.00507 rad / 0.26799 m / 0.00870 m`. Road-boundary violation, maximum road exceedance, and collision were all exactly `0`. Thus this is the first verified complete whole-platoon lane change under the strict clean geometry, not a timeout-shortened partial maneuver.
- Exported a real 44 s Isaac-rendered replay plus 11,000-row pose/control CSV, and a corrected fixed high-angle whole-platoon video (`38 s`, `1280x720`, `50 fps`). Generated overhead trajectory and front-steering/tracking plots in `logs/ackermann_expert_smoke/evidence/figures`. Full evidence index: `logs/ackermann_expert_smoke/evidence/RESULT.md`.
- Interpretation and next stage: this proves the path, corridor, vehicle geometry, and Ackermann actuation can complete the maneuver. It does not convert the failed checkpoint into a converged policy because expert projection is authoritative. Use this rollout as imitation/warm-start data, then disable/anneal expert authority and require the identical strict clean gate before starting the six-algorithm attack matrix.

## 2026-08-04 Real-Trajectory Imitation Warm Start and Strict Clean Gate

- Reconstructed the exact 11-dimensional per-vehicle policy observation from the real Isaac replay CSV and behavior-cloned the shared Ackermann actor on 10,995 samples. The validation speed/steering MAE is `0.0007235/0.006687` normalized (physical steering MAE approximately `0.00214 rad`). The reproducible trainer is `scripts/tools/pretrain_ackermann_bc.py`; output is `logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt` with SHA-256 `6b8c15acf17b7e250a65fc85609c304ba2cc14f8d0a65d8f34b4188f6e801615`.
- The warm-start checkpoint now resets task-local and outer update counters to zero, removes failed-policy actor/critic/AMS Adam moments, and resets AMS sampler chronology. All six algorithms therefore receive the same BC actor without silently inheriting failed-clean optimization history.
- Added configurable expert blending to the Safety Shield. The validation schedule held 100% expert for 200 steps, linearly withdrew it to 0% at step 1200, and kept it off thereafter. The whole platoon first completed at step `1691` and satisfied the two-second dwell at step `1790`; both occurred after complete expert withdrawal. Collision, road departure, road exceedance, and road termination were all zero.
- Pure-policy strict clean evaluation with expert completely disabled passed at step `1791` (`35.82 s`) in the 1-env/seed-42 run. A separate 8-env/seed-3407 evaluation with randomized command speed `0.30-0.45 m/s` also reached completion in every environment, with zero collisions, road departures, exceedance, or road terminations. This demonstrates policy completion rather than authoritative expert completion.
- Added `scripts/tools/verify_ackermann_clean_gate.py`. Formal gate `logs/ackermann_bc_warmstart/clean_gate.txt` is `status=passed`, records expert zero at step `1200`, dwell at step `1790`, and sets `attacks_unlocked=1`. Python compilation, shell syntax, clean warm-start state assertions, and `git diff --check` passed.
- Corrected `run_ackermann_full_six_attack_matrix.sh`: it now refuses to run without the passed gate, starts MAPPO/HAPPO-no-Meta/HAPPO-Meta-AMS/HARL-MAPPO-shared/HARL-HAA2C/HARL-HATRPO from identical BC checkpoints, skips the failed from-scratch foundation, uses checkpoint evaluation without the known fresh-environment hang, and forcibly disables expert tracking/blending throughout attack training and evaluation.
- After the clean gate passed, launched the formal serial light-to-hard-b matrix at `2026-08-04 04:03:15 +0800`. Tag/root is `ackermann_bc_six_attack_20260804_040303`; systemd service `ackermann-bc-six-attack-20260804.service` is active and began `light/mappo` with one Isaac process and no startup error. Final ladders will export all algorithms and attack levels as CSV data, figures, true top-down trajectories, real renderer videos, manifests, and checksums.
- Added hourly matrix health monitoring. Timer `ackermann-bc-six-monitor-20260804.timer` next triggers at `05:03:15 +0800`; snapshots go to the matrix `hourly_monitor.jsonl`, visible desktop notifications, and this notebook. An immediate snapshot at `04:03:25` confirmed active training, zero completed evaluations (startup), and no errors.
- Post-startup verification located run `2026-08-04_04-03-31_ackermann_bc_six_attack_20260804_040303_light_mappo`; the log explicitly confirms the BC task-local HAPPO state loaded successfully. Training reached update `8` with finite metrics, road-boundary-any `0`, collision `0`, one intended Isaac job, no traceback/runtime/CUDA error, and an initial ETA of approximately `31 min` for this first 600-update algorithm stage.
- 自动监测 `2026-08-04T04:03:25+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-04T05:03:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_04-34-01_ackermann_bc_six_attack_20260804_040303_light_happo_no_meta`，update `592`，已完成评估 `0`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T06:04:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_05-38-05_ackermann_bc_six_attack_20260804_040303_light_harl_mappo_shared`，update `575`，已完成评估 `0`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T07:05:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_06-37-31_ackermann_bc_six_attack_20260804_040303_light_harl_hatrpo`，update `531`，已完成评估 `0`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T08:06:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-04_06-37-31_ackermann_bc_six_attack_20260804_040303_light_harl_hatrpo`，update `600`，已完成评估 `5`，最近完成率/道路越界/碰撞 `0.0/0.015625/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T09:07:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_08-45-43_ackermann_bc_six_attack_20260804_040303_easy_0_happo_no_meta`，update `1041`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.34375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T10:08:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_09-50-09_ackermann_bc_six_attack_20260804_040303_easy_0_harl_mappo_shared`，update `1001`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T11:09:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_10-48-58_ackermann_bc_six_attack_20260804_040303_easy_0_harl_hatrpo`，update `990`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T12:10:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-04_10-48-58_ackermann_bc_six_attack_20260804_040303_easy_0_harl_hatrpo`，update `1200`，已完成评估 `10`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T13:11:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_12-56-27_ackermann_bc_six_attack_20260804_040303_easy_a_happo_no_meta`，update `1502`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.3125/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T14:12:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_14-01-33_ackermann_bc_six_attack_20260804_040303_easy_a_harl_mappo_shared`，update `1441`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T15:13:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_15-00-15_ackermann_bc_six_attack_20260804_040303_easy_a_harl_hatrpo`，update `1454`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T16:14:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-04_15-00-15_ackermann_bc_six_attack_20260804_040303_easy_a_harl_hatrpo`，update `1800`，已完成评估 `15`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T17:15:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_17-08-29_ackermann_bc_six_attack_20260804_040303_easy_b_happo_no_meta`，update `1944`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.0/0.4375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T18:16:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_18-12-50_ackermann_bc_six_attack_20260804_040303_easy_b_harl_mappo_shared`，update `1884`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T19:17:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_19-11-27_ackermann_bc_six_attack_20260804_040303_easy_b_harl_hatrpo`，update `1920`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.0/0.21875/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T20:18:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-04_19-11-27_ackermann_bc_six_attack_20260804_040303_easy_b_harl_hatrpo`，update `2400`，已完成评估 `21`，最近完成率/道路越界/碰撞 `0.0/0.109375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T21:19:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_21-18-46_ackermann_bc_six_attack_20260804_040303_med_a_happo_no_meta`，update `2418`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.859375/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T22:19:58+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_21-49-42_ackermann_bc_six_attack_20260804_040303_med_a_happo_meta_ams`，update `2935`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `0`。
- 自动监测 `2026-08-04T23:20:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-04_22-52-04_ackermann_bc_six_attack_20260804_040303_med_a_harl_haa2c`，update `2944`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.140625/0.19140625`，错误数 `0`。
- 自动监测 `2026-08-05T00:21:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-04_23-24-12_ackermann_bc_six_attack_20260804_040303_med_a_harl_hatrpo`，update `3000`，已完成评估 `26`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T01:22:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_01-01-45_ackermann_bc_six_attack_20260804_040303_med_b_mappo`，update `3421`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T02:23:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_02-02-39_ackermann_bc_six_attack_20260804_040303_med_b_happo_meta_ams`，update `3375`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.109375/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T03:24:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_03-05-08_ackermann_bc_six_attack_20260804_040303_med_b_harl_haa2c`，update `3361`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.359375/0.14453125`，错误数 `0`。
- 自动监测 `2026-08-05T04:25:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-05_03-38-00_ackermann_bc_six_attack_20260804_040303_med_b_harl_hatrpo`，update `3600`，已完成评估 `31`，最近完成率/道路越界/碰撞 `0.0/0.078125/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T05:26:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_05-15-24_ackermann_bc_six_attack_20260804_040303_hard_a_mappo`，update `3823`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.140625/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T06:27:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_06-17-06_ackermann_bc_six_attack_20260804_040303_hard_a_happo_meta_ams`，update `3791`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.046875/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T07:28:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_07-18-48_ackermann_bc_six_attack_20260804_040303_hard_a_harl_haa2c`，update `3789`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.125/0.03125`，错误数 `0`。
- 自动监测 `2026-08-05T08:29:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-05_07-50-52_ackermann_bc_six_attack_20260804_040303_hard_a_harl_hatrpo`，update `4200`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.078125/0.09765625`，错误数 `0`。
- 自动监测 `2026-08-05T09:30:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_09-29-10_ackermann_bc_six_attack_20260804_040303_hard_b_r1_mappo`，update `4230`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T10:30:59+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_10-00-16_ackermann_bc_six_attack_20260804_040303_hard_b_r1_happo_no_meta`，update `4800`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.0625/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T11:31:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_11-04-46_ackermann_bc_six_attack_20260804_040303_hard_b_r1_harl_mappo_shared`，update `4783`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.078125/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T12:32:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_12-03-59_ackermann_bc_six_attack_20260804_040303_hard_b_r1_harl_hatrpo`，update `4746`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.328125/0.00390625`，错误数 `0`。
- 自动监测 `2026-08-05T13:33:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-05_12-03-59_ackermann_bc_six_attack_20260804_040303_hard_b_r1_harl_hatrpo`，update `4800`，已完成评估 `47`，最近完成率/道路越界/碰撞 `0.0/0.25/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T14:34:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_14-11-40_ackermann_bc_six_attack_20260804_040303_hard_b_r2_happo_no_meta`，update `5256`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.046875/0.00390625`，错误数 `0`。
- 自动监测 `2026-08-05T15:35:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_15-16-06_ackermann_bc_six_attack_20260804_040303_hard_b_r2_harl_mappo_shared`，update `5219`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T16:36:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_16-15-18_ackermann_bc_six_attack_20260804_040303_hard_b_r2_harl_hatrpo`，update `5212`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.21875/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T17:37:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-05_16-15-18_ackermann_bc_six_attack_20260804_040303_hard_b_r2_harl_hatrpo`，update `5400`，已完成评估 `52`，最近完成率/道路越界/碰撞 `0.0/0.3125/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T18:38:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_18-22-24_ackermann_bc_six_attack_20260804_040303_hard_b_r3_happo_no_meta`，update `5719`，已完成评估 `54`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.01953125`，错误数 `0`。
- 自动监测 `2026-08-05T19:39:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_19-27-07_ackermann_bc_six_attack_20260804_040303_hard_b_r3_harl_mappo_shared`，update `5671`，已完成评估 `54`，最近完成率/道路越界/碰撞 `0.0/0.109375/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T20:40:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_20-26-01_ackermann_bc_six_attack_20260804_040303_hard_b_r3_harl_hatrpo`，update `5679`，已完成评估 `54`，最近完成率/道路越界/碰撞 `0.0/0.265625/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T21:41:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-05_20-26-01_ackermann_bc_six_attack_20260804_040303_hard_b_r3_harl_hatrpo`，update `6000`，已完成评估 `58`，最近完成率/道路越界/碰撞 `0.0/0.359375/0.0`，错误数 `0`。
- 自动监测 `2026-08-05T22:42:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_22-32-55_ackermann_bc_six_attack_20260804_040303_hard_b_r4_happo_no_meta`，update `6190`，已完成评估 `60`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.00390625`，错误数 `0`。
- 自动监测 `2026-08-05T23:43:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-05_23-38-26_ackermann_bc_six_attack_20260804_040303_hard_b_r4_harl_mappo_shared`，update `6113`，已完成评估 `60`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T00:44:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-06_00-37-19_ackermann_bc_six_attack_20260804_040303_hard_b_r4_harl_hatrpo`，update `6137`，已完成评估 `60`，最近完成率/道路越界/碰撞 `0.0/0.234375/0.0`，错误数 `0`。

## 2026-08-06 Six-Algorithm Attack Matrix Status Audit

- At approximately `01:26 +0800`, service `ackermann-bc-six-attack-20260804.service` remains active after about 45 hours. It has completed all six-algorithm training/evaluation packages for `light`, `easy_0`, `easy_a`, `easy_b`, `med_a`, `med_b`, `hard_a`, and `hard_b_r1-r3`. All six `hard_b_r4` training runs are complete; evaluation is currently on `happo_no_meta`. The evaluator is actively using CPU/GPU and its 3500-row first-checkpoint CSV was updated within two minutes, so this is not a lifecycle hang. No traceback, runtime, CUDA, or non-finite error was found.
- The decisive behavioral finding is catastrophic forgetting immediately after the BC seed. In the first `light` evaluation, each algorithm's pre-update `model_0.pt` still completed the whole-platoon maneuver under light attack: ever-completion rates were MAPPO `0.5625`, HAPPO-no-Meta `0.8125`, HAPPO-Meta-AMS `0.71875`, shared MAPPO `0.65625`, HAA2C `0.875`, and HATRPO `1.0`. All had road-boundary-any `0`, collision `0`, and timeout-only done rate `1/3500`.
- After the first 600 on-policy updates under light attack, every algorithm's final-checkpoint whole-platoon completion became exactly `0`. Every evaluated checkpoint in all later `easy`, `medium`, `hard-a`, and completed `hard-b` rounds also has completion `0`. Thus the clean/BC policy itself was viable, but attack-time on-policy optimization destroyed the maneuver; this is not evidence that additional hard-b rounds are converging.
- The AMS hard-b acceptance statistic is exactly `0.0` in rounds 1, 2, and 3 versus the required `0.50`. Round 4 is still being evaluated and round 5 will run automatically if round 4 also fails. With the present trajectory, the launcher is expected to exit after round 5 without reaching the gate rather than generate the final ladders.
- Artifacts currently preserved under `logs/rsl_rl/platoon_happo/ackermann_bc_six_attack_20260804_040303`: 260 CSV files, 30 comparison PNGs, 132 checkpoints, and 10 stage archives. There are currently zero MP4 files and the `final_ladders` directory is empty because renderer videos/top-down final ladders are downstream of the hard-b gate.
- This turn was a read-only status audit, so the active round was not interrupted or modified. The script's missing per-stage retention gate is now evident: it allowed progression from light even after all trained final checkpoints lost clean/attack completion. The next corrective run should preserve or roll back to the best completion-capable checkpoint at every stage and constrain the policy away from the BC actor during attack fine-tuning, rather than repeatedly training the already-collapsed lineage.

## 2026-08-06 Recovery Protocol If The Current Matrix Fails

- Do not extend the collapsed hard-b lineage or declare more updates as recovery. Preserve the current run as negative evidence, return all methods to the immutable real-trajectory BC seed, and first evaluate that fixed seed at every attack level without learning. The existing light `model_0` results already prove this distinction is essential: the seed succeeds while the trained descendants fail.
- Before retraining, correct the shared on-policy credit-assignment path. The current pipeline samples/log-probs a raw policy action but the Safety Shield/attack postprocessor can execute a different action; PPO/A2C/TRPO then assign the resulting reward to an action different from the one applied to the Ackermann vehicle. All six methods collapsing together is consistent with this common action/reward mismatch. Either train an explicit residual policy around the safety controller with a well-defined executed-action likelihood, or ensure the rollout buffer/log-prob corresponds to the actual action used for environment transition. Do not simply store a transformed action with the old raw-action log-prob.
- Add maneuver-retention loss during attack fine-tuning: replay the expert/BC observation-action pairs every update and optimize `RL loss + lambda_BC * action imitation + lambda_KL * distance from BC policy`. Keep expert execution off; expert data is an auxiliary training target only. Start with actor learning rate reduced by roughly 10x, short 25-50 update blocks, gradient clipping, and optionally freeze the shared feature layers/speed head during the first attack blocks.
- Mix clean and attacked rollouts in every block and ramp attack probability/magnitude continuously from zero to the stage target. A stage may advance only if both the strict clean regression gate and the current attack gate pass. Suggested retention/attack gates are clean ever/dwell `1.0`, light `>=0.80`, easy `>=0.70`, medium `>=0.60`, and the declared hard-b target for AMS `>=0.50`, with explicit road/collision limits.
- Evaluate every short block, keep the checkpoint with the best lexicographic score `(completion, safety, minimum-vehicle progress)`, and roll back immediately if completion falls. On failure, reduce learning rate/attack ramp and retry from the last passing checkpoint; never resume from the failed final checkpoint and never advance difficulty merely because the update budget ended.
- Apply the same BC initialization, clean/attack mixture, seeds, evaluation budget, retention regularizer, and rollback rules to all six algorithms for fairness. AMS/Meta remains the proposed algorithmic difference rather than receiving a uniquely easier environment.
- Decouple evidence export from the hard-b success gate. Every algorithm and every attempted difficulty must produce CSV data, reward/safety plots, true top-down trajectories, and real rendered videos even when its acceptance status is failed. The final comparison should display failures explicitly rather than suppressing videos because the terminal gate was not reached.
- 自动监测 `2026-08-06T01:45:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-06_00-37-19_ackermann_bc_six_attack_20260804_040303_hard_b_r4_harl_hatrpo`，update `6600`，已完成评估 `63`，最近完成率/道路越界/碰撞 `0.0/0.234375/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T02:46:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-06_02-45-10_ackermann_bc_six_attack_20260804_040303_hard_b_r5_happo_no_meta`，update `6630`，已完成评估 `66`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.00390625`，错误数 `0`。
- 自动监测 `2026-08-06T03:47:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-06_03-16-04_ackermann_bc_six_attack_20260804_040303_hard_b_r5_happo_meta_ams`，update `7181`，已完成评估 `66`，最近完成率/道路越界/碰撞 `0.0/0.078125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T04:48:04+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-06_04-17-36_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_haa2c`，update `7200`，已完成评估 `66`，最近完成率/道路越界/碰撞 `0.0/0.125/0.00390625`，错误数 `0`。
- 自动监测 `2026-08-06T05:48:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `68`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T06:49:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T07:50:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T08:51:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T09:52:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T10:53:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T11:54:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T12:55:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T13:56:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T14:57:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T15:58:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T16:59:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T18:00:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T19:01:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T20:02:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T21:03:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T22:04:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-06T23:05:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T00:06:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T01:07:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T02:08:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T03:09:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T04:10:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T05:11:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T06:12:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T07:13:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T08:14:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T09:15:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T10:16:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T11:17:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T12:18:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T13:19:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T14:20:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T15:21:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T16:22:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T17:23:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T18:24:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T19:25:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T20:26:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T21:27:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T22:28:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-07T23:29:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T00:30:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T01:31:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T02:32:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T03:33:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T04:34:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T05:35:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T06:36:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T07:37:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T08:38:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T09:39:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T10:40:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T11:41:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T12:42:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T13:43:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T14:44:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。
- 自动监测 `2026-08-08T15:45:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-06_04-48-18_ackermann_bc_six_attack_20260804_040303_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.203125/0.0`，错误数 `0`。

## 2026-08-08 Final Matrix Result Audit

- The six-algorithm attack matrix finished on `2026-08-06 06:25:07 +0800` and exited with status `8` because HAPPO-Meta-AMS whole-platoon hard-b completion remained `0.0` in all five hard-b rounds, below the declared `0.50` gate. There are no remaining Isaac training, evaluation, playback, or matrix-launcher processes; this is a completed failed experiment, not a hang.
- All 72 algorithm-stage evaluations were completed (12 difficulty/round packages x 6 algorithms, each with two checkpoint rows). Across all 144 evaluation-summary rows, no post-training checkpoint has non-zero whole-platoon completion. In hard-b round 5, every algorithm's final completion/dwell peak is `0`; AMS minimum-vehicle mean progress is only `0.000266`, with road-boundary-any `0.0717` and collision `0`.
- Final preserved artifact counts are 300 CSV files, 36 PNG comparison figures, 144 checkpoints, and 12 stage tar archives. There are zero MP4 videos, zero final-ladder artifacts, and no final checksum manifest because the script intentionally stopped at the failed hard-b gate before the downstream renderer/top-down ladder.
- The hourly monitor timer is still active even though the matrix service has been collected and no longer exists. It has appended repeated inactive warnings since completion; the last reviewed snapshot at `2026-08-08 15:45:56 +0800` reports no processes and 72 completed evaluations. This audit did not stop the timer because the user requested status only.
- The recovery direction remains unchanged: preserve this run as failure evidence, return to the immutable BC seed, fix the raw-versus-executed action/log-prob mismatch, add BC/KL retention and clean/attack mixed short update blocks, enforce rollback gates at every difficulty, and export videos/data even for failed stages before attempting another full six-algorithm matrix.

## 2026-08-11 AMS Clean Recovery And Attack Restart

- The untouched real-trajectory BC seed passed a full deterministic clean evaluation with 16 environments and 3500 steps: instant/dwell/ever completion all `1.0`, collision `0`, road-boundary-any `0`, road-boundary termination `0`, and attack enabled `0`. The stage selector was corrected to accept both direct-root and nested `eval_summary.csv` files and now enforces instant completion, road termination, and attack-disabled gates.
- HAPPO retention was corrected so BC/KL retention is applied once per logical runner update per unique actor rather than once per PPO minibatch. A 100-update AMS recovery still completed the spatial path safely but stretched longitudinal formation and failed completion; this was diagnosed as policy drift rather than insufficient episode time.
- A conservative 40-update AMS run from the immutable BC checkpoint used actor LR `1e-6`, BC coefficient `10`, KL coefficient `1`, AMS warmup/ramp `5/15`, and AMS final weight `0.02`. With the old longitudinal shield it completed the spatial path with zero collision/road violation but failed the 2-second whole-platoon dwell because the leader/follower speed mismatch accumulated.
- The first shield-only reevaluation with `catchup_action=0.40` matched platoon/leader speed (`0.456/0.455 m/s`) and reached instantaneous completion, but only `0.0625` of environments held the criterion for 2 seconds because the actual gap was pinned near the `0.35 m` completion-error boundary.
- The final strict reevaluation changed only the longitudinal projection to `d_drop=1.50`, `catchup_action=0.40`; no expert and no attack were enabled and no completion tolerance was relaxed. Result: return `75084.505`, leader/platoon speed `0.455/0.454 m/s`, mean minimum pair gap `1.502 m`, mean absolute gap error `0.270 m`, instant/dwell/ever completion `1.0/1.0/1.0`, maximum completion streak `41.04 s`, collision `0`, road-boundary-any `0`, road termination `0`, attack enabled `0`.
- Strict evidence: `logs/ackermann_ams_clean_recovery/ams_clean_conservative_20260811/evaluation_catchup040_ddrop150/eval_summary.csv`; gate: `logs/ackermann_ams_clean_recovery/ams_clean_conservative_20260811/clean_gate.json`; promoted checkpoint list: `logs/ackermann_ams_clean_recovery/ams_clean_conservative_20260811/promoted_strict_clean_runs.txt`. The gate passed and one checkpoint was promoted.
- The validated longitudinal settings were propagated to `run_ackermann_progressive_clean.sh`, `run_ackermann_full_six_attack_matrix.sh`, and `run_platoon_spatial_attack_ladder.sh`. Shell syntax, Python compilation, and `git diff --check` passed.
- The corrected six-algorithm low-to-high attack matrix is queued as user service `ackermann-ams-six-attack-20260811.service`, output root `logs/rsl_rl/platoon_happo/ackermann_six_full_20260811_strictclean`, with 32 training environments, 16 evaluation environments, 600 updates per stage, and the full 3500-step evaluation. It waits for the already-scheduled `m5-cagan-frozen-eval-20260811-v2.service` to finish and for four consecutive GPU-idle checks before starting at `light`, preventing concurrent Isaac workloads.
- Hourly monitoring is active via `ackermann-ams-six-monitor-20260811.timer`; first scheduled check is `2026-08-11 18:13 +0800`. It appends JSON snapshots under the matrix output root and human-readable status to this note file.

## 2026-08-21 Failed-Matrix Evidence Recovery And Retained Restart

- The queued six-algorithm run completed all 72 stage evaluations (12 stage/round packages x 6 algorithms) and produced 144 checkpoints, but the service failed on `2026-08-13 19:45 +0800` during the first final-ladder plot. The exact exception was `ModuleNotFoundError: No module named 'matplotlib'` from the systemd `/usr/bin/python3`; this was an evidence-export failure, not a training or Isaac failure.
- All AMS stage summaries were audited. AMS whole-platoon completion is `0.0` from light through hard-b round 5. The saved environment configuration proves retention was not propagated into this matrix: `happo_retention_enabled=false`, empty trajectory path, BC coefficient `0`, and KL coefficient `0`. Therefore the failed lineage is preserved as catastrophic-forgetting evidence and will not be used as the parent for corrected training.
- `run_platoon_spatial_attack_ladder.sh` now uses the verified plotting interpreter `/home/cnc/Downloads/ENTER/bin/python3`, supports `RESUME_EXISTING=1`, skips only complete evaluation/video pairs, and still regenerates both plots. A local smoke test generated valid reward/safety and true top-down PNG files. The final replay shield was aligned with the actual stage/strict-clean configuration (`d_crit=0.50`, `d_drop=1.50`, `catchup_action=0.40`, default lateral/centerline gains) instead of the unrelated legacy hard shield.
- The previous MAPPO/off evidence was recoverably moved to `final_ladders/mappo/off_pre_alignment_20260821` so it is preserved while the aligned MAPPO/off result is regenerated. No training checkpoint or evaluation result was deleted.
- Evidence recovery is active as `ackermann-ams-failed-evidence-20260821.service`. It resumes from `stages/hard_b_r5/training/runs.txt`, exports every algorithm and every off-to-hard-b level, continues to later algorithms if one ladder fails, and will create CSVs, real renderer MP4s, reward/safety plots, top-down true-XY plots, matrix heatmaps, and checksums. At `10:35 +0800` it was actively evaluating MAPPO/off with 16 environments and 3500 steps; GPU usage was about `3180 MiB`, `32%`.
- The corrected matrix launcher now explicitly applies the same retention settings to all six algorithms: actor LR `1e-6`, real-trajectory BC coefficient `10`, reference-policy KL coefficient `1`, retention batch `1024`, expert execution off, and validated longitudinal shield parameters. AMS uses weight `0.02` with warmup/ramp `5/15`.
- Corrected low-to-high training is queued as `ackermann-ams-retained-six-attack-20260821.service`, output root `logs/rsl_rl/platoon_happo/ackermann_six_retained_20260821`. It waits for failed-run evidence export plus four GPU-idle checks, then starts all six algorithms from the immutable BC seed with 100-update short blocks from light through hard-b and up to 10 hard-b recovery rounds; AMS hard-b gate remains `>=0.50`.
- The obsolete failed-service hourly timer was stopped. Hourly monitoring is now `ackermann-ams-retained-six-monitor-20260821.timer`, first due around `2026-08-21 11:36 +0800`; it observes the active evidence process while the corrected service waits, then follows corrected training automatically.
- 自动监测 `2026-08-11T18:13:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-11T19:14:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-11_18-49-45_ackermann_six_full_20260811_strictclean_light_happo_no_meta`，update `508`，已完成评估 `0`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-11T20:15:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-11_19-52-45_ackermann_six_full_20260811_strictclean_light_harl_mappo_shared`，update `499`，已完成评估 `0`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-11T21:16:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-11_20-51-06_ackermann_six_full_20260811_strictclean_light_harl_hatrpo`，update `495`，已完成评估 `0`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-11T22:17:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-11_21-56-19_ackermann_six_full_20260811_strictclean_easy_0_mappo`，update `1031`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-11T23:18:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-11_22-57-15_ackermann_six_full_20260811_strictclean_easy_0_happo_meta_ams`，update `991`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T00:19:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-11_23-58-26_ackermann_six_full_20260811_strictclean_easy_0_harl_haa2c`，update `1028`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T01:20:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_00-28-43_ackermann_six_full_20260811_strictclean_easy_0_harl_hatrpo`，update `1200`，已完成评估 `7`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T02:21:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_02-05-20_ackermann_six_full_20260811_strictclean_easy_a_mappo`，update `1532`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T03:22:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_03-05-48_ackermann_six_full_20260811_strictclean_easy_a_happo_meta_ams`，update `1508`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T04:23:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_04-07-21_ackermann_six_full_20260811_strictclean_easy_a_harl_haa2c`，update `1525`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T05:24:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_04-37-57_ackermann_six_full_20260811_strictclean_easy_a_harl_hatrpo`，update `1800`，已完成评估 `13`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T06:25:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_06-14-07_ackermann_six_full_20260811_strictclean_easy_b_mappo`，update `2034`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.0/0.3125/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T07:26:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_07-14-35_ackermann_six_full_20260811_strictclean_easy_b_happo_meta_ams`，update `2017`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T08:27:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_08-15-57_ackermann_six_full_20260811_strictclean_easy_b_harl_haa2c`，update `2036`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T09:28:37+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_08-46-10_ackermann_six_full_20260811_strictclean_easy_b_harl_hatrpo`，update `2400`，已完成评估 `19`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T10:28:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_10-22-34_ackermann_six_full_20260811_strictclean_med_a_mappo`，update `2522`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.65625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T11:29:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_11-23-23_ackermann_six_full_20260811_strictclean_med_a_happo_meta_ams`，update `2514`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T12:30:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_12-24-44_ackermann_six_full_20260811_strictclean_med_a_harl_haa2c`，update `2521`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T13:31:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_12-54-41_ackermann_six_full_20260811_strictclean_med_a_harl_hatrpo`，update `3000`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T14:32:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_14-30-33_ackermann_six_full_20260811_strictclean_med_b_mappo`，update `3042`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T15:33:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_15-31-35_ackermann_six_full_20260811_strictclean_med_b_happo_meta_ams`，update `3038`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.0625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T16:34:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_16-32-54_ackermann_six_full_20260811_strictclean_med_b_harl_haa2c`，update `3037`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T17:35:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_17-02-46_ackermann_six_full_20260811_strictclean_med_b_harl_hatrpo`，update `3600`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T18:36:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_17-02-46_ackermann_six_full_20260811_strictclean_med_b_harl_hatrpo`，update `3600`，已完成评估 `35`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T19:37:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_19-09-40_ackermann_six_full_20260811_strictclean_hard_a_happo_no_meta`，update `4167`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.28125/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T20:38:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_20-13-14_ackermann_six_full_20260811_strictclean_hard_a_harl_mappo_shared`，update `4155`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T21:39:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_21-11-28_ackermann_six_full_20260811_strictclean_hard_a_harl_hatrpo`，update `4140`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T22:40:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-12_21-11-28_ackermann_six_full_20260811_strictclean_hard_a_harl_hatrpo`，update `4200`，已完成评估 `41`，最近完成率/道路越界/碰撞 `0.0/0.1875/0.0`，错误数 `0`。
- 自动监测 `2026-08-12T23:41:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-12_23-18-10_ackermann_six_full_20260811_strictclean_hard_b_r1_happo_no_meta`，update `4673`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.40625/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T00:42:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_00-21-46_ackermann_six_full_20260811_strictclean_hard_b_r1_harl_mappo_shared`，update `4657`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T01:43:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_01-19-51_ackermann_six_full_20260811_strictclean_hard_b_r1_harl_hatrpo`，update `4654`，已完成评估 `42`，最近完成率/道路越界/碰撞 `0.0/0.0625/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T02:44:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-13_01-19-51_ackermann_six_full_20260811_strictclean_hard_b_r1_harl_hatrpo`，update `4800`，已完成评估 `46`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T03:45:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_03-27-08_ackermann_six_full_20260811_strictclean_hard_b_r2_happo_no_meta`，update `5172`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.25/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T04:46:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_04-30-54_ackermann_six_full_20260811_strictclean_hard_b_r2_harl_mappo_shared`，update `5145`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T05:47:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_05-28-54_ackermann_six_full_20260811_strictclean_hard_b_r2_harl_hatrpo`，update `5160`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.1875/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T06:48:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-13_05-28-54_ackermann_six_full_20260811_strictclean_hard_b_r2_harl_hatrpo`，update `5400`，已完成评估 `52`，最近完成率/道路越界/碰撞 `0.0/0.21875/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T07:49:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_07-35-47_ackermann_six_full_20260811_strictclean_hard_b_r3_happo_no_meta`，update `5678`，已完成评估 `54`，最近完成率/道路越界/碰撞 `0.0/0.1875/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T08:50:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_08-39-45_ackermann_six_full_20260811_strictclean_hard_b_r3_harl_mappo_shared`，update `5637`，已完成评估 `54`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T09:51:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_09-38-17_ackermann_six_full_20260811_strictclean_hard_b_r3_harl_hatrpo`，update `5654`，已完成评估 `54`，最近完成率/道路越界/碰撞 `0.0/0.21875/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T10:52:39+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-13_09-38-17_ackermann_six_full_20260811_strictclean_hard_b_r3_harl_hatrpo`，update `6000`，已完成评估 `58`，最近完成率/道路越界/碰撞 `0.0/0.09375/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T11:52:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_11-44-52_ackermann_six_full_20260811_strictclean_hard_b_r4_happo_no_meta`，update `6157`，已完成评估 `60`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T12:53:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_12-48-28_ackermann_six_full_20260811_strictclean_hard_b_r4_harl_mappo_shared`，update `6113`，已完成评估 `60`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T13:54:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_13-46-20_ackermann_six_full_20260811_strictclean_hard_b_r4_harl_hatrpo`，update `6159`，已完成评估 `60`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `0`。
- 自动监测 `2026-08-13T14:55:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-13_13-46-20_ackermann_six_full_20260811_strictclean_hard_b_r4_harl_hatrpo`，update `6600`，已完成评估 `63`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T15:56:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_15-52-47_ackermann_six_full_20260811_strictclean_hard_b_r5_happo_no_meta`，update `6678`，已完成评估 `66`，最近完成率/道路越界/碰撞 `0.0/0.28125/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T16:57:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_16-57-10_ackermann_six_full_20260811_strictclean_hard_b_r5_harl_mappo_shared`，update `6611`，已完成评估 `66`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T17:58:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-13_17-54-51_ackermann_six_full_20260811_strictclean_hard_b_r5_harl_hatrpo`，update `6673`，已完成评估 `66`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T18:59:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-13_17-54-51_ackermann_six_full_20260811_strictclean_hard_b_r5_harl_hatrpo`，update `7200`，已完成评估 `69`，最近完成率/道路越界/碰撞 `0.0/0.21875/0.0`，错误数 `0`。
- 自动监测 `2026-08-13T20:00:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-13T21:01:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-13T22:02:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-13T23:03:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T00:04:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T01:05:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T02:06:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T03:07:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T04:08:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T05:09:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T06:10:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T07:11:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T08:12:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T09:13:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T10:14:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T11:15:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T12:16:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T13:17:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T14:18:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T15:19:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T16:20:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T17:21:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T18:22:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T19:23:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T20:24:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T21:25:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T22:26:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-14T23:27:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T00:28:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T01:29:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T02:30:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T03:31:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T04:32:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T05:33:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T06:34:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T07:35:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T08:36:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T09:37:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T10:38:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T11:39:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T12:40:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T13:41:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T14:42:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T15:43:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T16:44:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T17:45:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T18:46:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T19:47:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T20:48:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T21:49:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T22:50:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-15T23:51:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T00:52:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T01:53:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T02:54:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T03:55:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T04:56:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T05:57:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T06:58:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T07:59:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T09:00:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T10:01:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T11:02:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T12:03:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T13:04:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T14:05:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T15:06:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T16:07:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T17:08:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T18:09:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T19:10:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T20:11:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T21:12:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T22:13:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-16T23:14:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T00:15:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T01:16:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T02:17:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T03:18:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T04:19:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T05:20:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T06:21:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T07:22:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T08:23:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T09:24:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T10:25:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T11:26:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T12:27:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T13:28:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T14:29:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T15:30:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T16:31:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T17:32:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T18:33:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T19:34:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T20:35:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T21:36:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T22:37:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-17T23:38:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T00:39:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T01:40:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T02:41:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T03:42:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T04:43:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T05:44:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T06:45:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T07:46:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T08:47:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T09:48:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T10:49:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T11:50:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T12:51:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T13:52:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T14:53:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T15:54:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T16:55:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T17:56:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T18:57:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T19:58:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T20:59:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T22:00:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-18T23:01:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T00:02:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T01:03:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T02:04:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T03:05:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T04:06:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T05:07:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T06:08:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T07:09:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T08:10:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T09:11:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T10:12:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T11:13:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T12:14:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T13:15:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T14:16:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T15:17:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T16:18:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T17:19:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T18:20:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T19:21:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T20:22:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T21:23:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T22:24:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-19T23:25:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T00:26:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T01:27:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T02:28:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T03:29:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T04:30:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T05:31:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T06:32:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T07:33:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T08:34:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T09:35:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T10:36:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T11:37:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T12:38:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T13:39:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T14:40:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T15:41:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T16:42:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T17:43:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T18:44:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T19:45:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T20:46:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T21:47:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T22:48:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-20T23:49:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T00:50:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T01:51:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T02:52:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T03:53:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T04:54:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T05:55:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T06:56:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T07:57:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T08:58:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T09:59:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-13_15-22-27_ackermann_six_full_20260811_strictclean_hard_b_r5_mappo`，update `7200`，已完成评估 `72`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0`，错误数 `1`。
- 自动监测 `2026-08-21T11:36:09+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T12:36:09+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T13:36:10+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T14:36:11+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T15:36:12+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T16:36:12+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T17:36:13+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T18:36:14+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T19:36:15+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T20:36:16+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T21:36:16+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T22:36:17+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-08-21T23:36:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-21_23-17-18_ackermann_six_retained_20260821_light_harl_hatrpo`，update `100`，已完成评估 `2`，最近完成率/道路越界/碰撞 `0.0/0.15625/0.0234375`，错误数 `0`。
- 自动监测 `2026-08-22T00:37:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_00-25-01_ackermann_six_retained_20260821_easy_0_harl_hatrpo`，update `200`，已完成评估 `6`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.1171875`，错误数 `0`。
- 自动监测 `2026-08-22T01:38:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-22_01-35-59_ackermann_six_retained_20260821_easy_a_mappo`，update `254`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-22T02:39:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_02-03-46_ackermann_six_retained_20260821_easy_a_harl_hatrpo`，update `300`，已完成评估 `14`，最近完成率/道路越界/碰撞 `0.0/0.3125/0.0546875`，错误数 `0`。
- 自动监测 `2026-08-22T03:40:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-22_03-37-10_ackermann_six_retained_20260821_easy_b_harl_haa2c`，update `371`，已完成评估 `18`，最近完成率/道路越界/碰撞 `0.6875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-22T04:41:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_03-42-40_ackermann_six_retained_20260821_easy_b_harl_hatrpo`，update `400`，已完成评估 `22`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0625`，错误数 `0`。
- 自动监测 `2026-08-22T05:42:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_05-21-59_ackermann_six_retained_20260821_med_a_harl_hatrpo`，update `500`，已完成评估 `25`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `0`。
- 自动监测 `2026-08-22T06:43:31+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-22_06-38-09_ackermann_six_retained_20260821_med_b_happo_no_meta`，update `600`，已完成评估 `30`，最近完成率/道路越界/碰撞 `0.78125/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-22T07:43:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_07-00-31_ackermann_six_retained_20260821_med_b_harl_hatrpo`，update `600`，已完成评估 `33`，最近完成率/道路越界/碰撞 `0.0/0.28125/0.0234375`，错误数 `0`。
- 自动监测 `2026-08-22T08:44:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-08-22_08-39-34_ackermann_six_retained_20260821_hard_a_harl_hatrpo`，update `696`，已完成评估 `36`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-22T09:45:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_08-39-34_ackermann_six_retained_20260821_hard_a_harl_hatrpo`，update `700`，已完成评估 `41`，最近完成率/道路越界/碰撞 `0.0/0.03125/0.0`，错误数 `0`。
- 自动监测 `2026-08-22T10:46:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `44`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `0`。
- 自动监测 `2026-08-22T11:47:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_09-51-24_ackermann_six_retained_20260821_hard_b_r1_mappo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T12:48:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_09-51-24_ackermann_six_retained_20260821_hard_b_r1_mappo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T13:49:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_09-57-05_ackermann_six_retained_20260821_hard_b_r1_happo_no_meta`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T14:50:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_09-57-05_ackermann_six_retained_20260821_hard_b_r1_happo_no_meta`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T15:51:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_10-02-43_ackermann_six_retained_20260821_hard_b_r1_happo_meta_ams`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T16:52:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_10-02-43_ackermann_six_retained_20260821_hard_b_r1_happo_meta_ams`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T17:53:01+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_10-08-50_ackermann_six_retained_20260821_hard_b_r1_harl_mappo_shared`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T18:53:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_10-08-50_ackermann_six_retained_20260821_hard_b_r1_harl_mappo_shared`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T19:54:31+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_10-13-59_ackermann_six_retained_20260821_hard_b_r1_harl_haa2c`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T20:54:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_10-13-59_ackermann_six_retained_20260821_hard_b_r1_harl_haa2c`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-22T21:55:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-22T22:56:56+08:00` [警告]：六算法攻击矩阵服务 `active`，模式 `playback`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-22T23:57:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T00:58:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T01:59:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T03:00:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T04:01:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T05:02:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T06:03:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T07:04:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T08:05:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T09:06:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T10:07:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T11:08:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T12:09:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T13:10:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T14:11:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T15:12:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T16:13:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T17:14:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T18:15:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T19:16:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T20:17:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T21:18:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T22:19:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-23T23:20:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T00:21:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T01:22:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T02:23:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T03:24:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T04:25:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T05:26:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T06:27:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T07:28:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T08:29:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T09:30:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。
- 自动监测 `2026-08-24T10:31:56+08:00` [警告]：六算法攻击矩阵服务 `inactive`，模式 `none`，当前运行 `2026-08-22_10-19-30_ackermann_six_retained_20260821_hard_b_r1_harl_hatrpo`，update `800`，已完成评估 `48`，最近完成率/道路越界/碰撞 `0.0/0.125/0.0078125`，错误数 `1`。

## 2026-08-24 Retained Matrix Final Status Audit

- Both long-running services have finished and no Isaac training/evaluation/playback process remains. The old failed-lineage evidence recovery completed on `2026-08-21 22:47 +0800`. The corrected retained matrix completed on `2026-08-22 23:39 +0800`.
- Evidence is complete. The old matrix contains all six algorithms x nine attack levels plus its preserved pre-alignment MAPPO/off run: 448 CSVs, 110 PNGs, 55 MP4s, and 144 checkpoints. The corrected matrix contains 349 CSVs, 110 PNGs, 54 MP4s, and 96 checkpoints. Every corrected algorithm/level has a non-empty evaluation summary, reward/safety plot, true top-down XY plot, and real renderer MP4; matrix CSV, completion/return heatmaps, and SHA-256 manifests also exist.
- Retention was genuinely active in the corrected run: actor LR `1e-6`, BC coefficient `10`, reference KL `1`, batch `1024`, `d_drop=1.5`, and `catchup_action=0.4`. This materially fixed forgetting. Independent final-ladder completion for MAPPO from off through hard-b is `1,1,1,1,0.9375,0.75,0.8125,0.1875,0.0625`; HAPPO-no-Meta is `1,1,1,1,1,1,0.875,0.25,0`; AMS is `1,1,1,1,0.3125,0.375,0.375,0,0`; shared MAPPO is `1,1,1,1,1,0.875,0.75,0.25,0`; HAA2C is `1,1,1,1,1,0.9375,0.875,0.5,0`; HATRPO is `0` throughout and exhibits non-zero collision/road violations.
- The training-stage hard-b gate reported AMS completion `1.0`, but that result is invalid as an independent robustness claim. The stage evaluator used `fresh_env_per_checkpoint=0` and evaluated `model_700.pt` first (`completion=0`) followed by `model_final.pt` (`completion=1`) in the same environment, so the second checkpoint inherited an advanced attack RNG/environment sequence. The independent final ladder evaluated the identical `model_final.pt` first with the same nominal seed/config and obtained completion `0`, dwell `0`, instant peak `0.125`, leader/platoon speed about `0.316/0.325 m/s`, and minimum-progress mean `0.576`. This closely matches the stage's first-checkpoint behavior and confirms evaluation-order leakage.
- The reliable AMS result is therefore: strict clean/off remains `1.0` with zero collision/road violation, low attacks through easy-a remain `1.0`, performance drops to `0.3125-0.375` at easy-b/medium, and independent hard-a/hard-b completion is `0`. The user's required independently reproducible AMS hard-b result of at least `0.50` has not yet been achieved.
- The hourly monitor timer remains active but now reports the collected transient service as inactive; no experiment is currently progressing. Any next recovery must enforce fresh environment per checkpoint and multi-seed hard-b validation before promotion, then resume from the last independently passing AMS checkpoint rather than treating the leaked hard-b gate as success.

## 2026-08-24 AMS Isolated Multi-Seed Recovery Started

- The obsolete retained-matrix monitor was stopped. Recovery starts from the independently passing easy-a AMS checkpoint `2026-08-22_01-47-02_ackermann_six_retained_20260821_easy_a_happo_meta_ams/model_final.pt`; the leaked hard-b checkpoint is not used as the parent.
- Added `scripts/tools/run_ackermann_ams_multiseed_recovery.sh`. Each candidate is trained for a 40-update short block with 50% clean environments, attack strength ramped from 25% to 100%, actor LR `1e-6`, BC `10`, reference KL `1`, retention batch `1024`, AMS weight `0.02`, expert execution disabled, and the validated Ackermann spatial-path/longitudinal parameters.
- Promotion is isolated and order-independent: one checkpoint per Isaac process, `--fresh_env_per_checkpoint`, one strict clean evaluation, and three attacked evaluations pairing environment seeds `42/3407/777` with attacker seeds `3407/9917/17011`. Clean requires instant/dwell/ever `1.0` and zero collision/road violation/road termination. Attack promotion thresholds are easy-b `0.75`, med-a `0.65`, med-b `0.55`, hard-a `0.50`, and hard-b `0.50`; hard-b uses the minimum completion across all three seeds, not the mean. Each phase permits at most three independent retries from the last promoted parent.
- Recovery service `ackermann-ams-multiseed-recovery-20260824.service` started at `2026-08-24 11:21:46 +0800`; output root is `logs/rsl_rl/platoon_happo/ams_multiseed_recovery_20260824`. It is actively training easy-b round 1 with 32 environments and no concurrent Isaac workload.
- Hourly monitoring is active as `ackermann-ams-multiseed-monitor-20260824.timer`; first scheduled snapshot is around `12:21 +0800` and will append to this file. If multi-seed hard-b passes, the pipeline automatically generates the full off-to-hard-b CSV, reward/safety plots, true top-down trajectories, real renderer videos, and SHA-256 manifest for the promoted AMS checkpoint.
- 自动监测 `2026-08-24T12:21:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T13:22:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T14:23:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T15:24:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T16:25:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T17:26:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T18:27:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T19:28:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T20:29:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T21:30:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T22:31:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-24T23:32:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T00:33:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T01:34:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T02:35:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T03:36:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T04:37:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T05:38:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T06:39:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T07:40:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T08:41:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T09:42:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T10:43:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T11:44:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T12:45:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T13:46:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T14:47:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T15:48:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T16:49:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T17:50:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T18:51:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T19:52:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T20:53:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T21:54:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T22:55:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-25T23:56:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T00:57:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T01:58:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T02:59:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T04:00:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T05:01:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T06:02:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T07:03:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T08:04:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T09:05:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T10:06:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T11:07:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T12:08:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T13:09:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T14:10:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T15:11:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T16:12:11+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T17:12:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T18:13:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T19:14:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T20:15:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T21:16:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T22:17:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-26T23:18:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T00:19:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T01:20:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T02:21:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T03:22:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T04:22:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T05:22:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T06:22:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T07:22:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T08:22:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T09:22:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T10:22:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T11:22:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。

## 2026-08-27 AMS 多种子恢复任务进度核查

- 当前没有 Isaac Lab 训练、评估或录像进程在运行；`ackermann-ams-multiseed-recovery-20260824.service` 已于 `2026-08-24 11:30:03` 失败退出。仍在运行的每小时监控 timer 只是在重复报告同一失败状态，并非实验仍有进展。
- 恢复任务只完成 `easy_b/r1`：从 easy-a 父模型追加训练 40 updates（301→340），随后完成一次框架内置的 fresh-env easy-b 评估。
- 该单次初步评估结果：整队变道 ever-completion `1.0`，持续完成率峰值 `0.6875`，碰撞率 `0`，道路边界违规率 `0`，平均最小车距 `1.447 m`，leader/platoon 跟踪误差约 `0.391/0.392`。结果看起来安全且有完整变道能力，但不能替代计划中的严格 clean 门禁和三个独立攻击种子验证。
- 失败原因不是训练或仿真崩溃，而是评估结束后的绘图脚本 `scripts/tools/plot_medium_algorithm_comparison.py` 使用了缺少 `matplotlib` 的 Python 解释器，触发 `ModuleNotFoundError`，使外层恢复脚本在进入自定义严格验证之前退出。
- 因此尚未生成 `validation_gate.json`，没有任何 stage promotion，也未进入 medium/hard（包括 hard-b）；本轮恢复目录目前为 3 个 CSV、2 个 PT、0 个 JSON、0 个 PNG、0 个 MP4。
- 后续应先修正绘图解释器，并从已完成的 easy-b 候选检查点继续严格 clean + 三攻击种子验证，避免重复训练；通过门禁后再逐级推进到 hard-b 及六算法全强度数据、图片和视频实验。

## 2026-08-27 18:53 AMS 恢复修复并续跑验证

- 已修复 `run_medium_algorithm_comparison.sh`：新增并实际使用 `PLOT_PYTHON=/home/cnc/Downloads/ENTER/bin/python3`，绘图前显式检查 `matplotlib`；该解释器已验证可导入 matplotlib 3.11.0，不再依赖 service 环境中的模糊 `python3`。
- 已增强 `run_ackermann_ams_multiseed_recovery.sh` 的断点续跑：若 round 已有有效 `model_final.pt` 则跳过重复训练；若某个 clean/攻击种子已有非空 `eval_summary.csv` 则复用该验证结果，从未完成的验证继续。
- 两个脚本均已通过 `bash -n` 和 `git diff --check`。
- 已于 `2026-08-27 18:53:46 +08:00` 重启 `ackermann-ams-multiseed-recovery-20260824.service`。服务确认复用 easy-b/r1 update 340 检查点，并已进入严格 clean 验证：16 environments、3500 steps、fresh environment、攻击关闭、expert blend 关闭。
- 启动后服务状态为 `active (running)`，Isaac eval PID `967439`，GPU 显存约 `2862 MiB`；当前仅有 URDF 合并惯量的弃用警告，没有新的错误。clean 通过后脚本会自动执行三个独立环境/攻击种子门禁，再根据结果 promotion 或进入下一轮恢复训练。
- 自动监测 `2026-08-27T12:22:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T13:22:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T14:22:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T15:22:25+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T16:22:25+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T17:22:26+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T18:22:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-24_11-22-02_ams_multiseed_recovery_20260824_easy_b_r1_happo_meta_ams`，update `340`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `1`。
- 自动监测 `2026-08-27T19:22:27+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_19-16-08_ams_multiseed_recovery_20260824_med_a_r1_happo_meta_ams`，update `380`，已完成评估 `1`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-27T20:22:28+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-27T21:22:29+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-27T22:22:29+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-27T23:22:30+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T00:22:31+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T01:22:31+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T02:22:32+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T03:22:33+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T04:22:33+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T05:22:34+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T06:22:34+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T07:22:35+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T08:22:36+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T09:22:36+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T10:22:37+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T11:22:38+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T12:22:38+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T13:22:39+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T14:22:40+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T15:22:40+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T16:22:41+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T17:22:42+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T18:22:42+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T19:22:43+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T20:22:44+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T21:22:44+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T22:22:45+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-28T23:22:46+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T00:22:46+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T01:22:47+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T02:22:48+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T03:22:49+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T04:22:49+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T05:22:50+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T06:22:51+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T07:22:51+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T08:22:52+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T09:22:53+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T10:22:53+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T11:22:54+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T12:22:55+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T13:22:55+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T14:22:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T15:22:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T16:22:58+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T17:22:58+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T18:22:59+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T19:23:00+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T20:23:00+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T21:23:01+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T22:23:02+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-29T23:23:02+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T00:23:03+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T01:23:04+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T02:23:04+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T03:23:05+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T04:23:05+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T05:23:06+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T06:23:07+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T07:23:07+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T08:23:08+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T09:23:09+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T10:23:09+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T11:23:10+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T12:23:11+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T13:23:11+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T14:23:12+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T15:23:13+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T16:23:13+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T17:23:14+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-27_20-17-21_ams_multiseed_recovery_20260824_med_b_r1_happo_meta_ams`，update `420`，已完成评估 `3`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。

## 2026-08-30 17:52 AMS 卡死修复并继续

- 核查发现服务表面保持 active，但 `med_b/r1` 内置评估实际自 `2026-08-27 20:25:18` 后无日志更新：`model_400.pt` 的 3500 步评估已结束，随后在为 `model_final.pt` 重建第二个 fresh Isaac 环境时卡死约 69.5 小时；Python 进程持续占用约 117% CPU 和 2862 MiB GPU。此前每小时监控把“进程存在”误判为正常进展。
- 恢复任务已取得的有效门禁：`easy_b/r1` 通过，三攻击种子 completion 最低 `0.9375`、平均 `0.9792`，clean completion/instant/dwell 均为 `1.0`，碰撞和道路越界均为 `0`；`med_a/r1` 未过 0.65 门槛，`med_a/r2` 通过，三攻击种子 completion 均为 `0.5625`，clean 严格指标均为 `1.0`，碰撞和道路越界均为 `0`。
- 已修改恢复脚本，对后续新训练 round 设置 `SKIP_EVAL=1`，跳过与严格门禁重复且可能在多 checkpoint fresh-env 重建时卡死的内置评估；训练完成后仍保留独立 clean + 三攻击种子验证，因此不会降低门禁要求。
- 脚本已通过 `bash -n` 与 `git diff --check`。已于 `2026-08-30 17:52:11 +08:00` 重启服务，旧卡死 cgroup 被正常停止，累计浪费 CPU 时间约 3 天 11 小时 51 分。
- 重启后已复用 easy-b、med-a 的训练和验证结果，并复用 med-b/r1 update 420 `model_final.pt`，当前正在执行 med-b/r1 严格 clean fresh-env 验证（16 environments、3500 steps、攻击关闭）；通过后自动继续三个独立攻击种子。
- 已同步修复每小时监控：评估统计改为包含内置及严格验证的全部 `eval_summary.csv`，新增最近日志/训练指标/评估产物的进度年龄与 `stale` 字段；服务有进程但超过 90 分钟无新产物时将标为“警告/进度停滞”，不再仅凭 PID 存在判定正常。监控脚本已通过 AST 语法及差异检查。
- 自动监测 `2026-08-30T18:23:15+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-30_18-14-49_ams_multiseed_recovery_20260824_med_b_r2_happo_meta_ams`，update `420`，已完成评估 `20`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T19:23:15+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-08-30_19-04-52_ams_multiseed_recovery_20260824_hard_a_r1_happo_meta_ams`，update `460`，已完成评估 `29`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T20:23:16+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T21:23:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T22:23:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-30T23:23:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T00:23:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T01:23:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T02:23:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T03:23:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T04:23:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T05:23:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T06:23:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T07:23:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T08:23:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T09:23:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T10:23:25+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T11:23:26+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T12:23:26+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T13:23:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T14:23:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T15:23:28+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T16:23:29+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T17:23:29+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T18:23:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T19:23:31+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T20:23:31+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T21:23:32+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T22:23:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-08-31T23:23:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T00:23:34+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T01:23:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T02:23:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T03:23:36+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T04:23:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T05:23:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T06:23:38+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T07:23:39+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T08:23:39+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T09:23:40+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T10:23:41+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T11:23:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T12:23:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T13:23:43+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T14:23:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T15:23:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T16:23:45+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T17:23:46+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T18:23:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T19:23:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T20:23:48+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T21:23:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T22:23:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-01T23:23:50+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T00:23:51+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T01:23:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T02:23:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T03:23:53+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T04:23:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T05:23:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T06:23:55+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T07:23:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T08:23:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T09:23:57+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T10:23:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T11:23:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T12:23:59+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T13:24:00+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T14:24:00+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T15:24:01+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T16:24:01+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T17:24:02+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T18:24:03+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T19:24:03+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T20:24:04+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T21:24:04+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T22:24:05+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-02T23:24:06+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T00:24:06+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T01:24:07+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T02:24:08+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T03:24:08+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T04:24:09+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T05:24:09+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T06:24:10+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T07:24:11+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T08:24:11+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T09:24:12+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T10:24:13+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T11:24:13+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T12:24:14+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T13:24:15+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T14:24:15+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T15:24:16+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T16:24:16+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T17:24:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T18:24:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T19:24:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T20:24:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T21:24:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T22:24:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-03T23:24:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T00:24:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T01:24:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T02:24:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T03:24:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T04:24:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T05:24:25+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T06:24:26+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T07:24:26+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T08:24:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T09:24:28+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T10:24:28+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T11:24:29+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T12:24:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T13:24:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T14:24:31+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T15:24:32+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T16:24:32+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T17:24:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T18:24:34+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T19:24:34+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T20:24:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T21:24:36+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T22:24:36+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-04T23:24:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T00:24:38+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T01:24:38+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T02:24:39+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T03:24:40+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T04:24:40+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T05:24:41+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T06:24:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T07:24:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T08:24:43+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T09:24:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T10:24:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T11:24:45+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T12:24:46+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T13:24:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T14:24:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T15:24:48+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T16:24:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T17:24:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T18:24:50+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T19:24:51+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T20:24:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T21:24:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T22:24:53+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-05T23:24:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T00:24:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T01:24:55+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T02:24:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T03:24:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T04:24:57+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T05:24:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T06:24:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T07:24:59+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T08:25:00+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T09:25:00+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T10:25:01+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T11:25:01+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T12:25:02+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T13:25:03+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T14:25:04+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T15:25:04+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T16:25:05+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T17:25:06+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T18:25:06+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T19:25:07+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T20:25:08+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T21:25:08+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-06T22:25:09+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-08-30_19-54-51_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。

## 2026-09-06 AMS 当前进度与有效性审计

- 恢复服务已于 `2026-08-30 20:19:37 +08:00` 以状态码 8 正常触发门禁失败退出，当前无训练/评估/录像进程；不是再次卡死。每小时 timer 仍 active，只是在持续报告 failed。
- `easy_b/r1` 门禁通过：严格 clean completion/instant/dwell 均为 `1.0`；三攻击种子 completion 最低 `0.9375`、平均 `0.9792`；碰撞和道路越界均为 `0`。
- `med_b` 最终在 r3 通过：三种子 completion `0.6875/0.6875/0.5625`，最低 `0.5625` 高于门槛 `0.55`；严格 clean 全完成，碰撞和道路越界均为 `0`。
- 名义 hard-a 三轮结果：r1 最低/平均 completion `0.125/0.2083`；r2 `0.375/0.4375`；r3 `0.375/0.4792`；所有 clean 均严格通过且碰撞、越界均为 `0`。服务因没有候选达到门槛而退出，未进入 hard-b，也未执行最终攻击梯度、六算法全强度和录像流水线。
- 重要有效性问题：`validate_round()` 内部三种子循环复用了未声明为 local 的 `idx`，污染了外层阶段索引。证据显示 hard-a/r1 manifest 为真实 hard 参数（level hard，FDI acc `0.75`，DoS `0.12`），但 hard-a/r2、r3 实际退回 medium 参数（FDI acc `0.50`，DoS `0.10`）；因此只有 r1 是有效 hard-a 结果，r2/r3 不能视作 hard-a 重试。相同问题使 med-a/r2 使用了下一档索引/`0.55` 门槛，阶段标签与实际配置不完全一致。
- 当前恢复目录有 `90 CSV / 12 JSON / 18 PT / 6 PNG / 0 MP4`；6 张图仅来自早期 med-a 内置评估。没有 hard-b 门禁文件，也没有本轮最终真实变道视频。

## 2026-09-06 23:07 修复索引污染并重启真实 hard-a

- 已修复 `run_ackermann_ams_multiseed_recovery.sh`：`validate_round()` 的三种子循环改用函数局部变量 `seed_idx`，不再覆盖外层阶段 `idx`；新增带范围校验的 `START_PHASE_INDEX`，支持从指定阶段和指定已通过 runs 文件干净续跑。
- 修复已通过 `bash -n` 与 `git diff --check`。
- 原 `hard_a_r1/r2/r3` 的目录、日志和 tar 包没有删除，已完整移动到 `ams_multiseed_recovery_20260824/invalid_index_bug_20260830/stages/`，释放正式 `stages/hard_a_r*` 名称以生成不混用旧结果的新证据。
- 已以 `START_PHASE_INDEX=3`、父模型为已通过三种子门禁的 `med_b_r3/training/runs.txt` 重启原恢复 service。服务于 `2026-09-06 23:07:33 +08:00` 进入 active/running。
- 新 hard-a/r1 已确认正在训练，seed `42`、32 environments、40 updates；真实配置为 attack level `hard`、FDI position `2.75`、FDI acceleration `0.75`、DoS `0.12`，训练阶段 50% clean env、攻击强度从 0.25 渐进到 1.0，模型起点为 med-b/r3。训练 PID `1186344`，GPU 显存约 `2788 MiB`。
- 训练完成后自动执行严格 clean + 三个独立攻击种子验证；只有最低 completion 达到 hard-a 门槛 `0.50` 且安全约束通过才晋级 hard-b。
- 自动监测 `2026-09-06T23:25:10+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-09-06_23-07-49_ams_multiseed_recovery_20260824_hard_a_r1_happo_meta_ams`，update `460`，已完成评估 `29`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T00:25:10+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T01:25:11+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T02:25:12+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T03:25:12+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T04:25:13+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T05:25:13+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T06:25:14+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T07:25:15+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T08:25:15+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T09:25:16+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T10:25:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T11:25:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T12:25:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。

## 2026-09-07 真实 hard-a 重跑结果

- 修复索引污染后的 hard-a 三轮已全部完成；三个 manifest 均确认使用真实 hard 参数：level `hard`、FDI position `2.75`、FDI acceleration `0.75`、DoS `0.12`，训练种子依次为 `42/3407/777`。因此本轮门禁结果有效。
- hard-a/r1：三攻击种子 completion `0.25/0.25/0.125`，最低 `0.125`、平均 `0.2083`；未达到门槛 `0.50`。
- hard-a/r2：三攻击种子 completion `0.1875/0.0625/0.0625`，最低 `0.0625`、平均 `0.1042`；未达到门槛。
- hard-a/r3：三攻击种子 completion `0.1875/0.25/0.125`，最低 `0.125`、平均 `0.1875`；未达到门槛。
- 三轮严格 clean 的 completion/instant/dwell 均为 `1.0`；所有 clean 和攻击验证的碰撞、道路越界、道路越界终止均为 `0`。说明基础变道与安全性保持，但当前 AMS 在 hard-a 攻击下完成率明显不足，主要瓶颈是攻击鲁棒性而非 clean 能力或安全约束。
- 服务于 `2026-09-07 00:23:00 +08:00` 按规则以状态码 8 退出，当前无训练/评估/录像进程；没有晋级 hard-b，也未运行最终攻击梯度、六算法全强度和视频生成。
- 当前有效恢复目录仍为 `90 CSV / 12 JSON / 18 PT / 6 PNG / 0 MP4`；错误索引的旧 hard-a 证据位于单独归档目录，不计入上述有效产物。
- 自动监测 `2026-09-07T13:25:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-06_23-58-10_ams_multiseed_recovery_20260824_hard_a_r3_happo_meta_ams`，update `460`，已完成评估 `39`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。

## 2026-09-07 hard-a 下一步方案

- 训练时长是首要问题：当前每轮 40 updates、每 update 32 simulation steps、控制周期 0.02 s，总计每环境仅 `25.6 s`，短于 `70 s` 完整回合。med-b/r3 与三个 hard-a 训练块的 `platoon_completion_ever_rate` 和 `reward_true_success` 全程均为 `0`，表明训练阶段从未看到完整变道成功信号；继续重复 40-update 随机种子没有意义。
- hard-a 失败主要表现为队尾落后而非出界/碰撞。代表性 hard-a 验证中，第1至第5车平均路径进度约 `0.830/0.773/0.715/0.658/0.600`，第5车是瓶颈；平均 gap error 约 `0.321 m`，接近完成门禁 `0.35 m`。安全盾 trigger rate 约 `0.995`，action scale mean 约 `0.274`，说明安全盾几乎全程强干预，可能限制尾车追赶动作，需要诊断但不应直接解除安全约束。
- 推荐先对 med-b/r3 父模型做 hard-a 基线与攻击分量筛查（完整 hard、position-only、acceleration-only、DoS-only），确认主导退化的攻击分量；先用一个固定种子筛查，再对最差分量扩展三种子。
- 推荐将 medium→hard-a 拆成累计课程，而不是一步从 `2.00/0.50/0.10` 跳到 `2.75/0.75/0.12`：bridge-1 `2.25/0.58/0.105`，bridge-2 `2.50/0.66/0.11`，bridge-3 `2.65/0.71/0.115`，最后 hard-a `2.75/0.75/0.12`。
- 每个课程块建议至少 `160–240 updates`；优先用 240 updates，使每环境覆盖约 `153.6 s`、超过两个完整 70 s 回合。攻击 ramp 延长到约 120 updates，并在后半段保留足够的全强度训练；clean env fraction 可由 `0.50` 降至 `0.30–0.35`，同时保留轨迹 retention BC/KL 与每阶段严格 clean 门禁。
- 阶段内失败候选不应一律回退 med-b：若严格 clean 仍为 1、碰撞/越界为 0 且三种子最低完成率没有退化，则把该候选作为下一训练块父模型；始终保存并回滚到历史最优最低完成率模型。选择依据应以三种子 `minimum completion` 为主，不以平均值代替。
- 若延长训练和细化课程后尾车仍停滞，再针对 `minimum path progress`、最差 gap/尾车追赶增加密集奖励，并对 shield 的高触发/强缩放做受控消融；在此之前不建议直接修改车道、安全边界或降低 hard-a 门槛。
- hard-a 严格三种子最低完成率达到 `0.50` 且 clean/safety 门禁通过后，才用相同方式细分 hard-a→hard-b；hard-b 通过后再启动六算法全攻击强度、图片和真实视频流水线。

## 2026-09-07 14:18 启动累计 hard→hard-b 课程

- 已扩展 `run_ackermann_ams_multiseed_recovery.sh`，新增 `hard_to_hardb` 课程配置、攻击分量筛查、可配置训练时长/ramp/clean fraction、阶段内安全最佳候选累计续训，并继续使用函数局部 seed 索引防止阶段污染。
- 课程顺序与攻击预算（position/acceleration/DoS）：hard-bridge-1 `2.25/0.58/0.105`，bridge-2 `2.50/0.66/0.110`，bridge-3 `2.65/0.71/0.115`，hard-a `2.75/0.75/0.120`，hard-b-bridge-1 `3.05/0.88/0.135`，bridge-2 `3.35/1.00/0.150`，bridge-3 `3.70/1.15/0.165`，hard-b `4.00/1.30/0.180`。
- 每个训练块为 `240 updates`（每环境约 153.6 s、超过两个70 s回合），攻击 ramp 为 `120 updates`，clean environment fraction 为 `0.35`，每阶段最多2个累计训练块。失败候选只有在严格 clean 与安全约束通过且三种子最低完成率不低于阶段历史最佳时，才成为下一块父模型。
- 正式训练前自动对 med-b/r3 父模型执行同一固定种子的 `full_hard / position_only / acceleration_only / dos_only` 四组 fresh-env 3500步诊断，结果写入独立 `component_screen.json`。
- 新输出目录：`logs/rsl_rl/platoon_happo/ams_hard_curriculum_20260907`。脚本通过 `bash -n`、`git diff --check` 和绘图环境检查。
- 已停止旧恢复目录的重复告警 timer，启动 `ackermann-ams-hard-curriculum-20260907.service` 和对应每小时监控 timer。主服务于 `2026-09-07 14:18:51 +08:00` 为 active/running，已进入第一项 `component_screen/full_hard`；Isaac eval PID `1208106`，GPU 显存约 `2788 MiB`，当前无新错误。
- hard-b 门禁通过后，脚本将自动生成最终攻击梯度的数据、图片与视频；六算法全强度对比仍须以该合格 hard-b AMS 检查点为前置条件。
- 自动监测 `2026-09-07T14:19:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-09-07T15:20:20+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-09-07_15-18-15_ams_hard_curriculum_20260907_hard_bridge_1_r2_happo_meta_ams`，update `692`，已完成评估 `4`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T16:20:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-07_15-18-15_ams_hard_curriculum_20260907_hard_bridge_1_r2_happo_meta_ams`，update `900`，已完成评估 `8`，最近完成率/道路越界/碰撞 `0.5625/0.0/0.0`，错误数 `0`。

## 2026-09-07 hard课程首轮结果

- 攻击分量诊断完成。med-b/r3 父模型在固定 seed 下：position-only completion `1.0`，DoS-only `1.0`，acceleration-only `0.1875`，full-hard `0.25`；四组碰撞/道路越界均为 `0`。结论是加速度/动作 FDI 是主要性能瓶颈，位置 FDI 与当前 DoS 单独作用不是主要问题。
- hard-bridge-1（预算 `2.25/0.58/0.105`）r1 完成240 updates：严格 clean 全指标 `1.0`；三攻击种子 completion `0.50/0.50/0.4375`，最低 `0.4375`、平均 `0.4792`，低于晋级门槛 `0.55`；碰撞/越界均为 `0`。
- r1 安全且 clean 不退化，已按新逻辑作为父模型累计训练 r2 的240 updates。r2 严格 clean 仍为 `1.0`；三种子 completion `0.8125/0.4375/0.4375`，最低仍为 `0.4375`、平均提高到 `0.5625`；碰撞/越界仍为 `0`。平均性能和一个种子明显改善，但跨种子最差性能未改善，因此门禁仍失败。
- 延长训练已使学习信号恢复：r1训练期间 completion 平均/最大约 `0.414/0.656`、成功奖励非零；r2 completion 平均/最大约 `0.418/0.625`，最后一个 rollout completion `0.5625`。这验证了240 updates方向正确，但仅靠当前奖励/盾牌配置不足以提升最差种子鲁棒性。
- 服务于 `2026-09-07 15:54:05 +08:00` 按规则以状态码8退出，当前无训练/评估进程；未进入 bridge-2、hard-a 或 hard-b。
- 新课程目录当前有 `26 CSV / 3 JSON / 4 PT / 0 PNG / 0 MP4`。图片和视频仍需在 hard-b 门禁通过后生成。

## 2026-09-07 当前退出/停止规则说明

- 当前 hard_to_hardb 课程每个阶段最多 `MAX_ROUNDS_PER_PHASE=2` 个训练块；每块为240 updates，随后执行1组严格 clean和3组独立攻击种子评估。
- clean 门禁要求同时满足：ever completion `>=1.0`、instant peak `>=1.0`、dwell peak `>=1.0`、collision `<=0`、road any `<=0`、road termination `<=0`，且确认 attack disabled。
- 攻击门禁要求：恰好3个独立攻击结果；三者中的最低 ever completion 达到该阶段阈值；三者最大 collision `<=0.01`、最大 road-any `<=0.05`。当前 bridge-1 阈值为 `0.55`，后续 bridge/hard-a/hard-b 均为 `0.50`。攻击门禁当前不要求 instant/dwell 达标，也未把 attack road-termination 单独写入布尔判断（现有结果该项均为0）。
- 单轮未过门禁不会立即停止：若 clean 通过、攻击碰撞/越界仍在安全范围，且最低完成率不低于本阶段历史最佳，则保留该候选作为下一轮父模型累计训练。
- 当一个阶段用完2轮仍无任何候选同时通过 clean和攻击门禁时，脚本打印 `no candidate passed` 并以状态码 `8` 退出；systemd 会将其显示为 `failed`，这是门禁主动停止，不等于程序崩溃。
- 本次停止的直接条件：hard-bridge-1/r1 和 r2 的三种子最低完成率都为 `0.4375 < 0.55`；虽然 clean=1、碰撞=0、越界=0，仍在第2轮后触发阶段失败退出。
- 其他前置/异常退出：初始 runs 或轨迹缺失、课程配置/起始索引非法会 exit 2；发现另一个 Isaac train/eval/play 工作负载会 exit 3；由于全局 `set -euo pipefail`，任一训练、评估、绘图或最终视频流水线返回非零也会停止。
- 只有所有8个课程阶段逐级通过，写出 `hard_b_multiseed_gate.txt`，并且最终攻击梯度数据/图片/视频流水线也成功完成，整个服务才以状态码0结束。
- 自动监测 `2026-09-07T17:20:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-07_15-18-15_ams_hard_curriculum_20260907_hard_bridge_1_r2_happo_meta_ams`，update `900`，已完成评估 `8`，最近完成率/道路越界/碰撞 `0.5625/0.0/0.0`，错误数 `0`。

## 2026-09-07 17:22 修正 Ackermann 动作 FDI 并继续课程

- 根因复核确认 pipeline 顺序正确：策略动作先经过攻击器，再经过安全盾，最终还会按动作上限裁剪；本次不是攻击绕过安全盾。
- 修正了动作攻击语义：单车 Ackermann 动作为 `[纵向驱动, 前轮转向]`，此前 `max_fdi_acc` 对两个维度都施加随机 FDI，等价于把“加速度攻击”错误扩展成额外的转向攻击。现在默认 `action_fdi_longitudinal_only=true`，加速度 FDI 只进入纵向驱动通道；DoS 仍按完整动作通信通道处理，不改变前轮转向/后轮不可转向的车辆模型。
- 同时修正生成式/CA-GAN 攻击器的动作维度：不再使用硬编码 `action_dim=4`，改为从环境推断单车实际 `action_dim=2`，并在生成器训练目标、参考攻击样本和 rollout 注入中统一使用纵向 FDI 掩码。新增攻击配置与 CSV 字段，便于后续审计实际动作维度和掩码状态。
- 校验通过：4个 Python 文件 `py_compile`、恢复脚本 `bash -n`、`git diff --check` 均通过；独立 profile/generator 单测均确认输出形状为 `(env, 5, 2)`、纵向扰动非零、转向 FDI 非零元素为0，generator action dimension 为2。
- 旧 `ams_hard_curriculum_20260907` 目录完整保留，不覆盖历史结果；其旧监控 timer 已停止。新课程从旧 hard-bridge-1/r2 的 clean/safety 合格模型暖启动，结果写入 `logs/rsl_rl/platoon_happo/ams_longitudinal_fdi_20260907`。
- 新服务 `ackermann-ams-longitudinal-fdi-20260907.service` 已于 `17:21:25 +08:00` 启动并保持 active/running，当前正在重新执行 `component_screen/full_hard`。随后会自动执行 position-only、acceleration-only、DoS-only，再进入 hard-bridge-1 累计训练。
- 课程参数保持每块240 updates、ramp 120、clean fraction 0.35；每阶段最多轮数从2增至3，但 bridge-1 门槛仍为0.55、后续门槛仍为0.50，没有降低完成率或安全标准。新每小时 timer `ackermann-ams-longitudinal-fdi-monitor-20260907.timer` 已启用。
- hard-b 严格门禁通过前仍不会启动六算法全攻击强度最终对比；通过后再自动生成汇总数据、图片和真实变道视频。
- 自动监测 `2026-09-07T17:22:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-09-07T18:23:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-09-07_18-20-41_ams_longitudinal_fdi_20260907_hard_bridge_1_r2_happo_meta_ams`，update `1194`，已完成评估 `4`，最近完成率/道路越界/碰撞 `0.5/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T19:24:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-09-07_18-57-01_ams_longitudinal_fdi_20260907_hard_bridge_1_r3_happo_meta_ams`，update `1620`，已完成评估 `10`，最近完成率/道路越界/碰撞 `0.53125/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T20:25:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-07_18-57-01_ams_longitudinal_fdi_20260907_hard_bridge_1_r3_happo_meta_ams`，update `1620`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.53125/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-07T21:25:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-07_18-57-01_ams_longitudinal_fdi_20260907_hard_bridge_1_r3_happo_meta_ams`，update `1620`，已完成评估 `12`，最近完成率/道路越界/碰撞 `0.53125/0.0/0.0`，错误数 `0`。

## 2026-09-07 22:07 纵向 FDI 修正后课程结果

- 新服务已于 `19:32:48 +08:00` 在 hard-bridge-1 完成三轮后按门禁以状态码8停止，当前无 Isaac train/eval/play 工作负载；这是门禁主动退出，不是进程崩溃。每小时监测 timer 仍 active。
- 修正后的固定种子组件筛查：position-only completion `1.0`、DoS-only `1.0`、acceleration-only `0.3125`、full-hard `0.25`，碰撞和道路越界均为0。相较修正前 acceleration-only `0.1875`，纵向 FDI 语义修正带来改善，但加速度扰动仍是主要瓶颈。
- hard-bridge-1/r1 三攻击种子 completion（env42/env3407/env777）为 `0.25/0.0/0.25`，最低 `0.0`、平均 `0.1667`；r2 为 `0.4375/0.125/0.3125`，最低 `0.125`、平均 `0.2917`；r3 为 `0/0/0`。三轮都低于门槛 `0.55`。
- 三轮严格 clean completion/instant/dwell 均为 `1.0`；所有 clean/attack 的 collision、road-any、road-termination 均为0，基础完整变道和安全性没有退化。阶段内安全最佳候选是 r2（跨种子最低 completion `0.125`），r3 未替代它。
- 训练 rollout 并非没有成功信号：r1/r2/r3 的 completion 均值约 `0.414/0.417/0.419`，最大值 `0.656/0.625/0.625`；但独立攻击验证无法稳定维持2秒 dwell，r3 虽有 instant completion `0.1875/0.25/0.3125`，ever/dwell completion 均为0。
- 尾车仍是明确瓶颈：攻击验证中 leader 平均路径进度约 `0.837–0.838`，第5车仅约 `0.617–0.624`；clean 第5车约 `0.688–0.689`。安全盾训练期 trigger rate 约 `99.45%`，scale mean 约 `0.232`，说明纵向攻击下队尾追赶与安全盾强干预之间仍不匹配。
- 当前新目录产物为 `35 CSV / 4 JSON / 6 PT / 0 PNG / 0 MP4`；没有进入 bridge-2、hard-a、hard-b，也尚未触发六算法最终对比和视频流水线。
- 建议下一步不再盲目追加同配置轮数：以 r2 安全最佳模型为起点，加强 minimum path progress/尾车 gap 的密集奖励，并对纵向 catch-up 与 shield 触发做受控调整；保持 clean 和碰撞/越界门禁不变后重新验证 bridge-1。

## 2026-09-07 22:21 统一 CTH 车距逻辑并启动 gap-stability 课程

- 对上一轮失败逐步指标复核后，确认严格完成的直接限制项是车距稳定性，不是路径、横向或航向：攻击验证的路径进度能达到1.0，最大中心线误差约 `0.04–0.05 m < 0.12 m`，最大航向误差约 `0.03–0.04 rad < 0.12 rad`，最大相邻横向误差约 `0.17–0.185 m < 0.20 m`；但 whole-platoon 最大纵向 gap error 的时序均值约 `0.43–0.54 m`，超过完成阈值 `0.35 m`，从而无法稳定保持2秒 dwell。
- 找到 shield 与完成条件的车距模型冲突：奖励/完成条件使用 CTH `d_des=1.5+0.6*v`，正常行驶时约为1.7 m；旧追赶盾却在固定 gap `>=1.5 m` 时触发并把纵向动作置为0.4，导致正常移动间距也被长期当成掉队，容易产生纵向振荡。
- 已将追赶阈值统一为动态 CTH：`d_trigger=d_drop + 0.60*v_follower + 0.10 m`。当前速度0/0.35/0.45 m/s对应阈值1.60/1.81/1.87 m；只有明显大于动态期望车距时才强制追赶。保留 `dynamic_catchup_gap=false` 的旧固定阈值兼容入口。
- 已启用此前为0的本地 gap 密集惩罚：`local_reward_gap_coef=0.25`，最后一车 gap scale=`2.0`，让策略直接学习降低阻止 dwell 的纵向间距误差；其余 centerline/heading/turn、retention BC/KL 和攻击课程保持不变。
- 修正 shield 诊断指标：历史 `shield_trigger_rate≈0.995` 实际是“该批32个环境中是否至少一个触发”的 batch 指标，`shield_scale_mean≈0.232` 实际是投影后动作绝对值均值，并非缩放比例。新运行将 `shield_trigger_rate` 改为真实的逐环境触发率，并新增 `shield_batch_trigger_rate`、`shield_catchup_rate`、`shield_brake_rate`，以后可以区分纵向追赶、制动与横向修正。
- 校验通过：Python `py_compile`、恢复脚本 `bash -n`、`git diff --check` 通过；动态阈值单测确认 `[0,0.35,0.45] m/s -> [1.60,1.81,1.87] m`，关闭动态模式时仍为固定1.50 m。
- 新服务 `ackermann-ams-gap-stability-20260907.service` 于 `22:20:14 +08:00` 启动，父模型为纵向 FDI 课程的安全最佳 r2；新目录 `logs/rsl_rl/platoon_happo/ams_gap_stability_20260907`。当前 active/running，正在以新 shield 对 r2 做 `component_screen/full_hard`，配置已由 IsaacLab 成功解析，无启动错误。
- 新每小时监控 `ackermann-ams-gap-stability-monitor-20260907.timer` 已启用，旧纵向-FDI失败服务的监控 timer 已停止。课程仍为每块240 updates、ramp120、clean fraction0.35、每阶段最多3轮；clean/碰撞/越界和 bridge-1 `0.55` 门槛均未降低。
- 自动监测 `2026-09-07T22:21:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `n/a`，update `n/a`，已完成评估 `0`，最近完成率/道路越界/碰撞 `n/a/n/a/n/a`，错误数 `0`。
- 自动监测 `2026-09-07T23:22:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `training`，当前运行 `2026-09-07_23-20-19_ams_gap_stability_20260907_hard_bridge_1_r2_happo_meta_ams`，update `1662`，已完成评估 `4`，最近完成率/道路越界/碰撞 `0.0/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T00:23:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-09-07_23-57-05_ams_gap_stability_20260907_hard_bridge_2_r1_happo_meta_ams`，update `2100`，已完成评估 `10`，最近完成率/道路越界/碰撞 `0.4375/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T01:24:56+08:00` [正常]：六算法攻击矩阵服务 `active`，模式 `evaluation`，当前运行 `2026-09-08_01-09-56_ams_gap_stability_20260907_hard_bridge_3_r2_happo_meta_ams`，update `2580`，已完成评估 `16`，最近完成率/道路越界/碰撞 `0.53125/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T02:25:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T03:25:28+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T04:25:28+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T05:25:29+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T06:25:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T07:25:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T08:25:31+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T09:25:31+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T10:25:32+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T11:25:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T12:25:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T13:25:34+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T14:25:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T15:25:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T16:25:36+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T17:25:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `evaluation`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T18:25:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T19:25:38+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T20:25:39+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T21:25:39+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。

## 2026-09-08 22:20 gap-stability 课程结果

- gap-stability 改进有效，组件筛查在同一 r2 父模型上发生显著提升：full-hard completion 从修改前 `0.25` 提升至 `0.6875`，acceleration-only 从 `0.3125` 提升至 `0.625`；position-only 与 DoS-only 仍为 `1.0`，四类碰撞/越界均为0。full-hard 平均 gap error 从约 `0.3107 m` 降至 `0.1800 m`。
- hard-bridge-1/r1 最低/平均攻击完成率 `0.375/0.500`；r2 提升到 `0.5625/0.6458`，三种子（env42/env3407/env777）为 `0.750/0.625/0.5625`，严格 clean=1、安全项全0，正式通过 `0.55` 门槛并晋级。
- hard-bridge-2 首轮直接通过：三种子完成率 `0.625/0.5625/0.500`，最低/平均 `0.500/0.5625`；严格 clean=1，碰撞、越界、越界终止均为0。
- hard-bridge-3（攻击预算 position/acceleration/DoS=`2.65/0.71/0.115`）未通过：r1 三种子 `0.3125/0.625/0.375`，最低/平均 `0.3125/0.4375`；r2 为 `0.375/0.25/0.375` 且 clean 只有 `0.9375`；r3 从安全最佳 r1 回滚起训后为 `0.1875/0.125/0.0625`。阶段安全最佳仍是 r1，最低攻击完成率 `0.3125`。
- 训练期 gap 指标较旧课程大幅降低：bridge-1/r2 gap error absolute mean 约 `0.117 m`、completion max-gap error 约 `0.190 m`；bridge-2/r1 分别约 `0.111/0.180 m`。动态追赶率约 `0.16`，制动率与 critical rate 均为0；新逐环境 shield trigger rate 约 `0.85`，而 batch-any rate 约 `0.98`，证实新统计能区分二者。
- 主服务于 `2026-09-08 02:23:04 +08:00` 在 bridge-3 用完3轮后按门禁以状态码8主动退出，当前无 Isaac train/eval/play 工作负载；不是崩溃或仍在后台训练。每小时监控 timer 仍 active。
- 当前目录产物 `62 CSV / 9 JSON / 12 PT / 0 PNG / 0 MP4`。尚未进入 hard-a、hard-b，因而六算法最终攻击矩阵和真实视频尚未触发。
- 自动监测 `2026-09-08T22:25:40+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-08T23:25:41+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T00:25:41+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T01:25:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T02:25:43+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T03:25:43+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T04:25:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T05:25:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T06:25:45+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T07:25:46+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T08:25:46+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T09:25:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T10:25:48+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T11:25:48+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T12:25:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T13:25:50+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T14:25:50+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T15:25:51+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T16:25:51+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T17:25:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T18:25:53+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T19:25:53+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T20:25:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T21:25:55+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T22:25:55+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-09T23:25:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T00:25:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T01:25:57+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T02:25:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T03:25:59+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T04:25:59+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `training`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T05:26:00+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T06:26:01+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T07:26:01+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T08:26:02+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T09:26:03+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T10:26:04+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T11:26:04+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T12:26:05+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T13:26:06+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T14:26:06+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T15:26:07+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T16:26:07+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T17:26:08+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T18:26:09+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T19:26:09+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T20:26:10+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T21:26:11+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T22:26:11+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-10T23:26:12+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T00:26:12+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T01:26:13+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T02:26:14+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T03:26:14+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T04:26:15+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T05:26:16+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T06:26:16+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T07:26:17+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T08:26:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T09:26:18+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T10:26:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T11:26:19+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T12:26:20+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T13:26:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T14:26:21+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T15:26:22+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T16:26:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T17:26:23+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T18:26:24+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T19:26:25+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T20:26:25+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T21:26:26+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T22:26:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-11T23:26:27+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T00:26:28+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T01:26:29+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T02:26:29+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T03:26:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T04:26:30+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T05:26:31+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T06:26:32+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T07:26:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T08:26:33+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T09:26:34+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T10:26:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T11:26:35+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T12:26:36+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T13:26:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T14:26:37+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T15:26:38+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T16:26:38+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T17:26:39+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T18:26:40+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T19:26:40+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T20:26:41+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T21:26:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T22:26:42+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-12T23:26:43+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T00:26:44+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T01:26:45+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T02:26:45+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T03:26:46+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T04:26:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T05:26:47+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T06:26:48+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T07:26:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T08:26:49+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T09:26:50+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T10:26:51+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T11:26:51+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T12:26:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T13:26:52+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T14:26:53+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T15:26:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T16:26:54+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T17:26:55+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T18:26:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T19:26:56+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T20:26:57+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T21:26:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T22:26:58+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-13T23:26:59+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。
- 自动监测 `2026-09-14T00:27:00+08:00` [警告]：六算法攻击矩阵服务 `failed`，模式 `none`，当前运行 `2026-09-08_01-46-40_ams_gap_stability_20260907_hard_bridge_3_r3_happo_meta_ams`，update `2580`，已完成评估 `24`，最近完成率/道路越界/碰撞 `0.46875/0.0/0.0`，错误数 `0`。

## 2026-09-14 GPT-6 新对话交接

- 用户将切换到已打开的 GPT-6 新窗口继续项目；新对话应先完整读取本文件，再检查代码、systemd 服务、运行进程和最新实验产物，避免重复或冲突运行。
- 当前有效结论：gap-stability 已严格通过 hard-bridge-1 和 hard-bridge-2；hard-bridge-3 尚未通过，安全最佳为 `hard_bridge_3_r1`。当前主服务已按门禁退出，无有效训练继续推进；六算法最终攻击矩阵、数据图、真实变道视频和俯视轨迹尚未执行。
- 新对话继续要求：从安全最佳模型细化 bridge-2 到 bridge-3 的攻击课程，保持 clean、碰撞、越界和完成率门禁不降低；hard-b 通过后完成六算法、全攻击强度、全部数据/图片/视频对比。
- 持久要求：每次对话结束前将关键进展追加回本文件。

## 2026-09-14 01:04 接管初查（GPT-6）

- 已分段完整读取本文件原有1635行；首次整文件输出被截断，随后用连续 sed 范围补齐读取。仓库 HEAD 仍为 c262642；存在18个既有 tracked 修改及大量未跟踪文件，后续只改本任务相关文件并保留原状。
- 宿主机检查命令：`systemctl --user list-units --all --no-pager 'ackermann*' '*ams*' '*cagan*'`、`systemctl --user list-timers --all --no-pager`、`pgrep -af 'scripts/reinforcement_learning|run_ackermann|run_platoon|isaac-sim/kit'`、`nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader`。沙箱首次无法访问用户总线/GPU，已通过正式提权工具重查宿主机。
- AMS gap-stability 主服务实际 failed，旧监控 timer 仍 waiting；但当前 GPU 并非空闲：隔离目录 IsaacLab-cagan-redesign 的 `latest-paper-heldout-cagan-bank-seed888-20260914.service` 正在训练，Isaac PID 1550395、64 env，RTX3080显存3724 MiB、利用率40%。因此当前存在外部 Isaac 冲突，不能立即并发启动 AMS。
- 本轮尚未更改训练代码或启动仿真；下一步核对 gap-stability 门禁、r1模型及逐步数据，在外部任务运行期间完成课程/验证/证据导出的离线修复与测试，再按全局空闲条件串行续跑。不会把旧监控中的其他训练模式当作AMS进展。

### 01:08 门禁与失败机制复核

- 宿主机 journal/systemctl 确认主服务在9月8日02:23:04以8退出；桥1/r2最低完成率0.5625、桥2/r1为0.500，均通过原严格clean与安全门禁。桥3/r1三种子为0.3125/0.625/0.375，clean全部1，碰撞/越界/越界终止全0；r2 clean=0.9375，r3最低0.0625，均不应继承。
- r1检查点实际路径：`logs/rsl_rl/platoon_happo/2026-09-08_00-33-13_ams_gap_stability_20260907_hard_bridge_3_r1_happo_meta_ams/model_final.pt`，大小10557959字节，SHA256=`c76c0f214d34c1c79fe0a6c4cfc8e6cddd2c4c538b19a4c9ba6d75a662ede8ba`。gap-stability目录实际62 CSV/9 JSON/12 PT/0 PNG/0 MP4；可用磁盘519G。
- 独立读取r1的4份3500行逐步CSV：三个攻击seed在末500个非reset步的五车最低路径进度均为1，最大车距误差时序均值为0.5117/0.4888/0.5403m；clean为0.2863m。瓶颈是完成路径后的车距稳定和2秒dwell，不是第五车尚未走完整条路径。均值路径进度不应被误解为终点进度。
- 当前代码确认五车使用mini_akm URDF与2D AckermannAction；后轴lb/rb只写速度目标，前转向left/right只写位置目标；空间参考按每车x查询同一条路径。
- 验证实现仍有风险：只凭summary非空复用、不核查checkpoint/预算/步数；攻击门禁未检查attack enabled/road termination；validate_round放在if内导致shell errexit失效，子评估失败可能继续读旧文件。准备修复为可追溯、完整性校验及明确错误返回，并把桥2→3细分为小步课程。完成率阈值不降低；新攻击安全验收收紧至全程零碰撞/零越界/零道路终止。

### 01:15 验证修复与离线产物

- 已增加 `scripts/tools/ackermann_validation.py`，用checkpoint/源码/命令和CSV的SHA256验证缓存，核查3500步、16环境、5车、实际攻击预算、无专家、纵向FDI、全强度和原始安全指标；任何失败明确中止，不读取旧summary冒充本轮结果。新增单进程单checkpoint保护。
- `run_ackermann_ams_multiseed_recovery.sh` 增加 bridge2_refined：桥2与桥3之间插入25%/50%/75%小步，桥3后增加hard-a中点；每阶段先评估同强度父模型，避免首个失败候选无条件替换更强父模型。仍使用真实轨迹BC10/KL1保留、专家控制0，完成率门禁0.50不下降。
- 新增阶段绘图器；接入真实renderer视频和env0轨迹导出，修复ladder的off开关、全攻击强度、单级导出、COMMON配置传递。已完成bash语法与Python编译检查；新验证器离线读取r1四份历史CSV成功，复现clean1和攻击0.3125/0.625/0.375（仅离线审计，不为旧文件签发新实验证明）。历史六阶段奖励/进度/车距/安全图正在CPU生成。
- 01:14:51宿主机复查：外部CA-GAN服务仍active，进入s01_light训练，显存3720MiB、GPU36%，不能启动并行AMS。下一步完善负面回归测试、串行等待启动器及持久状态记录；外部任务完全退出后再启动safe-r1的clean与新课程。原文件备份在 `backups/20260914_ams_takeover/`。

### 接管修复离线验收与排队准备

- CPU负面回归 `python3 scripts/tools/test_ackermann_validation.py`：5组测试全部通过，覆盖重复seed、clean退化、缺步/乱序、CSV篡改、NaN、关闭攻击、弱化强度、clean混入、专家介入、错误车数/action维度及单步越界不得被均值稀释。`bash -n`、Python编译、`git diff --check`均通过。
- 历史绘图首次通配符误包含stage.log导致中断，已修复仅处理目录并完整重跑。六个历史阶段已新增30张PNG（每阶段4张验证诊断+1张训练奖励）；`hard_bridge_3_r1/plots/attack_env42_seed3407_progress_gap_safety.png`已人工查看：五车按顺序到达路径末端，但车距误差约0.45–0.55m，完成率停在31.25%。图明确标为多环境诊断，不能代替真实视频或XY轨迹证据。
- 串行启动器 `scripts/tools/run_ams_refined_when_idle.py` 已配置：核对safe-r1 SHA256、文件锁、等待外部CA-GAN整个systemd服务退出、无train/play/eval进程且GPU<1500MiB/利用率<10%连续4次；30秒更新queue_status.json。恢复参数240更新、120渐进、起始scale0.98、actor LR5e-7、clean混合0.35；专家控制始终0。
- 下一步注册独立 `ackermann-ams-bridge-refined-20260914.service` 运行等待器；停用旧AMS监控timer以避免把外部CA-GAN当成AMS进展。该服务先clean+三种子父模型对照；父模型已过某一级时直接保存证据并升级，避免重复训练；否则训练最多3轮，每轮验证/真实视频/轨迹后才决定晋级。hard-b与六算法矩阵仍未通过/完成，不预报成功。

### 2026-09-14 01:19:20 AMS串行续跑服务

- 已排队，等待外部服务 `latest-paper-heldout-cagan-bank-seed888-20260914.service` 完全退出及连续4次GPU空闲。输出 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_bridge_refined_20260914`；不终止任何外部任务。

### 2026-09-14 01:20 本轮交接状态

- 实际执行：`systemctl --user stop ackermann-ams-gap-stability-monitor-20260907.timer`；`systemd-run --user --unit=ackermann-ams-bridge-refined-20260914 --description='AMS refined Ackermann curriculum, queued for exclusive GPU' --property=WorkingDirectory=/home/cnc/SSD_1T/xzw/IsaacLab-main --property=Restart=no --property=KillMode=control-group /usr/bin/python3 /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/run_ams_refined_when_idle.py`。
- 宿主机验收：新服务active/running，主PID1554204，01:19:20启动；旧AMS监控timer已inactive。01:19:50状态文件明确 `waiting_for_exclusive_gpu`、external_state=active、quiet_checks=0；CA-GAN仍占用约3.7GiB GPU，AMS没有启动训练或评估。
- 新结果根：`logs/rsl_rl/platoon_happo/ams_bridge_refined_20260914/`；实时状态 `queue_status.json`，启动后日志 `continuation.log` 和 `launch_config.json`。本轮只产生离线审计/30张PNG及排队状态，没有新训练成功率或新视频；仍以桥1最低56.25%、桥2最低50%、桥3/r1最低31.25%为有效进度。
- 当前结论：hard-bridge-3、hard-b、六算法完整矩阵均未完成。下一步由已运行服务在资源释放后自动执行严格clean、细分课程和逐级证据；门禁失败会保留失败数据并停止，不降低阈值继续。六算法旧驱动另有非fresh评估和只检查AMS完成率的问题，尚未修复或接入自动链路；必须在hard-b实际通过后按独立进程/严格clean与安全协议完成六算法对比。

## 2026-09-14 AMS与桌面MGRS论文创新性对照（只读分析）

- 用户询问相对桌面 `Security Control of Vehicle Platoons under Hybrid Attacks via Meta-Gradient Reward Shaping.pdf` 的创新点。本轮用pdfinfo/pdftotext读取12页论文（整文件输出截断后分章节补读），并检查ams.py、runner.py、student.py、actor.py、AckermannAction及r1 env.yaml；未改变训练算法或进程。
- 论文既有贡献：物理信息Reward Teacher、双层MGRS、上下行CA-GAN、渐进攻击预算、训练安全投影、一阶代理元梯度与条件局部最终有界；仿真为scalar纵向控制/直线参考，五车、六算法最高强度对比，实物为虚拟领航+四实车圆轨跟随。仿真对比保留投影，实物移除Teacher/投影。不能将这些已有设计重复声明为AMS原创。
- 当前方法增量：联合动作辅助双Q、QR正交对称邻域平均TD目标、局部Q优势混入HAPPO（权重0.02）；五车2D速度/前转向同空间路径变道；真实Isaac轨迹BC与参考策略KL保留+clean/多seed门禁的课程续训。注意AMS基本思想明确来自Lin的AMS论文/代码 https://github.com/Lin-Zhihao98/AMS ，已联网核查原始作者/仓库，不能宣称首创动作流形平滑。
- 重要边界：当前Q邻域是归一化动作空间QR扰动+clip，并未定义Ackermann可达流形或攻击方向自适应度量；neighborhood variance只记录未加入鲁棒风险惩罚；runner命名executed_actions实际来自Student保存的原始策略动作，不能称为真实执行动作因果建模。局部优势在所有车间广播，尚非逐车反事实信用分配。
- r1实际attack_mode=profile，不是CA-GAN；shield仍开启。不能直接与论文CA-GAN/部署去投影结果比较，更不能称AMS已实现无shield鲁棒控制。现有桥3未过、缺相同协议AMS开关消融，暂不足以宣称优于MGRS。下一步建议先完成同任务Meta/AMS/组合的2×2消融，再分离保留学习与shield贡献。
- 拟向用户给出的判断：最可讲的方法增量是“物理奖励校正+联合动作价值平滑”的两层稳定化及其on-policy HAPPO适配；最明确的任务增量是Ackermann纵横耦合、同空间路径全队变道；保留学习+严格晋级为训练协议贡献；全车联合dwell完成率为评价贡献。基础AMS、BC、KL、渐进课程均非本项目首创。建议后续真正强化Ackermann/攻击语义感知的可行邻域与局部风险约束，但目前没有实现，不作为既有创新。
- 01:22:50读取queue_status.json：服务仍waiting_for_exclusive_gpu，外部active，GPU3781MiB/38%，本轮没有新训练/评估指标或视频。已有产物路径及有效门禁结论不变；本轮唯一写入为本工作记录。

## 2026-09-21 新论文思路的代码差距审计

- 用户询问按三项新论文贡献目前哪些代码需要修改。本轮只读检查 git 状态、Teacher/AMS/runner/pipeline/router/wrapper、AckermannAction、轨迹/完成门禁、模仿数据和训练配置；未修改代码、未启动训练。
- 已具备雏形：`happo/ams.py`有联合双Q与QR正交对称邻域平均，`teacher.py`有物理分数、shaping和外层协方差代理，`ackermann_action.py`已实现2D速度+前轮转向且后轮仅速度目标，`trajectory.py`已有同空间路径和五车dwell完成判据，`actor.py`已有BC/KL保留损失，验证脚本已有clean/多seed/安全门禁。
- 最高优先级缺口：`wrappers.py`中的攻击/安全盾后实际动作只用于`env.step()`和诊断；`router.observe_step`传给Student/Teacher/AMS的仍主要是`pending_actions`原始策略动作。需要把raw、attacked、shield-projected、Ackermann physical executed四种动作写入StepBatch/extras并进入rollout buffer，Teacher物理评分和AMS anchor/next target使用真实执行动作，否则“物理奖励校正+执行动作价值平滑”叙述不成立。
- AMS算法层需要改：邻域采样从归一化欧氏动作改为Ackermann可达动作（速度、前轮转角/曲率、转角变化率约束后再映射回策略空间）；增加邻域Q方差或下置信界风险项；局部优势按车辆/角色保留，不把全局AMS优势无条件广播给五车；明确使用执行动作与攻击/盾状态，增加配置与日志。
- Teacher层需要改：外层更新应绑定完整post-update物理窗口、使用执行动作和未受攻击的scene状态，明确记录raw/attacked/executed物理代价；当前是first-order covariance代理，论文应称代理方法，若声称严格meta-gradient则需实现可微一阶surrogate或降低理论表述；Teacher与AMS应明确更新顺序和独立消融开关。
- Ackermann任务层需补：保留前轮转向/后轮不转向单元测试；将路径进度、路径切向/曲率、变道阶段、CTH误差等作为版本化观测特征，并同步更新模仿数据重建与旧checkpoint兼容策略；验证动作邻域不会生成不可达或越过转角/速度/道路约束的样本。当前同路径和完成判据已有，不应重写成新贡献。
- 能力保持课程需改：当前BC/KL是常数式runner retention update，且新攻击脚本常显式关闭专家路径跟踪；需要显式阶段状态（clean acquisition→expert decay→attack adaptation→hard promotion）、专家blend/BC/KL/clean fraction按update调度、冻结安全最佳策略作为KL参考、失败候选回退；每阶段记录clean regression、attack safety、全车dwell和expert=0证据。当前课程脚本的晋级逻辑可复用但需接入算法阶段状态。
- 工程层需改：`PlatoonAlgorithmCfg`在`agents.py`和`config.py`有重复定义，应集中或同步；增加`method_variant`/`ams_geometry`/`ams_risk_coef`/`use_executed_action_trace`等配置；新增Teacher×AMS、Retention on/off、Ackermann-neighborhood on/off的2×2/3×2消融入口和统一评估协议。
- 推荐实施顺序：1）动作链路和StepBatch真实执行动作闭环；2）Ackermann可达邻域及局部信用；3）Teacher严格窗口/指标；4）能力保持阶段调度；5）消融、单元测试和新训练。暂不应先改奖励权重或直接重跑hard-b。

### 新论文三项贡献表述整理

- 用户认可前三个方向，要求整理为新论文创新点。本轮仅进行论文写作整理，不调整训练、不启动实验、不生成新的性能结论；未重新检查服务，不能将上次排队状态作为本轮实时状态。
- 统一主线：面向混合攻击下Ackermann车队协同变道，围绕奖励目标失配、联合动作价值估计不稳定以及攻击适应中的基础机动能力遗忘，构建物理奖励校正、联合动作价值平滑与能力保持课程相结合的方法。
- 创新点1（方法）：在既有MGRS与AMS基础上，构建物理奖励校正与联合动作邻域价值平滑协同的多智能体学习框架。Teacher利用训练期物理信息调整奖励；辅助双Q通过正交对称邻域平均构造TD目标，并将局部Q优势接入on-policy HAPPO更新。贡献定位为协同机制及多智能体适配，而非首次提出MGRS或AMS，不声称尚未建立的Lipschitz/安全理论保证。
- 创新点2（控制问题）：将纵向跟驰扩展为非完整Ackermann约束下的纵横耦合协同变道。每车速度与前轮转向联合决策、后轮不转向，以同一空间路径及每车位置确定局部参考；在攻击引起非同步到达时维持路径和队形协调，目标为全车完成并满足路径/航向/车距/道路约束。不是仅更换URDF；Ackermann结构与学习平滑邻域的可达性不能混同。
- 创新点3（训练）：设计基础机动能力保持的渐进攻击适应机制，先用物理仿真实际轨迹进行模仿暖启动并撤除专家控制，再结合示范BC、参考策略KL与clean/攻击混合训练，按无攻击能力和多种子攻击安全/完成门禁晋级与回退。目标是抑制遗忘，而非保证无遗忘；渐进课程、BC与KL均为借鉴组件，论文增量在任务针对性的组合、选模规则及其验证。
- 写作要求：正文使用“提出/构建/设计/旨在”，不预写“显著优于”“已保证稳定”“完成实车验证”；实验需补Teacher×AMS消融、能力保留消融和统一攻击/安全盾设置。待实测后填入性能改善与理论结果。本轮产物为用户回复中的中文引言贡献草稿及本记录，现有hard-b/六算法未完成结论不变。

### 2026-09-14 02:50:33 AMS串行续跑服务

- 外部服务及Isaac进程已退出，GPU连续4次空闲；开始safe-r1严格clean/父模型基线，然后执行bridge2_refined。
- 参数：32训练环境/16评估环境、3500步、240更新/120渐进、起始强度0.98、actor LR5e-7、clean混合0.35、每级最多3轮、BC10/KL1、专家控制0。
- 命令：`bash scripts/tools/run_ackermann_ams_multiseed_recovery.sh`；完整环境参数见 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_bridge_refined_20260914/launch_config.json`。

### 2026-09-14 03:35:01 AMS串行续跑服务

- 状态 `stopped_at_gate_or_error`，exit=1；日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_bridge_refined_20260914/continuation.log`。
- 必须读取各validation_gate.json核实实际进度；未通过不可升级。六算法完整矩阵仍待hard-b通过后执行，不能以此服务退出推断已完成。

### 2026-09-22 00:18 新论文思路第1至第6步代码落地（未启动训练）

- 实际状态检查：`ackermann-ams-bridge-refined-20260914.service` 当前为 `failed`，外部 CA-GAN 服务为 `inactive`；未发现本项目 `IsaacLab-main` 的 `train.py`、`eval_happo_platoon.py` 或 `play.py` 进程。本轮没有启动、停止或接管任何训练/评估任务；另一个 `/home/cnc/SSD_1T/xzw/IsaacLab-cagan-redesign` 的历史训练进程不属于本项目，未修改。
- 第1步（真实执行动作闭环）：`StepBatch.extras`、HAPPO rollout buffer、Student、Router、Wrapper 现在传递并保存 `policy → attacked → shield → executed_semantic` 四阶段动作；Ackermann action term 额外记录物理速度/前轮转向目标。Teacher 物理评分及 AMS anchor 使用执行语义动作，策略 log-prob 仍使用原策略动作，避免把攻击/盾动作误当策略动作。
- 第2步（Ackermann AMS）：AMS 增加 `geometry/wheelbase/max_steering/curvature/risk_coef/use_executed_action` 配置；反事实邻域先投影到速度-前轮转角可行集合，局部优势按车辆逐车构造，其他车辆固定执行动作；执行转移本身不再被事后投影重写；邻域方差进入风险惩罚项。
- 第3步（Teacher 窗口）：Teacher 外层信号按完整窗口平均物理目标与前一完整窗口比较，并在上下文中记录 `teacher_window_steps`、`teacher_post_policy_update`、`teacher_outer_mode=first_order_window_covariance_proxy`。仍明确是一阶协方差代理，不宣称严格可微 meta-gradient。
- 第4步（Ackermann/空间观测）：空间观测特征可选扩展为 progress、参考曲率、CTH gap error；模仿数据重建同步支持这些字段且在新特征模式下缺字段会显式报错，旧11维模式保持兼容。修正了先使用 `rel_terms` 后计算的顺序问题，并统一参考速度为平面速度范数。
- 第5步（能力保持课程/状态）：新增 `clean_acquisition → expert_decay → awaiting_clean_validation → attack_adaptation → awaiting_hard_validation` 状态；攻击混合只有在外部验证器用 checkpoint 哈希绑定的 clean completion/instant/dwell 与零碰撞/越界/道路终止证据调用 `approve_clean_gate()` 后才开始。Router checkpoint 保存 pipeline 状态、方法变体和 observation schema；重启后自动撤销 clean 批准，要求重新验证。默认 legacy 配置不改变既有训练行为。
- 第6步（消融与对比入口）：新增 `method_variant`（legacy/baseline/mhrs/ams/retention/combined_no_retention/combined）并同步到两份任务配置；新增 `scripts/tools/run_innovation_ablation.sh`，默认只生成六算法×攻击强度命令清单，不执行训练；`RUN_ABLATION=1` 才显式执行。`run_medium_algorithm_comparison.sh` 增加 `METHOD_VARIANT` 覆盖。
- 测试命令及结果：`python3 -m py_compile`（全部本轮修改的 Python 文件）通过；`bash -n scripts/tools/run_medium_algorithm_comparison.sh scripts/tools/run_innovation_ablation.sh` 通过；`python3 scripts/tools/test_ams_ackermann.py` 输出 `AMS_ACKERMANN_TESTS_OK`；`python3 scripts/tools/test_ackermann_validation.py` 为 `Ran 5 tests ... OK`；`git diff --check` 通过；干运行 `OUT_ROOT=/tmp/ams_innovation_manifest_test ATTACK_LEVELS='off hard-b' bash scripts/tools/run_innovation_ablation.sh` 成功生成 `commands.txt`，未启动训练。
- 当前实验结论未改变：hard-bridge-3 尚未严格通过，hard-b 与六算法完整攻击矩阵仍未完成；不得把上述静态测试或旧 r1 指标写成新实验结果。下一步是先由外部严格 clean 验证器调用课程批准接口，再从安全最佳模型串行训练/验证桥3；hard-b 通过后才执行六算法全强度矩阵，并为每个阶段保存 checkpoint 哈希、CSV、奖励曲线、XY/俯视轨迹、变道图和视频。
- 00:19 修订：`run_innovation_ablation.sh` 默认强度清单扩展为 `off light easy_0 easy_a easy_b med_a med_b hard_a hard_b`，仍保持 `RUN_ABLATION=0` 干运行，不会启动任何训练。

### 2026-09-22 01:00 启动桥3续训队列

- 完整读取本记录（1730行/470552字节）；当前 `hard_bridge_3_r1/validation_gate.json` 仍为 `passed=false`，clean completion=1.0，攻击 completion=0.625/0.3125/0.375，碰撞/越界均为0，不能晋级。
- 首次直接启动 `ackermann-ams-bridge3-20260922.service` 被安全检查以 exit=3 拒绝：外部 `/home/cnc/SSD_1T/xzw/IsaacLab-cagan-redesign` 正在运行 easy 阶段训练（PID 1696533/1696538）；未终止该任务。
- 已改为持久等待服务 `ackermann-ams-bridge3-20260922-wait.service`，当前 active/running。它等待所有 Isaac train/play/eval 进程消失且 GPU memory<1500MiB、utilization<10%，随后执行：
  `ROOT=/home/cnc/SSD_1T/xzw/IsaacLab-main INITIAL_RUNS_FILE=.../hard_bridge_3_r1/training/runs.txt RECOVERY_ROOT=.../ams_bridge_refined_20260922 CURRICULUM_PROFILE=bridge2_refined START_PHASE_INDEX=3 NUM_ENVS=32 EVAL_NUM_ENVS=16 BLOCK_UPDATES=40 ATTACK_RAMP_UPDATES=40 TRAIN_CLEAN_FRACTION=0.50 MAX_ROUNDS_PER_PHASE=3 EVAL_STEPS=3500 bash scripts/tools/run_ackermann_ams_multiseed_recovery.sh`。
- 该流程会先用当前源码哈希重新生成 clean 验证；只有 clean 通过才进行 bridge-3 refined 训练/三种子攻击验证。当前没有新的训练指标或晋级结果。
- 01:01 状态复查：`ackermann-ams-bridge3-20260922-wait.service` 仍为 active/running；外部 easy 训练 PID 1696533/1696538 仍存在，GPU 约 509 MiB、利用率29%，等待条件尚未满足。桥3 clean/训练尚未启动，指标与门禁结论不变。

### 2026-09-30 13:40 进度核实：桥3与hard-b已通过，最终导出中断

- 用户本轮询问进度。本轮仅检查服务、journal、磁盘、checkpoint/CSV校验和视频元数据，并追加本记录；未修改算法、启动训练、重启服务或改变已有实验产物。
- 实际检查命令：`systemctl --user show ackermann-ams-bridge3-20260922-wait.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p Result -p ExecMainStartTimestamp -p ExecMainExitTimestamp`；`journalctl --user -u ackermann-ams-bridge3-20260922-wait.service --no-pager`；`pgrep -af 'scripts/reinforcement_learning/.*/(train|play|eval_happo_platoon)\.py'`；`nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`；Python只读调用 `ackermann_validation.load_verified()`；`ffprobe` 检查hard-b视频。当前无匹配的Isaac训练/评估进程，GPU约464MiB/18%。
- 9月22日队列实际执行了续训，不再处于等待。桥3第2轮严格通过，三组攻击完成率0.50/0.625/0.5625；hard-a-mid与hard-a父模型直接通过；hard-b-bridge-1第2轮、bridge-2父模型、bridge-3第3轮均通过；hard-b第2轮于9月22日20:56完成验证并于21:04晋级。
- 七份promoted_gate均实际核对到对应阶段的4份verified.json/原始CSV；每阶段checkpoint SHA一致、评估源码SHA一致，CSV与checkpoint内容校验通过。桥3三种子最低50%；hard-a-mid最低56.25%；hard-a最低50%；hard-b-bridge-1最低50%；bridge-2最低50%；bridge-3最低62.5%；hard-b最低62.5%、均值70.8333%。门禁阈值仍为0.50。
- hard-b最终模型：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-09-22_20-29-34_ams_bridge3_20260922_hard_b_r2_happo_meta_ams/model_final.pt`，SHA256=`faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7`。正式门禁：`logs/rsl_rl/platoon_happo/ams_bridge_refined_20260922/stages/hard_b_r2/validation_gate.json` 与 `hard_b_promoted_gate.json`，均passed=true。
- hard-b评估协议：16环境×3500步，攻击预算FDI-pos=4.0、FDI-acc=1.30、DoS=0.18；三个独立环境/攻击seed的全车ever完成率为0.625/0.75/0.75。clean completion/instant/dwell均1.0；所有clean/attack评估的全程collision/road_any/road_termination最大值均0，expert blend为0。攻击instant峰值0.75/0.6875/0.8125、dwell峰值0.3125/0.375/0.5，不将ever完成率表述为末帧同时完成率或所有环境100%完成。
- 产物根：`logs/rsl_rl/platoon_happo/ams_bridge_refined_20260922`，实有213 CSV、81 PNG、12 MP4、18 PT。hard-b证据：`stages/hard_b_r2/evidence/hard_b/trajectory_env0.csv`有17500行，五车各3500行，各车reference_progress最大值均1；`topdown_true_xy.png`、`reward_attack_safety.png`已存在；`video/rl-video-step-0.mp4`为H264、1280×720、3500帧、70秒、2611536字节。该env0轨迹与视频是单环境展示，不能代替多环境完成门禁。
- 当前服务是failed/MainPID=0/exit=1，退出时间9月22日21:05:02；失败发生在已晋级hard-b之后的 `final_ladder/off/evaluation`。日志明确报错：`/algorithm/max_fdi_pos`期望float却收到int，源于ladder的off预算写为`0`而非`0.0`；因此全强度最终导出未完成。hard-b独立验证通过的证据仍完整。
- 六算法全攻击强度对比仍未完成，未发现9月22日续跑之后启动的新Ackermann六算法矩阵；现有旧六算法目录/服务不能算本轮完成。下一步应修复off浮点预算并从已通过hard-b checkpoint补齐全强度证据，不重复桥3/hard-b训练；随后修复/使用严格统一协议执行六算法矩阵。本轮为状态回复，没有执行这些下一步操作。

### 2026-10-02 22:06 当前进度复核

- 用户本轮仅询问当前进度。读取记录末尾，检查systemctl/pgrep、hard_b_promoted_gate.json、hard_b_multiseed_gate.txt及产物修改时间，并调用ackermann_validation.load_verified逐一核验hard_b_r2四份评估；无代码修改、重启或新训练/评估。
- 当前无匹配Isaac train/play/eval进程；ackermann-ams-bridge3-20260922-wait.service仍failed、MainPID=0、exit=1，退出时间仍9月22日21:05:02。AMS结果根最新文件亦停留在该时刻，9月30日后没有新增AMS实验进展。
- hard-b四份评估checkpoint/CSV完整性校验仍通过：clean全车完成率1.0；攻击三种子0.625/0.75/0.75，最低0.625、均值0.708333；碰撞/越界/道路终止均0，原门禁0.50且passed=true。桥3/hard-a/hard-b既有通过结论不变。
- 产物/模型仍为：logs/rsl_rl/platoon_happo/ams_bridge_refined_20260922/stages/hard_b_r2，checkpoint为2026-09-22_20-29-34_ams_bridge3_20260922_hard_b_r2_happo_meta_ams/model_final.pt。已有70秒真实视频和全车轨迹/俯视图；全强度final_ladder仍中断于off预算整数/浮点类型错误，六算法全部强度完整对比仍未完成。
- 下一步仍是修复off预算类型并用现成hard-b模型补齐全强度证据，随后执行严格统一协议六算法矩阵；本轮按状态请求未启动这些操作。

### 2026-10-02 22:13 GPU占用归因（只读）

- 用户看到GPU仍在活动，询问正在做什么。本轮执行 `nvidia-smi`、`nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`、`nvidia-smi pmon -c 1 -s um`、`ps -eo pid,ppid,stat,etime,%cpu,%mem,comm,args --sort=-%cpu`、`systemctl --user list-units --type=service --state=running`，未启动或停止任何程序。
- 22:13:28 GPU快照：RTX3080显存872MiB/10240MiB、GPU利用率16%、功率124W、温度63℃。占用为Xorg PID2794/279MiB，GNOME Shell PID2967/69MiB，Edge GPU进程PID57932/33MiB，ToDesk视频会话PID83371/366MiB，iKuuuVPN PID1816264/56MiB，向日葵awesun PID4117273/35MiB。
- 单次pmon采样显示ToDesk SM约7%、编码器约7%，GNOME Shell SM约6%；ToDesk命令为 `/opt/todesk/bin/ToDesk_Session --localPort=35600 --isVideoSession=true`，活动来自远程桌面画面编码/桌面渲染，未发现Isaac/PyTorch实验GPU进程。pmon为瞬时采样，不能解释所有历史峰值。
- 实验状态、模型与产物路径均不变：hard-b已通过，但AMS没有运行。下一步实验仍为修复全强度导出和补齐六算法矩阵；本轮仅提供GPU占用解释，不操作远程桌面进程。

### 2026-10-02T22:45:37 AMS后续实验

- 开始后续服务mode=all，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×600更新。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/followup_status.json`。

### 2026-10-02 22:47 全强度导出修复与后续串行服务落地

- 用户授权继续下一步。本轮先核对笔记、现有脚本和原始验证；启动前无Isaac train/play/eval进程、GPU compute列表为空，未动ToDesk/Xorg等桌面进程。旧等待服务仍failed，未重启旧课程；已有hard-b四份load_verified/CSV/checkpoint完整性检查通过，clean completion/instant/dwell=1.0，攻击三seed=0.625/0.75/0.75，全程碰撞/越界/道路终止=0。模型SHA仍为faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7，不重复桥3/hard-b训练。
- 修改 `scripts/tools/run_platoon_spatial_attack_ladder.sh`：off预算全部改成0.0；对单级攻击预算验证并规范化float；复用验证必须匹配模型、3500步、环境数、seed、攻击开关/级别/三个预算；off严格clean未通过则停止导出后续攻击。旧脚本备份于 `backups/20261002_ams_export_resume/`。
- 新增 `scripts/tools/run_ackermann_verified_followup.py`；旧 `run_ackermann_full_six_attack_matrix.sh` 改为严格驱动入口。先验证已通过hard-b参考，再串行补齐参考模型off→hard_b九强度的三攻击seed验证、CSV、奖励/安全图、env0真实视频/全五车XY与俯视轨迹；参考证据独立存入 `reference_ams_ladder`，不混入新六算法模型数据。
- 六算法部分采用同一真实轨迹BC模型（SHA6b8c15acf17b7e250a65fc85609c304ba2cc14f8d0a65d8f34b4188f6e801615）；保存BC来源报告和轨迹哈希，绝不以已攻击训练的AMS模型给其他算法暖启动。每算法先16环境×3500步、expert=0严格clean；必要时clean训练最多3×240更新；攻击级别单算法逐级训练，每级最多3×600更新、训练32环境/clean混合0.35/起始强度0.25/120更新渐进。六算法共同保留BC10/KL1、同路径、同完成判据、同安全盾参数；训练/评估/视频的不同shield脚本默认值已显式统一。
- 固定预声明晋级阈值：light/easy_0/easy_a/easy_b三seed最低完成率≥0.75，med_a≥0.65，med_b≥0.55，hard_a/hard_b≥0.50；每候选clean completion/instant/dwell必须=1且clean/attack全程碰撞/越界/道路终止最大值=0。失败模型保留但不晋级、不成为下一轮父模型；失败3轮后该算法停止更强攻击训练，用最后安全通过模型做全部强度独立assessment，输出停止级别，不能称为已通过该强度。任何clean获取失败不得解锁攻击。
- 驱动增加独占GPU文件锁、竞争进程检查、评估/训练超时、状态JSON、命令日志、源码/checkpoint/产物哈希和可验证复用；部分训练目录/视频归档保存，不覆盖用户原实验。绘图增加六算法训练奖励曲线，以及矩阵三seed均值/最低/标准差、全程安全最大值、晋级状态和模型SHA，避免只看AMS或安全均值掩盖失败。
- 测试：`python3 -m py_compile scripts/tools/run_ackermann_verified_followup.py scripts/tools/plot_ackermann_six_attack_matrix.py scripts/tools/plot_ackermann_recovery_stage.py scripts/tools/test_ackermann_followup.py`通过；`bash -n`三个驱动通过；`python3 scripts/tools/test_ackermann_validation.py`5测试通过；`python3 scripts/tools/test_ackermann_followup.py`4测试通过（clean失败锁攻击、off浮点、失败候选回退、晋级必须3seed）；`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/test_ams_ackermann.py`输出AMS_ACKERMANN_TESTS_OK；`git diff --check`通过。未修改/回滚已有dirty worktree其他文件。
- 执行预检：`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/run_ackermann_verified_followup.py --output logs/rsl_rl/platoon_happo/ams_verified_followup_20261002 --mode all --preflight-only`，通过参考hard-b原始验证、BC来源与协议固化；首版预检manifest保留为 `preflight_manifest_224456.json`，正式版本为 `followup_manifest.json`。
- 实际启动：`systemd-run --user --unit=ackermann-ams-followup-20261002 --description='Verified AMS ladder evidence then isolated six-method curriculum' --property=WorkingDirectory=/home/cnc/SSD_1T/xzw/IsaacLab-main --property=Restart=no --property=KillMode=control-group --property=StandardOutput=append:/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/service.log --property=StandardError=inherit /home/cnc/Downloads/ENTER/bin/python3 -u /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/run_ackermann_verified_followup.py --output /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002 --mode all`。
- 22:46复查：服务active/running、主PID196224、启动22:45:36；实际Isaac评估PID196234、GPU compute约2862MiB/总显存3382MiB、利用率29%。当前任务是hard-b固定模型的off/clean复核，不是桥3重新训练；已成功解析0.0预算、创建cuda:0的16环境并进入仿真初始化，未见原float/int错误。完整3500步尚未结束，不能把预检/启动写成新clean通过或视频/矩阵完成。
- 产物根 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/`；实时 `followup_status.json`、总日志 `service.log`；当前原始日志 `reference_ams_ladder/off/validation/clean_env42/evaluation.log`，命令及验证器日志 `reference_ams_ladder/off/validation/clean_env42_validator.log`。后续六算法阶段 `stages/<algorithm>/`，最终全强度 `final_ladders/<algorithm>/`，结果矩阵/热图在根目录。SSD可用487GiB。
- 下一步由已运行持久服务完成严格clean→参考模型九强度证据→六算法独立课程及全强度assessment；门禁/执行失败即记录并停止相关推进，不自动降低门禁。本轮新完成的是代码修复、静态/CPU验证和实际服务启动，完整六算法矩阵仍未完成。

### 2026-10-02T22:59:27 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/off`。

### 2026-10-02T23:25:32 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/light`。

### 2026-10-02T23:51:28 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/easy_0`。

### 2026-10-03T00:17:37 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/easy_a`。

### 2026-10-03T00:43:41 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/easy_b`。

### 2026-10-03T01:09:56 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/med_a`。

### 2026-10-03T01:35:59 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/med_b`。

### 2026-10-03T02:02:17 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/hard_a`。

### 2026-10-03T02:28:30 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/hard_b`。

### 2026-10-03T03:19:19 AMS后续实验

- 后续服务中止：RuntimeError: Command exit=1; log=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/stages/mappo/light_r1/evidence/evidence_export.log；状态/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002`。已完成验证保留，未宣告后续矩阵成功。

### 2026-10-03 18:52 进度核查：AMS九档证据完成，六算法在MAPPO首轮中断

- 用户本轮询问进度。仅执行只读状态/日志/原始验证/产物检查及追加本记录；未修改算法或驱动，未启动、停止、重启训练服务。检查命令：`systemctl --user show ackermann-ams-followup-20261002.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p Result -p ExecMainStartTimestamp -p ExecMainExitTimestamp`；`tail`读取service.log、工作记录、失败证据日志及阶段训练日志；`pgrep -af 'scripts/reinforcement_learning/.*/(train|play|eval_happo_platoon)\.py'`；`nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`；Python逐档调用ackermann_validation.load_verified/digest；`ffprobe`读取hard-b视频元数据。
- 实际服务failed/MainPID=0/exit=1，运行时间为10月2日22:45:36至10月3日03:19:19，当前没有Isaac训练/评估/播放进程。GPU快照约540MiB/8%，compute列表仅ToDesk_Session PID241629/366MiB，不是AMS训练。
- AMS参考固定模型九档（off、light、easy_0、easy_a、easy_b、med_a、med_b、hard_a、hard_b）的validation_gate均passed=true，ladder_complete.json存在，10月3日02:28:30完成最后hard-b证据。off严格clean completion/instant/dwell均1.0；light至med_b六档的三个攻击seed完成率全1.0；hard_a为1.0/0.9375/0.9375；hard_b为0.625/0.75/0.75，最低62.5%、均值70.8333%。完成率是全车ever/dwell事件比例，不冒充末帧同时完成率。
- 本轮逐档重验clean/攻击verified.json与原始CSV/checkpoint完整性，并核对全部evidence_complete.json记录的图/视频/轨迹文件SHA（off5项，其余每档7项），均通过。所有原始评估全程碰撞/越界/道路终止最大值为0，协议仍16环境×3500步，expert=0；hard-b预算4.0/1.30/0.18，旧安全最佳模型SHA仍faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7。
- AMS九档各有真实视频和env0轨迹，轨迹各17500行（五车×3500步）；hard-b视频h264、1280×720、3500帧/70秒，五车各自reference_progress最大值都为1.0。单环境展示不替代多环境严格门禁。产物根 `logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/reference_ams_ladder/`，各档video/、trajectory_env0.csv、topdown_true_xy.png、reward_attack_safety.png均已保存。本次后续实验总目录实有92 CSV、22 PNG、10 MP4、2 PT（不含外部run目录中的模型）。
- 六算法已启动第一种MAPPO，但未完成任何算法的全攻击课程。MAPPO共同BC初始模型严格clean为completion/instant/dwell=1.0且全安全指标0；保存clean_seed/off视频。随后light_r1完成600训练更新，614400 timesteps、训练1804.4秒，model_final.pt及奖励曲线/CSV已保存。新候选重新严格clean得到completion/instant/dwell全部0.0，碰撞/越界/道路终止仍0；这是基础变道能力门禁失败，不是碰撞门禁失败，没有进行该候选三seed攻击验证或晋级。
- MAPPO失败候选SHA9e544c015f823561e347129b36f80a1f4b1e289e9d9a3d507ea4；模型 `logs/rsl_rl/platoon_happo/2026-10-03_02-42-41_ams_verified_followup_20261002_mappo_light_r1_mappo/model_final.pt`；门禁 `.../ams_verified_followup_20261002/stages/mappo/light_r1/validation_gate.json`，原始clean数据 `validation/clean_env42/`，奖励曲线 `plots/mappo_training_reward.png`。
- 串行流程中断的直接原因：驱动对clean失败候选调用off证据导出，而ladder按严格clean门禁拒绝，日志明确 `Strict clean gate failed; attack ladder remains locked`，错误向上传播导致整个服务退出。原计划的安全父模型回退及light_r2/r3未执行，其他五种算法也未开始。不是又出现float/int错误，也不影响已完成的AMS九档证据；目前不能称六算法完整对比已完成。
- 下一步需修复“失败clean诊断证据与正常晋级导出”混用的错误处理路径，保留失败CSV/图且不解锁攻击；再从通过clean的BC安全父模型处理MAPPO训练退化，按原clean/完成率/零碰撞越界门禁继续，并保留已完成AMS证据避免重复。本轮仅报告状态，未实施这些修改或续跑。

### 2026-10-03T19:11:46 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/followup_status.json`。

### 2026-10-03T19:23:25 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/followup_status.json`。

### 2026-10-03T19:23:25 AMS后续实验

- 已保存MAPPO旧失败候选的70秒clean-only诊断视频；completion=0.0，仍拒绝晋级，续训父模型为通过clean的BC，不使用该失败模型。诊断 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/prior_failure_diagnostics/mappo_light_r1`。

### 2026-10-03T19:23:25 AMS后续实验

- 复用同checkpoint/源码/协议的原始评估 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/stages/mappo/clean_seed/validation/clean_env42` → `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/clean_seed/validation/clean_env42`；没有重跑仿真。

### 2026-10-03T19:23:26 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[277075, 280814]，待运行命令 `bash`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/training_driver.log`。未终止任何外部进程。

### 2026-10-03T19:24:11 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-03 19:26 失败clean流程修复、诊断视频补存与真实续训

- 用户授权“进行修复，然后继续进行训练”。启动前核查工作记录/驱动/训练配置/原始CSV、服务及GPU：旧服务failed/MainPID=0，无Isaac train/play/eval；GPU compute仅ToDesk，不终止桌面会话或外部任务。原AMS九档passed门禁及evidence_complete图/视频/轨迹SHA本轮再次核对通过，参考模型和已有产物保持不变，未重复桥3/hard-b或九档导出。
- 退化核查：MAPPO旧light_r1的BC损失由首更新0.000676升至600更新0.064315，reference KL由0.000100升至1.487493；旧模型actor optimizer LR确为1e-6、BC初始化optimizer列表为空，并非错误继承了旧大LR。原始clean CSV的五车路径progress都达到1，但末段聚合中心线误差约0.144m（阈值0.12m）、车间侧向偏差约0.223m（阈值0.20m），导致whole-platoon completion/instant/dwell=0。因此“走到路径终点”不等于严格完成，不能放宽判据。CSV的真实XY是世界坐标、reference_y为局部参考，不能直接相减把环境原点偏移误报成道路越界。
- 修复 `run_platoon_spatial_attack_ladder.sh`：新增显式DIAGNOSTIC_CLEAN_ONLY=1；只接受单off级别、攻击关闭、三个预算均0及哈希校验完整的原始数据。正常导出入口仍要求严格clean通过；诊断单独写diagnostic_only=true/attacks_unlocked=false/gate_passed=false。新增模式不允许任何攻击展示或模型晋级。
- 修复 `run_ackermann_verified_followup.py`：clean失败候选存入failed_clean_diagnostics而非正常晋级证据路径，门禁保持false，完成诊断后继续从原安全父模型重试，不再因预期clean失败而退出整个六算法服务；正常完成训练基础门禁仍锁定三seed攻击评估。新增诊断视频/PNG/CSV完整性、3500帧/70秒视频验收及产物SHA缓存，失败和部分视频可恢复归档保留。
- 训练协议更新独立版本2：六算法同一BC初始模型，32训练/16评估环境，严格3500步评估；攻击每级最多3轮×40更新（替代旧600更新长块），actor LR5e-7，训练clean混合0.50，强度起始scale0.25/ramp_updates=块更新数；BC10/KL1和expert=0保持。AMS/Meta分支的最终覆盖优先级已测试，不能被common中的1e-6悄悄覆盖。新训练参数适用于全部六算法，旧600更新结果独立保留，不与新协议混为同一次公平比较。
- 门禁未降低：light/easy三seed最低完成率≥0.75，med_a≥0.65，med_b≥0.55，hard_a/hard_b≥0.50；clean completion/instant/dwell峰值均1、全程碰撞/越界/道路终止均0、专家0。失败候选不得续作父模型；六算法完成矩阵仍待后续实际训练/验证。
- 复用规则：新增--reuse-root；只允许模型SHA、完整原始CSV哈希、评估源码SHA、环境/seed/3500步/攻击预算及完整命令语义相同的评估，唯一可不同的命令参数是输出文件路径；BC展示还核对旧manifest、五车×3500行和视频。MAPPO BC clean及展示已经校验复制到新结果根，没有重新跑这份clean或视频；不会复用旧失败600更新模型来继续学习。旧AMS参考证据仍在原reference_ams_ladder，mode=matrix不重复导出。
- 原失败候选实际诊断：19:11启动本方修复服务录制旧MAPPO失败模型，真实play.py运行攻击关闭的1环境；原float预算、严格协议核对和diagnostic-only标记正确通过。录制已完成，h2641280×720/3500帧/70秒、17500行五车轨迹、奖励/安全图和俯视轨迹已保存且SHA校验通过。产物 `logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/prior_failure_diagnostics/mappo_light_r1/off/`，render_complete.json明确diagnostic_only=true，旧候选completion=0仍被拒绝。
- 并发问题及额外修复：我方19:11启动后，外部Teacher方向诊断 `teacher_direction_20261003/collect_direction.py` 于约19:13开始占用同GPU，原pgrep只覆盖标准train/play/eval路径，漏掉自定义Kit脚本。本轮临时用 `systemctl --user kill --kill-who=main --signal=SIGSTOP ackermann-ams-matrix-repair-20261003.service` 阻止本方launcher追加训练，当前子视频正常完成后，用SIGKILL只回收本方已暂停的主协调进程。没有终止外部Teacher任务，已经完成的CSV/图/视频保留，无数据删除。
- 新增通用/proc可执行路径检查：每次启动子任务前等待所有外部Isaac Kit Python及诊断协调器退出，15秒轮询，状态waiting_for_exclusive_gpu；覆盖自定义collect脚本及其launcher，防止在外部多cell任务间隙插入新训练。Coordinator-only manifest迁移只允许本驱动文件哈希变化，训练参数、门禁、核心算法、评估器和渲染器哈希任意变化都拒绝；旧manifest保留为followup_manifest.coordinator_previous_1791026604204305528.json；已完成诊断按原命令/标记/视频完整性验收后采用，无重复录制。
- 代码/回归验证：`python3 scripts/tools/test_ackermann_followup.py`11测试通过（包括复现原clean失败分支后继续18次安全回退、实际shell门禁正常失败/仅clean诊断成功/诊断攻击拒绝、覆盖优先级、缓存源码/seed/预算匹配、协调器迁移不能改变门禁、外部自定义Isaac等待）；`python3 scripts/tools/test_ackermann_validation.py`5测试通过；`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/test_ams_ackermann.py`输出AMS_ACKERMANN_TESTS_OK；py_compile、bash -n及git diff --check通过。修改前文件备份 `backups/20261003_ams_matrix_repair/`，保留其他dirty worktree更改。
- 预检命令：`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/run_ackermann_verified_followup.py --output logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003 --mode matrix --stage-updates 40 --actor-lr 5e-7 --reuse-root logs/rsl_rl/platoon_happo/ams_verified_followup_20261002 --diagnose-prior-failure logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/stages/mappo/light_r1 --resume-coordinator-update --preflight-only`，通过。
- 当前持久服务启动命令：`systemd-run --user --unit=ackermann-ams-matrix-repair-queued-20261003 --description='Repaired six-method training, wait for exclusive Isaac GPU' --property=WorkingDirectory=/home/cnc/SSD_1T/xzw/IsaacLab-main --property=Restart=no --property=KillMode=control-group --property=StandardOutput=append:/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/service.log --property=StandardError=inherit /home/cnc/Downloads/ENTER/bin/python3 -u /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/run_ackermann_verified_followup.py --output /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003 --mode matrix --stage-updates 40 --actor-lr 5e-7 --reuse-root /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002 --diagnose-prior-failure /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_followup_20261002/stages/mappo/light_r1 --resume-coordinator-update`。
- 19:23服务active/running、主PID281310；先正确处于waiting_for_exclusive_gpu，未并发启动训练。19:24:11外部九checkpoint Teacher诊断全部退出后自动继续，19:24:27生成真实MAPPO训练run；19:26:03训练达到25/40更新（learning iteration24/40），bc_loss=0.000702、reference_kl=0.005976；实际Isaac train PID281366/GPU约2850MiB，当前GPU任务是AMS六算法中的MAPPO续训，而不是旧Teacher诊断或桥3重训。初始BC clean完成/瞬时峰值/dwell峰值=1.0、全安全指标0，模型SHA6b8c15acf17b7e250a65fc85609c304ba2cc14f8d0a65d8f34b4188f6e801615。
- 新结果根 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/`；实时followup_status.json、service.log、training_driver.log、followup_manifest.json；MAPPO训练日志stages/mappo/light_r1.log；真实run `logs/rsl_rl/platoon_happo/2026-10-03_19-24-27_ams_verified_matrix_repair_20261003_mappo_light_r1_mappo/`。本轮新产物是失败诊断视频/图/数据、校验复用的安全BC数据、测试及正在运行的新训练；不能宣告新候选clean/攻击门禁已过或MAPPO退化已经实证解决。
- 下一步由持久服务完成本40更新块后独立3500步clean，再在clean通过条件下执行light三攻击seed验证、保存奖励曲线与真实视频/XY；失败保留诊断并回到安全父模型继续，逐算法/强度推进。同一GPU出现外部任务时自动排队，不降低安全/完成门禁，也不终止外部任务。

- 19:28交接复查：首个新MAPPO light_r1训练块已实际完成40更新，训练129.09秒，model_39.pt/model_final.pt已保存，training_complete.json及阶段tar.gz生成成功；服务于19:26:51自动进入该新checkpoint的16环境×3500步clean评估，19:28仍active/running。实际env.yaml已核对actor_lr=5e-7、attack_clean_env_fraction=0.5、start_scale=0.25、ramp_updates=40、expert_path_tracking_enabled=false/expert_blend_ratio=0。当前新候选尚无完整clean门禁结论，不能报告已晋级；下一步评估/安全回退由服务自动进行。

### 2026-10-03T19:59:39 AMS后续实验

- 六算法 `mappo` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/light_r1`。

### 2026-10-03T20:34:58 AMS后续实验

- 六算法 `mappo` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/easy_0_r1`。

### 2026-10-03T21:10:07 AMS后续实验

- 六算法 `mappo` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/easy_a_r1`。

### 2026-10-03T21:45:42 AMS后续实验

- 六算法 `mappo` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/easy_b_r1`。

### 2026-10-03T22:21:19 AMS后续实验

- 六算法 `mappo` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/med_a_r1`。

### 2026-10-03T22:57:00 AMS后续实验

- 六算法 `mappo` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/med_b_r1`。

### 2026-10-03T23:32:29 AMS后续实验

- 六算法 `mappo` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/hard_a_r1`。

### 2026-10-03 23:46 续训进度复核：MAPPO七档已过，hard-b评估中

- 用户询问当前进度。本轮仅检查服务/最新日志/原始验证/产物并追加本记录，未修改代码或训练参数、未重启/停止/启动实验。检查命令：`systemctl --user show ackermann-ams-matrix-repair-queued-20261003.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p Result -p ExecMainStartTimestamp -p ExecMainExitTimestamp`；读取followup_status.json、service.log及工作记录；Python调用ackermann_validation.load_verified/clean_passed/digest逐一检查八个MAPPO阶段的可用clean/attack及render_complete.json；`nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv`；ffprobe核对hard-a视频。
- 服务仍active/running，主PID281310，19:23:24启动，尚未退出；当前Isaac eval PID311937/GPU约2826MiB。状态为MAPPO hard_b第一组攻击评估env_seed42/attack_seed3407，23:41:22启动；23:46日志推进到约2540/3500步，无完整verified攻击结论，不能报hard-b通过。
- MAPPO已完成light_r1、easy_0_r1、easy_a_r1、easy_b_r1、med_a_r1、med_b_r1、hard_a_r1七个攻击阶段，全部首轮40更新严格通过，各阶段clean completion=1.0、instant峰值=1.0、dwell峰值=1.0。每阶段clean及三seed攻击checkpoint SHA与评估源码SHA一致，原始CSV/checkpoint内容检查通过；所有完整评估全程碰撞/越界/道路终止最大值0。
- light至med_b六档三seed全车ever/dwell完成率全为1.0；hard_a三seed为1.0/0.9375/0.9375，最低93.75%、均值95.8333%，门禁仍0.50。各级原阈值保留（light/easy0.75、med_a0.65、med_b0.55），没有通过放宽clean或安全门禁晋级。本轮短块续训在已验证的七阶段中未再出现旧600更新模型的clean=0退化，但不把这一结果写成所有算法已解决遗忘。
- 当前hard_b_r1的40更新训练已经完成，新模型 `logs/rsl_rl/platoon_happo/2026-10-03_23-32-46_ams_verified_matrix_repair_20261003_mappo_hard_b_r1_mappo/model_final.pt`；其严格clean已经load_verified核实completion/instant/dwell峰值均1.0、碰撞/越界/道路终止0，已解锁三seed full-budget hard-b验证。第一seed尚在执行，另两seed与hard-b阶段证据尚未完成，validation_gate.json尚未生成。
- 七个通过阶段各4项render_complete证据文件SHA全部复核通过；每阶段保存训练奖励图、验证进度/安全图、真实变道视频、五车轨迹CSV和俯视图。hard-a视频实际h264、1280×720、3500帧/70秒。新修复根当前实有108 CSV、53 PNG、9 MP4、16 PT（不计外部run目录其余checkpoint），根为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/`；阶段证据 `stages/mappo/<stage>/evidence/<level>/`，实时followup_status.json及service.log。
- 六算法目前只运行到第一种MAPPO；其余HAPPO-no-meta、HAPPO-Meta-AMS、shared-MAPPO、HAA2C、HATRPO尚未开始。MAPPO最终固定模型九档导出和六算法总矩阵CSV也尚未完成（six_algorithm_attack_matrix.csv不存在）。独立AMS参考模型原九档全通过证据仍保留，不重复执行。
- 下一步由正在运行的持久服务完成MAPPO hard-b剩余三seed验证及视频/轨迹，门禁成功才晋级；随后完成该算法最终九档assessment，再顺序推进另外五种算法。失败继续保留数据/诊断并从安全父模型回退，外部GPU任务出现则排队，不降低门禁。本轮无需重启服务。

### 2026-10-04T00:00:39 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[320780, 320781]，待运行命令 `/home/cnc/Downloads/ENTER/bin/python3`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/hard_b_r1/plotting.log`。未终止任何外部进程。

### 2026-10-04T00:17:57 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-04T00:26:01 AMS后续实验

- 六算法 `mappo` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.5625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/mappo/hard_b_r1`。

### 2026-10-04T00:40:09 AMS后续实验

- mappo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/off`。

### 2026-10-04T01:06:44 AMS后续实验

- mappo 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/light`。

### 2026-10-04T01:33:25 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/easy_0`。

### 2026-10-04T01:59:59 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/easy_a`。

### 2026-10-04T02:26:18 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/easy_b`。

### 2026-10-04T02:52:54 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/med_a`。

### 2026-10-04T03:19:27 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/med_b`。

### 2026-10-04T03:46:20 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/hard_a`。

### 2026-10-04T04:13:17 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/mappo/hard_b`。

### 2026-10-04T05:02:37 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/light_r1`。

### 2026-10-04T05:37:50 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/easy_0_r1`。

### 2026-10-04T06:13:16 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/easy_a_r1`。

### 2026-10-04T06:48:37 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/easy_b_r1`。

### 2026-10-04T07:23:59 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/med_a_r1`。

### 2026-10-04T07:59:19 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/med_b_r1`。

### 2026-10-04T08:34:27 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/hard_a_r1`。

### 2026-10-04T09:09:49 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.3125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/hard_b_r1`。

### 2026-10-04T09:45:16 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_b_r2`，passed=True，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_no_meta/hard_b_r2`。

### 2026-10-04T09:59:18 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/off`。

### 2026-10-04T10:25:56 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/light`。

### 2026-10-04T10:52:26 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/easy_0`。

### 2026-10-04T11:19:06 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/easy_a`。

### 2026-10-04T11:45:50 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/easy_b`。

### 2026-10-04T12:12:30 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/med_a`。

### 2026-10-04T12:39:11 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/med_b`。

### 2026-10-04T13:06:13 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/hard_a`。

### 2026-10-04T13:32:54 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_no_meta/hard_b`。

### 2026-10-04T14:21:23 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/light_r1`。

### 2026-10-04T14:56:06 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-04T15:30:55 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-04T16:05:38 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-04T16:40:30 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/med_a_r1`。

### 2026-10-04T17:15:30 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/med_b_r1`。

### 2026-10-04T17:50:38 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-04T18:25:50 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-04T18:39:34 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/off`。

### 2026-10-04T19:05:47 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/light`。

### 2026-10-04T19:32:03 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/easy_0`。

### 2026-10-04T19:58:14 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/easy_a`。

### 2026-10-04T20:24:18 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/easy_b`。

### 2026-10-04T20:50:36 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/med_a`。

### 2026-10-04T21:16:46 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/med_b`。

### 2026-10-04T21:42:59 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/hard_a`。

### 2026-10-04T22:09:13 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/happo_meta_ams/hard_b`。

### 2026-10-04T22:58:09 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/light_r1`。

### 2026-10-04T23:33:11 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_0_r1`，passed=False，clean_ok=True，攻击最低完成率=0.625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/easy_0_r1`。

### 2026-10-05T00:08:44 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_0_r2`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/easy_0_r2`。

### 2026-10-05T00:43:57 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/easy_a_r1`。

### 2026-10-05T01:19:03 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/easy_b_r1`。

### 2026-10-05T01:54:23 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_b_r2`，passed=False，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/easy_b_r2`。

### 2026-10-05T02:29:20 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_b_r3`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/easy_b_r3`。

### 2026-10-05T03:04:46 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `med_a_r1`，passed=False，clean_ok=True，攻击最低完成率=0.5625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/med_a_r1`。

### 2026-10-05T03:40:00 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `med_a_r2`，passed=False，clean_ok=True，攻击最低完成率=0.625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/med_a_r2`。

### 2026-10-05T04:15:17 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `med_a_r3`，passed=False，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_mappo_shared/med_a_r3`。

### 2026-10-05T04:29:22 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/off`。

### 2026-10-05T04:56:14 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/light`。

### 2026-10-05T05:22:43 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/easy_0`。

### 2026-10-05T05:49:02 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/easy_a`。

### 2026-10-05T06:15:42 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/easy_b`。

### 2026-10-05T06:42:27 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/med_a`。

### 2026-10-05T07:09:07 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/med_b`。

### 2026-10-05T07:35:37 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/hard_a`。

### 2026-10-05T08:02:22 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_mappo_shared/hard_b`。

### 2026-10-05T08:51:33 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/light_r1`。

### 2026-10-05T09:26:44 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/easy_0_r1`。

### 2026-10-05T10:02:06 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/easy_a_r1`。

### 2026-10-05T10:37:35 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/easy_b_r1`。

### 2026-10-05T11:13:07 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/med_a_r1`。

### 2026-10-05T11:48:16 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/med_b_r1`。

### 2026-10-05T12:23:49 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/hard_a_r1`。

### 2026-10-05T12:59:17 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.3125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/hard_b_r1`。

### 2026-10-05T13:34:56 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_b_r2`，passed=False，clean_ok=True，攻击最低完成率=0.375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/hard_b_r2`。

### 2026-10-05T14:10:19 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_b_r3`，passed=False，clean_ok=True，攻击最低完成率=0.3125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_haa2c/hard_b_r3`。

### 2026-10-05T14:24:24 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/off`。

### 2026-10-05T14:50:58 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/light`。

### 2026-10-05T15:17:39 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/easy_0`。

### 2026-10-05T15:44:03 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/easy_a`。

### 2026-10-05T16:10:44 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/easy_b`。

### 2026-10-05T16:37:20 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/med_a`。

### 2026-10-05T17:03:58 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/med_b`。

### 2026-10-05T17:30:48 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/hard_a`。

### 2026-10-05T17:57:40 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_haa2c/hard_b`。

### 2026-10-05T18:28:23 AMS后续实验

- 六算法 `harl_hatrpo` 阶段 `light_r1`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_hatrpo/light_r1`。

### 2026-10-05T18:45:08 AMS后续实验

- 六算法 `harl_hatrpo` 阶段 `light_r2`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_hatrpo/light_r2`。

### 2026-10-05T19:01:51 AMS后续实验

- 六算法 `harl_hatrpo` 阶段 `light_r3`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/stages/harl_hatrpo/light_r3`。

### 2026-10-05T19:15:47 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/off`。

### 2026-10-05T19:42:21 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/light`。

### 2026-10-05T20:09:10 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/easy_0`。

### 2026-10-05T20:35:32 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/easy_a`。

### 2026-10-05T21:02:05 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/easy_b`。

### 2026-10-05T21:28:42 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/med_a`。

### 2026-10-05T21:55:14 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/med_b`。

### 2026-10-05T22:22:04 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/hard_a`。

### 2026-10-05T22:48:47 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/final_ladders/harl_hatrpo/hard_b`。

### 2026-10-05T22:49:11 AMS后续实验

- 后续实验mode=matrix已完成，产物 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003`。

### 2026-10-09T11:06:08+0800 当前进度复核

- 以磁盘实际状态复核：`followup_status.json` 为 `state=complete, mode=matrix`，完成时间 `2026-10-05T22:49:10`；`six_algorithm_attack_matrix.csv` 共 54 行，即 6 个算法 × off/light/easy_0/easy_a/easy_b/med_a/med_b/hard_a/hard_b 九档完整评估。
- 当前无 Isaac 训练/评估/coordinator 进程。`ackermann-ams-matrix-repair-20261003.service` 是历史瞬态服务，systemd 保留为 failed/signal9；实际矩阵日志已正常写到 `complete`，不是尚未完成的运行任务。GPU 计算进程只有 ToDesk（约 366 MiB），没有 Isaac 进程。
- 训练到 hard-b 且九档矩阵门禁全部通过的算法：`mappo`（hard-b 三种子最低完成率 0.5625）、`happo_no_meta`（0.5000）、`happo_meta_ams`（0.7500）。三者所有 clean/off 完成率均 1.0，攻击阶段碰撞、越界、道路终止均为 0。
- 训练/门控未完整通过的算法：`harl_mappo_shared` 在 med_a 停止，固定模型 hard-b 评估门禁失败（最低 0）；`harl_haa2c` 训练到 hard-b 但固定模型 hard-b 失败（最低 0.4375）；`harl_hatrpo` 在 light clean 门禁停止，攻击评估均未通过（hard-b 最低 0）。这些仍已按协议完成九档 assessment，不能写成 hard-b 通过。
- 完整证据已生成：矩阵 CSV `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/six_algorithm_attack_matrix.csv`，热图 `completion_heatmap.png`、`completion_min_heatmap.png`、`return_heatmap.png`；六个算法各九档 final ladder 的视频、五车轨迹、俯视轨迹和奖励/安全图均在 `final_ladders/`。矩阵根目录当前统计约 109 个 MP4、452 个 PNG、1175 个 CSV、96 个 PT。
- 本轮仅做状态检查，未修改代码、未重启服务、未启动新实验。执行了 notes 完整读取、systemd/进程/GPU 查询、状态 JSON/CSV 汇总、产物计数和 git 工作树检查；工作树仍保留此前项目的既有修改与未跟踪文件。
- 下一步应基于 54 行矩阵和 `method_outcomes.json` 做论文级横向分析；若要提高结论强度，优先针对 `harl_mappo_shared`、`harl_haa2c` 和 `harl_hatrpo` 的 clean/Hard-B 失败单独修复和复验，不能覆盖或重写本次完整证据。

### 2026-10-09T12:34:36+0800 六算法实验结果汇总

- 按 `six_algorithm_attack_matrix.csv` 汇总：协议为 32 个训练环境、16 个评估环境、每阶段 40 updates、每个攻击强度最多 3 轮；攻击评估使用 3 个固定种子，clean/off 使用 1 个种子；每条完整变道评估为 3500 steps/70 s，五车共用同一空间变道路径，专家 blend=0，前轮 Ackermann 转向约束保持启用。
- 攻击强度参数阶梯为 light(0.75/0.10/0.02)、easy_0(1.0/0.18/0.035)、easy_a(1.25/0.25/0.05)、easy_b(1.5/0.35/0.07)、med_a(1.75/0.45/0.09)、med_b(2.0/0.5/0.10)、hard_a(2.75/0.75/0.12)、hard_b(4.0/1.3/0.18)，对应攻击门槛为 0.75、0.75、0.75、0.75、0.65、0.55、0.50、0.50。以下汇总均采用三攻击种子中的最低全车完成率。
- `happo_meta_ams`：八档攻击最低完成率为 1/1/1/1/1/1/1/0.75，平均 96.875%，全部门禁通过，训练到 hard-b；在 hard-b 上为六算法最高。
- `mappo`：1/1/1/1/1/1/0.9375/0.5625，平均 93.750%，全部门禁通过，训练到 hard-b。
- `happo_no_meta`：1/1/1/1/1/1/0.9375/0.50，平均 92.969%，全部门禁通过，训练到 hard-b。
- `harl_haa2c`：1/1/1/1/0.875/0.75/0.8125/0.4375，平均 85.938%；训练到 hard-b，但 hard-b 固定模型评估未过门禁。
- `harl_mappo_shared`：1/1/0.8125/0.875/0.6875/0.625/0.50/0，平均 68.750%；在 med_a 训练门禁停止，固定模型 hard-b 评估失败。
- `harl_hatrpo`：0.375/0.125/0.125/0.0625/0/0.125/0.0625/0，平均 10.938%；light clean 门禁失败后停止，后续攻击强度属于 assessment-only，不能视为训练通过。
- 54 个矩阵单元的碰撞率、道路越界率、道路终止率均为 0；六个算法 clean/off 全部全车完成率 1.0。与 HAPPO 无 Meta 相比，Meta+AMS 在 hard-b 的最低完成率提升 25 个百分点、平均攻击最低完成率提升约 3.906 个百分点；与普通 MAPPO 相比 hard-b 提升 18.75 个百分点。
- final_ladders 已保存 54 个视频、54 个奖励/安全图、54 个俯视真实 XY 图及多种子轨迹；矩阵根目录另有 `completion_heatmap.png`、`completion_min_heatmap.png`、`return_heatmap.png`、`method_outcomes.json`。本轮仅整理汇总，未改代码、未重启或重跑实验。

### 2026-10-09T17:24:19+0800 算法“是否最好”结论边界

- 当前可以严谨声称：在本项目统一 Ackermann 五车、同一路径、同一攻击阶梯、同一 clean/安全门禁和本轮固定评估种子下，`happo_meta_ams` 是已完成 hard-b 训练的比较方法中表现最好的；hard-b 最低整车完成率 75%，高于 MAPPO 56.25% 和 HAPPO 无 Meta 50%，八档攻击最低完成率均值 96.875% 也最高。
- 不能仅凭本轮结果声称“对所有随机种子、所有环境和所有攻击模型都普遍最优”或“统计显著领先”：每个算法目前只有一次训练流程，攻击侧为 3 个固定种子，且没有独立训练种子置信区间/显著性检验。
- 公平性限制：`harl_mappo_shared` 在 med_a、`harl_haa2c` 在 hard_b、`harl_hatrpo` 在 light clean 门禁停止；它们的后续九档是 assessment-only 或使用最后安全模型/BC 模型，不等价于完整 hard-b 训练预算。因此当前最强结论应限定为“统一门控协议下的最佳已验证结果”，并优先与完整训练到 hard-b 的 MAPPO、HAPPO 无 Meta 比较。
- 若要把“最好”升级为论文中的强结论，后续需要至少进行独立训练种子复验（建议 3–5 个）、报告均值±标准差/置信区间，并补充 HAPPO+Meta（不启用 AMS）与 HAPPO+AMS（不启用 Meta）的正交消融；当前矩阵已证明效果优势，但尚未完成这些统计和机制隔离证据。

### 2026-10-09T17:34:40+0800 正交消融已启动

- 已将正交设计落实为同一共享 HAPPO actor 架构的 2×2 因子：`happo_base_shared`（Meta=0, AMS=0）、`happo_meta`（Meta=1, AMS=0）、`happo_ams_only`（Meta=0, AMS=1），第四格 `happo_meta_ams`（Meta=1, AMS=1）复用 2026-10-03 六算法矩阵中同一协议/配置的已完成结果。这样不把 actor sharing 差异误判为 Meta/AMS 效果。
- 代码修改：`run_ackermann_verified_followup.py` 增加正交方法注册、方法子集参数 `--methods` 和动态门控计数；`run_medium_algorithm_comparison.sh` 增加 `happo_base_shared`/`happo_ams_only` 训练与评估分支，并显式关闭 Meta-only 的 AMS。历史六算法默认集合保持不变。
- 验证：`python3 scripts/tools/test_ackermann_followup.py` 11/11 通过，Python 编译和 `bash -n scripts/tools/run_medium_algorithm_comparison.sh` 通过；正交子集 preflight 通过。早期仅含两个分支的 preflight 目录为 `ams_orthogonal_ablation_20261009`，实际实验使用修正后的 `ams_orthogonal_ablation_20261009_v2`。
- 已启动持久服务：`ackermann-ams-orthogonal-20261009.service`；命令为 `systemd-run --user ... run_ackermann_verified_followup.py --mode matrix --methods happo_base_shared happo_meta happo_ams_only --output .../ams_orthogonal_ablation_20261009_v2 --stage-updates 40 --actor-lr 5e-7 --rounds 3`。当前 service active/running，PID 1183556，正在执行 `happo_base_shared` 的 clean/off 3500-step、16-env 严格评估；无并发 Isaac 任务。
- 新实验根目录：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v2`；状态文件为 `followup_status.json`，服务日志为 `service.log`。未覆盖原六算法矩阵。
- 后续顺序：先完成三新分支的 clean 门禁；通过后按 light→hard-b 逐级训练/三种子验证并导出视频、五车轨迹和图；完成后将三新分支与既有 `happo_meta_ams` 合并成正交 2×2 结果表，报告主效应和 Meta×AMS 交互效应。

### 2026-10-09T19:45:04+0800 正交消融首轮渲染修复与重启

- 首个正交服务 `ackermann-ams-orthogonal-20261009.service` 在 `happo_base_shared` 的 clean 评估完成后停止，评估本身通过（completion/instant/dwell=1.0，collision/road_any/road_termination=0）；失败点是 `run_platoon_spatial_attack_ladder.sh` 的标签白名单缺少新标签，日志为 `unsupported ALGORITHM_LABEL=happo_base_shared`，不是策略或门禁失败。
- 已在渲染脚本加入 `happo_base_shared`、`happo_meta`、`happo_ams_only` 三个明确的前轮 Ackermann/HAPPO 配置分支；Shell 语法、Python 编译及 11 项协调器测试再次通过。
- 为避免旧服务的 manifest/source identity 与渲染脚本修复混用，未覆盖 v2 目录，改用隔离的 v3 根目录重新排队：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3`。v2 保留为失败诊断证据，v3 是当前正式消融。
- 当前服务 `ackermann-ams-orthogonal-20261009-v3.service` 为 active/running（PID 1350710），状态为 `happo_base_shared/off` 的 evidence rendering；无并发 Isaac 任务。v3 状态文件和服务日志已开始写入，原六算法矩阵未修改。

### 2026-10-09T17:34:43 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v2/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v2/followup_status.json`。

### 2026-10-09T17:41:21 AMS后续实验

- 后续服务中止：RuntimeError: Command exit=5; log=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v2/stages/happo_base_shared/clean_seed/evidence/evidence_export.log；状态/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v2`。已完成验证保留，未宣告后续矩阵成功。

### 2026-10-09T19:44:36 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/followup_status.json`。

### 2026-10-09T19:44:36 AMS后续实验

- 复用同checkpoint/源码/协议的原始评估 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v2/stages/happo_base_shared/clean_seed/validation/clean_env42` → `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/clean_seed/validation/clean_env42`；没有重跑仿真。

### 2026-10-09T20:28:30 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/light_r1`。

### 2026-10-09T21:03:55 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/easy_0_r1`。

### 2026-10-09T21:40:03 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/easy_a_r1`。

### 2026-10-09T21:40:03 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[1393741]，待运行命令 `bash`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/training_driver.log`。未终止任何外部进程。

### 2026-10-09T21:43:18 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-09T21:46:00 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[1395321]，待运行命令 `/home/cnc/Downloads/ENTER/bin/python3`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/easy_b_r1/validation/clean_env42_validator.log`。未终止任何外部进程。

### 2026-10-09T21:51:00 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-09T22:23:59 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/easy_b_r1`。

### 2026-10-09T22:59:19 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/med_a_r1`。

### 2026-10-09T23:35:00 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/med_b_r1`。

### 2026-10-10T00:10:30 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/hard_a_r1`。

### 2026-10-10T00:46:23 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/hard_b_r1`。

### 2026-10-10T01:21:43 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r2`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/hard_b_r2`。

### 2026-10-10T01:57:15 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r3`，passed=True，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_base_shared/hard_b_r3`。

### 2026-10-10T02:11:14 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/off`。

### 2026-10-10T02:37:58 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/light`。

### 2026-10-10T03:04:34 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/easy_0`。

### 2026-10-10T03:31:26 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/easy_a`。

### 2026-10-10T03:58:17 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/easy_b`。

### 2026-10-10T04:25:00 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/med_a`。

### 2026-10-10T04:51:36 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/med_b`。

### 2026-10-10T05:18:25 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/hard_a`。

### 2026-10-10T05:45:22 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_base_shared/hard_b`。

### 2026-10-10T06:35:05 AMS后续实验

- 六算法 `happo_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/light_r1`。

### 2026-10-10T07:10:49 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/easy_0_r1`。

### 2026-10-10T07:46:27 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/easy_a_r1`。

### 2026-10-10T08:22:03 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/easy_b_r1`。

### 2026-10-10T08:57:43 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/med_a_r1`。

### 2026-10-10T09:33:13 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/med_b_r1`。

### 2026-10-10T10:08:50 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/hard_a_r1`。

### 2026-10-10T10:44:29 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/hard_b_r1`。

### 2026-10-10T11:20:14 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_b_r2`，passed=True，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_meta/hard_b_r2`。

### 2026-10-10T11:34:19 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/off`。

### 2026-10-10T12:00:58 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/light`。

### 2026-10-10T12:27:33 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/easy_0`。

### 2026-10-10T12:54:10 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/easy_a`。

### 2026-10-10T13:20:53 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/easy_b`。

### 2026-10-10T13:47:44 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/med_a`。

### 2026-10-10T14:14:28 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/med_b`。

### 2026-10-10T14:41:08 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/hard_a`。

### 2026-10-10T15:07:56 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_meta/hard_b`。

### 2026-10-10T15:56:02 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `light_r1`，passed=False，clean_ok=True，攻击最低完成率=0.25；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_ams_only/light_r1`。

### 2026-10-10T16:12:20 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `light_r2`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_ams_only/light_r2`。

### 2026-10-10T16:46:50 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `light_r3`，passed=False，clean_ok=True，攻击最低完成率=0.0625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/stages/happo_ams_only/light_r3`。

### 2026-10-10T17:00:33 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/off`。

### 2026-10-10T17:26:30 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/light`。

### 2026-10-10T17:52:32 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/easy_0`。

### 2026-10-10T18:18:32 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/easy_a`。

### 2026-10-10T18:44:48 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/easy_b`。

### 2026-10-10T19:10:58 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/med_a`。

### 2026-10-10T19:37:02 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/med_b`。

### 2026-10-10T20:03:06 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/hard_a`。

### 2026-10-10T20:29:19 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/final_ladders/happo_ams_only/hard_b`。

### 2026-10-10T20:29:31 AMS后续实验

- 后续实验mode=matrix已完成，产物 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3`。

### 2026-10-11T23:14:41+0800 正交消融完成结果复核

- v3 正交消融服务已完成并退出：`followup_status.json` 为 `state=complete, mode=matrix`，无活动 Isaac 训练/评估进程，GPU 仅有 ToDesk；服务没有运行异常。
- 三个新分支与既有 `happo_meta_ams` 的 manifest 在 protocol revision、攻击阶梯、32/16 环境、40 updates、3 rounds、actor LR、common overrides、BC checkpoint/SHA 上一致，能够组成同协议 2×2 结果；既有 full arm 的结果来自 `ams_verified_matrix_repair_20261003`。
- `happo_base_shared`（Meta=0, AMS=0）完整训练到 hard-b，八档攻击最低完成率 `[1,1,1,1,1,0.875,0.875,0.5]`，平均 0.90625，八档全部通过。
- `happo_meta`（Meta=1, AMS=0）完整训练到 hard-b，八档攻击最低完成率 `[1,1,1,1,1,0.9375,0.875,0.5]`，平均 0.9140625，八档全部通过。
- `happo_ams_only`（Meta=0, AMS=1）在 light 训练门禁停止：light 三轮攻击均未通过（约 0.25、clean 失败、0.0625），因此没有安全晋级 checkpoint；其 final_ladder 使用 BC 作为最后安全父模型，八档 assessment 最低完成率 `[0.1875,0,0.125,0,0,0.0625,0.0625,0]`，不能当作“训练完成的 AMS-only 策略”结论。
- 既有 `happo_meta_ams`（Meta=1, AMS=1）同协议结果为八档最低完成率 `[1,1,1,1,1,1,1,0.75]`，平均 0.96875，hard-b 75%。因此当前 2×2 初步现象是 full 组合优于 base/meta-only，而 standalone AMS 在本课程中未能形成可晋级策略；这提示存在 Meta×AMS 协同或 AMS-only 稳定性问题，但在修复 AMS-only 训练门禁前不能把交互效应写成最终因果结论。
- v3 产物为 27 个新 assessment cells；根目录实际统计约 52 MP4、214 PNG、564 CSV、44 PT，热图和 `six_algorithm_attack_matrix.csv` 已生成：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3`。
- 下一步应先诊断 AMS-only 在 light 的 clean/攻击失败（检查 AMS critic/advantage 权重、checkpoint AMS state、retention 与 shared actor 的交互），再决定是否以较保守 AMS 权重/暖启动重跑该单元；在此之前只报告“初步正交结果”，不宣称 AMS 单独有效或无效。

### 2026-10-11T23:19:32+0800 组合协同作为论文创新点的表述边界

- 当前结果支持将“Meta reward shaping 与 AMS action-manifold smoothing 的协同组合”作为创新点：Meta 负责任务/车队层面的风险敏感奖励与策略方向塑形，AMS 负责动作邻域价值平滑与局部控制更新，二者在五车空间变道和混合攻击下形成互补。
- 证据应写成初步协同现象而非无条件因果定论：shared HAPPO base hard-b=50%、Meta-only=50%、full Meta+AMS=75%；AMS-only 在 light 门禁失败且没有晋级 checkpoint。这个结果说明组合最强，但也提示 AMS 的收益依赖 Meta 的稳定策略引导，AMS-only 的独立机制仍需修复/复验。
- 推荐论文创新表述：提出一种 Meta-guided Action-Manifold Smoothing（Meta-AMS）协同控制框架，将高层奖励/风险塑形与低层动作流形平滑统一到同一车队策略更新中；通过正交消融验证二者不是简单叠加，而是在强攻击下表现出互补的鲁棒性增益。
- 推荐结果表述：在统一 Ackermann 五车、同一路径和同一攻击阶梯下，Meta-AMS 在 hard-b 达到 75% 最低整车完成率，优于 shared HAPPO base 和 Meta-only 的 50%，且碰撞/越界/道路终止保持为 0；同时明确 AMS-only 仍处于失败诊断阶段。

### 2026-10-11T23:24:13+0800 独立训练随机种子需求确认

- 当前 2×2 结果每个分支只有一条训练随机链；已有的 3 个攻击种子（3407、9917、17011）属于评估随机性，不等同于独立训练种子。因此若要支撑论文中的“Meta-AMS 协同优势/最优”强结论，必须补充独立训练随机种子。
- 最小可接受方案：四个正交单元（base、Meta-only、AMS-only、Meta+AMS）各使用 3 个独立训练种子，并保持同一 BC 初始化、物理/路径/课程/学习率和三攻击评估种子；理想方案为 5 个训练种子。不能只给新缺失单元补种子，否则四格比较的方差不对称。
- 每个训练种子都应先经过 clean 门禁，再逐级训练到 hard-b；对每个最终安全 checkpoint 评估同一三攻击种子，并报告 hard-b 最低整车完成率、八档攻击平均完成率、clean 完成率和安全违规率的均值±标准差/95% CI。以训练种子为统计样本做配对比较或 bootstrap，而不是把攻击环境数当作训练重复数。
- 当前单种子结果可作为 pilot：full Meta+AMS=75%、base/Meta-only=50%，但 AMS-only 尚未形成晋级模型。因此下一步顺序应是先诊断/修复 AMS-only 的 light 稳定性，再启动四格多训练种子复验；在此之前论文中使用“初步协同证据”表述。

### 2026-10-12T00:24:00+0800 AMS-only light 失败诊断与四格多训练种子复验启动

- 本轮先完整读取本日志，并按磁盘状态检查了 v3 正交产物、checkpoint state、训练指标、clean/attack 门禁、评估汇总、当前进程和服务。当前没有 AMS 项目的 train/eval 进程；外部 `/home/cnc/SSD_1T/xzw/IsaacLab-cagan-redesign` 的 CPU smoke eval PID=1807497 仍在运行，复验启动器没有终止它，而是处于 `waiting_for_exclusive_gpu`。
- AMS-only v3 light 三轮原始门禁：`light_r1` clean=1.0、攻击 completion=[0.25,0.3125,0.5]、collision/road_any/road_termination 全为0；`light_r2` clean completion=0；`light_r3` clean=1.0、攻击 completion=[0.0625,0.1875,0.3125]、安全项全为0。失败是严格整车完成/dwell 门禁失败，不是碰撞或越界。
- 训练数值本身没有 NaN/Inf：以 r1 为例 AMS q_loss 约 2.58→0.057、q_grad_norm 37.65→11.56、AMS advantage weight 0→0.02；BC/KL/critic 均有限。评估上 AMS-only 的纵向 gap_error_abs_mean 约 0.224–0.249、min_pair_gap 约 1.36–1.40，而 shared base/meta 分别约 0.122–0.127/1.53–1.69 与 0.128–0.142/1.53–1.58；横向、航向、collision、road boundary 均正常。故失败机制是 AMS-only 破坏纵向车队间距，导致五车不能稳定保持 completion dwell。
- 已确认一个会使原 2×2 结论失效的配置错误：正交定义和验证命令都要求 `happo_share_actor=true`，但旧版 `scripts/tools/run_medium_algorithm_comparison.sh` 的 `happo_ams_only` 训练分支写死为 `false`；旧 AMS-only 三个 model state 的 `runner.cfg.share_actor` 实际均为 `False`，而 base/meta 为 `True`。因此旧 AMS-only 结果混入了“独立 actor 架构差异”，不能作为 standalone AMS 因果结论，也不能和另外三格直接比较。
- 修复内容：将 `run_medium_algorithm_comparison.sh` 的 AMS-only train/eval 分支统一改为 `happo_share_actor=true`；在 `run_ackermann_verified_followup.py` 增加 `--seed-index {0,1,2}`，训练链明确映射 seed=42/3407/777；新增 `scripts/tools/run_ackermann_orthogonal_multiseed.sh`，串行执行四格（base shared、Meta-only、AMS-only、Meta+AMS）×三条独立训练链，每条从同一 BC checkpoint 开始、rounds=1、逐级 clean→light→…→hard-b，评估仍使用固定三攻击 seeds。这样不会把三种评估 seed 冒充训练重复。
- 修复验证：`python3 -m py_compile scripts/tools/run_ackermann_verified_followup.py`、`bash -n scripts/tools/run_medium_algorithm_comparison.sh scripts/tools/run_ackermann_orthogonal_multiseed.sh`、`python3 scripts/tools/test_ackermann_followup.py`（11/11 OK）、`git diff --check` 均通过；三条新复验 manifest 的 preflight 均通过，路径分别为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/followup_manifest.json`、`..._s1/followup_manifest.json`、`..._s2/followup_manifest.json`。
- 已启动命令：`setsid nohup env OUTPUT_BASE=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed LOG=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed.log /home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/run_ackermann_orthogonal_multiseed.sh`。launcher PID=1808341，当前 seed s0 的四格链在 `..._multiseed_s0/followup_status.json` 显示等待外部 PID=1807497；s1/s2 已完成 preflight，尚未开始仿真。
- 当前复验产物根目录：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0`、`..._s1`、`..._s2`；运行总日志：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed.log`。下一步是等待外部 smoke eval 退出后先完成 s0 的 clean/四格训练，再依次完成 s1/s2；全部三条链完成后才计算四格 hard-b/八档 completion 的均值±标准差/CI，并重新判断 Meta×AMS 协同，当前不沿用旧 AMS-only 结果作结论。

### 2026-10-12T00:22:45 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_test_s0/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_test_s0/followup_status.json`。

### 2026-10-12T00:22:45 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[1807497]，待运行命令 `/home/cnc/Downloads/ENTER/bin/python3`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_test_s0/stages/happo_base_shared/clean_seed/validation/clean_env42_validator.log`。未终止任何外部进程。

### 2026-10-12T00:23:19 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/followup_status.json`。

### 2026-10-12T00:23:19 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[1807497]，待运行命令 `/home/cnc/Downloads/ENTER/bin/python3`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/clean_seed/validation/clean_env42_validator.log`。未终止任何外部进程。

### 2026-10-12T00:25:49 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-12T00:32:47 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[1810986]，待运行命令 `bash`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/clean_seed/evidence/evidence_export.log`。未终止任何外部进程。

### 2026-10-12T02:12:02 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-12T02:12:02 AMS后续实验

- 后续服务中止：RuntimeError: Another Isaac workload is active: 1816447 /bin/bash /home/cnc/SSD_1T/xzw/isaac-sim/python.sh /home/cnc/SSD_1T/xzw/IsaacLab-cagan-redesign/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Marl-Platoon-HAPPO-v0 --num_envs 64 --headless --seed 141127 --run_name e1_full_teacher_off_s141127_20261012_s05_hard_b --max_iterations 400 --kit_args=--/rtx/verifyDriverVersion/enabled=false --/app/viewport/grid/enabled=false --/app/viewport/defaults/guide/grid/visible=false --device cuda:0 --resume --load_run 2026-10-12_01-41-53_e1_full_teacher_off_s141127_20261012_s04_hard_a --checkpoint model_final.pt env.algorithm.num_vehicles=5 env.algorithm.num_policy_agents=4 env.algorithm.enable_teacher=true env.algorithm.enable_shield=true env.algorithm.happo_action_clip=0.4 env.algorithm.policy_action_dim=1 env.algorithm.leader_reference_feedforward_gain=1.1 env.algorithm.leader_reference_feedback_gain=0.0 env.algorithm.leader_reference_action_clip=0.8 env.algorithm.leader_reference_soft_start_steps=80 env.algorithm.attack_curriculum_warmup_updates=0 env.algorithm.attack_curriculum_start_mode=cagan env.algorithm.attack_update_interval=1 env.algorithm.attack_generator_lr=5.0e-5 env.algorithm.attack_discriminator_lr=5.0e-5 env.algorithm.attack_max_grad_norm=1.0 env.algorithm.attack_objective_ema_tau=0.97 env.algorithm.attack_normalized_objective_clip=2.0 env.algorithm.attack_max_consecutive_drops=3 env.algorithm.attack_decay_start_updates=0 env.algorithm.stability_gate_enabled=true env.algorithm.stability_window_size=50 env.algorithm.stability_evaluation_interval=100 env.algorithm.stability_hold_updates=100 env.algorithm.stability_max_holds_per_stage=3 env.algorithm.stability_max_bad_orientation_rate=0.05 env.algorithm.stability_max_collision_rate=0.001 env.algorithm.stability_max_value_loss=10000.0 env.algorithm.stability_max_critic_grad_norm=10000.0 env.commands.base_velocity.ranges.lin_vel_x=[0.30,0.45] env.safety_shield.d_drop=1.45 env.safety_shield.catchup_action=-0.355 env.safety_shield.lateral_turn_gain=0.32 env.safety_shield.centerline_turn_gain=0.28 env.safety_shield.pair3_lateral_gain_scale=0.85 env.safety_shield.pair4_lateral_gain_scale=0.90 env.actions.joint_vel_1.scale=12.5 env.actions.joint_vel_2.scale=12.5 env.actions.joint_vel_3.scale=12.5 env.actions.joint_vel_4.scale=12.5 env.actions.joint_vel_5.scale=12.5 env.algorithm.enable_attack=true env.algorithm.attack_level=hard env.algorithm.attack_mode=cagan env.algorithm.attack_target_mode=all env.algorithm.attack_seed=241127 env.algorithm.max_fdi_pos=4.0 env.algorithm.max_fdi_acc=1.3 env.algorithm.max_dos_rate=0.18 env.algorithm.attacker_every_student_updates=16 env.algorithm.attack_start_fdi_pos=3.0 env.algorithm.attack_start_fdi_acc=0.9 env.algorithm.attack_start_dos_rate=0.14 env.algorithm.attack_budget_ramp_start_updates=2600 env.algorithm.attack_budget_ramp_end_updates=2800 env.algorithm.stability_stage_id=hard_b env.algorithm.stability_stage_start_update=2600 env.algorithm.stability_stage_end_update=3000 env.algorithm.enable_teacher=false
；状态/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0`。已完成验证保留，未宣告后续矩阵成功。

### 2026-10-12T11:51:44 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/followup_status.json`。

### 2026-10-12T11:51:45 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[1877694]，待运行命令 `bash`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/clean_seed/evidence/evidence_export.log`。未终止任何外部进程。

### 2026-10-12T11:52:15 AMS多训练种子复验续跑

- 本轮实际检查确认外部 `/home/cnc/SSD_1T/xzw/IsaacLab-cagan-redesign` 训练仍占用 `cuda:0`：协调器 PID=1862496，当前 Isaac 训练 PID=1877694，运行 `e1_full_teacher_on_s141127_20261012_s02_easy`；未终止或干扰该外部任务。
- 上一轮 s0 的中断原因已保留在磁盘：`ams_orthogonal_ablation_20261012_multiseed_s0/followup_status.json` 为 `stopped_at_error`，发生在 `happo_base_shared/clean_seed/evidence` 导出阶段；s1/s2 仍为 `preflight_passed`。当前没有新的四格指标，不能提前宣告复验完成。
- 已重新启动串行续跑器，launcher PID=1886606，当前子进程 PID=1886611；命令为 `setsid nohup env OUTPUT_BASE=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed LOG=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed.log scripts/tools/run_ackermann_orthogonal_multiseed.sh`。s0 已从既有 BC/验证目录恢复，当前状态 `waiting_for_exclusive_gpu`，待外部 PID=1877694 退出后自动继续；随后按 s0→s1→s2 串行执行。
- 续跑保持独占 GPU 保护、四格同 BC 初始化、训练种子 42/3407/777、固定攻击评估种子和 clean→light→…→hard-b 门禁；已有安全产物会按 manifest 复用，不覆盖为失败候选。运行日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed.log`，状态目录分别为 `..._s0`、`..._s1`、`..._s2`。
- 下一步：外部任务释放 GPU 后核对 s0 首个 clean/evidence 是否完成，再继续四格训练；三条训练种子链全部结束后才汇总 hard-b/八档完成率的均值、标准差和置信区间。

### 2026-10-12T11:55:00 参考论文理论主线复核

- 已从桌面原文 `/home/cnc/Desktop/Security Control of Vehicle Platoons under Hybrid Attacks via Meta-Gradient Reward Shaping.pdf` 提取并复核理论部分。文章的核心不是直接把攻击当作普通噪声，而是指出混合 DoS/FDI 同时污染局部观测、执行命令和由观测计算的环境奖励，导致策略更新目标与真实物理车队目标发生偏离。
- 理论链条为：CA-GAN 按当前车队状态生成受预算约束的条件混合攻击；下层 HATRL 学生在受攻击观测上执行 CTDE 策略更新；训练期间用未被攻击的真实物理状态和实际执行输入计算 spacing/velocity/acceleration、jerk、超速、碰撞及投影代价；上层 Reward Teacher 根据学生更新后的真实物理表现调整 shaping function，使下层更新重新对齐真实安全目标。
- 元梯度分析先用一阶可微 HATRL surrogate `θ̄+ = θ− + α g_in(θ−,ϕ)`，得到 `∇ϕ J_out = (∂θ̄+/∂ϕ)^T ∇θ̄+ J_phy`；实现上用物理分数变化与 teacher 输出的协方差、归一化和 warm-up/ramp 近似该外层方向。动作 projection 只在集中训练时约束危险探索，部署时移除 teacher、projection 和 privileged state。
- 稳定性结论是条件性的 local ultimate boundedness：在局部闭环耗散、攻击/扰动/hold 误差有界、有限训练残差有界的假设下，构造归一化 spacing/velocity/acceleration 误差 Lyapunov 函数，得到误差最终进入由扰动界和闭环残差决定的上界；满足 prescribed margins 时得到车队跟驰误差的最终有界，而不是无条件全局安全证明。
- 与当前 AMS 的关系：原文解决的是“攻击使奖励错位，如何用真实物理外层反馈纠正奖励方向”；当前 Meta-AMS 在此基础上增加动作流形/局部动作价值平滑，理论叙事应保持层次分工：Meta reward shaping 负责真实安全目标的全局方向，AMS 负责受攻击策略更新的局部动作几何稳定性。当前多训练种子复验仍在 GPU 排队，尚未用新数据扩展理论结论。

### 2026-10-12T13:13:18 AMS论文LaTeX工程整理

- 按桌面参考论文的 IEEE 控制类写法，在 /home/cnc/Desktop/AMS 创建了新论文工程。主文件为 ams_paper.tex，参考文献目录为 references/，图片目录为 figures/；另有 README.md 和 DATA_PROVENANCE.md 记录构建方式、数据来源和统计边界。
- 论文内容包含：Ackermann 五车动力学与前轮转向/后轮驱动约束；共享 quintic 空间变道路径；混合 DoS--FDI 攻击模型；Meta Reward Teacher 双层目标和一阶元梯度代理；AMS 双 Q、正交邻域、Ackermann 可行投影、执行动作锚定、warm-up/ramp 和 HAPPO 优势融合；条件局部最终有界性命题；真实轨迹 BC 暖启动、攻击课程、严格整车完成门禁和六算法比较。
- 纳入的当前实验证据来自磁盘实际 CSV/图像：Meta-AMS 单训练链 clean 到 hard-a 全部整车完成，hard-b whole-platoon completion mean=0.75、worst seed=0.75，collision/road violation/road termination 均为0；当前正交 pilot 中 base shared 与 Meta-only hard-b mean=0.5208、worst seed=0.50，早期 AMS-only mean=0.0417、worst seed=0、light 门禁停止。AMS-only 已在正文标为诊断性消融，不作为最终因果结论。
- 图片已从验证产物复制：五车同一路径图、Ackermann 跟踪/前轮转向图、Meta-AMS hard-b 俯视轨迹、hard-b 奖励/误差/安全图、训练奖励曲线、六算法均值和 worst-seed 热图；图片文件均存在且 LaTeX 引用路径通过检查。
- 静态校验通过：引用 key 全部存在、无未定义 ref/eqref、begin/end 环境计数一致、所有 includegraphics 文件存在；主文件497行，参考文献121行。当前主机没有 pdflatex、latexmk 或 tectonic，因此本轮未生成 PDF，仅完成可编译结构校验；README.md 给出 TeX Live/MiKTeX 构建命令。
- GPU 任务未被论文整理打断：多训练种子续跑 launcher PID=1886606、当前 s0 子进程 PID=1886611 仍在运行；论文不使用尚未完成的多训练种子结果。下一步可在有 IEEEtran 的 TeX 环境编译并做版面修订，实验方面继续等待 s0→s1→s2 完成后再更新统计表。

### 2026-10-12T13:06:11 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-12T13:14:49 AMS多训练种子复验状态更新

- 论文整理后重新核对磁盘：外部 cagan 训练已退出，未发现其它 Isaac 训练/评估进程；多训练种子 launcher PID=1886606、s0 子进程 PID=1886611 仍在运行。
- s0 状态已由 waiting_for_exclusive_gpu 进入 rendering，当前阶段为 happo_base_shared 的 clean/off evidence 导出；尚未有新的四格完成指标。论文工程不修改训练目录，也未影响 GPU 任务。

### 2026-10-12T13:51:15 AMS论文交接包与压缩归档

- 按用户要求继续整理了可交给另一台电脑上 Codex/GPT 的 AMS 交接材料，写入 `/home/cnc/Desktop/AMS`：`GPT_HANDOFF.md`（先读说明、当前状态、可信表述边界和后续步骤）、`CODE_LOGIC.md`（Ackermann 执行器、共享空间路径、completion gate、Reward Teacher、AMS 双Q/正交邻域、HAPPO接入和训练脚本的数据流）、`THEORY_WRITING_GUIDE.md`（理论符号、理想元梯度与实现代理的区分、AMS 局部动作几何、条件局部最终有界性命题和消融写法）、`EXPERIMENT_HANDOFF.md`（六算法矩阵、正交 pilot、多训练种子路径、指标和复现命令）、`FILE_MANIFEST.txt`（本地工程与外部代码/日志路径映射）。
- 交接文档明确记录：当前 zip 是论文/图片/参考文献/说明文件包，不包含 Isaac Lab 源码、Isaac Sim、模型权重和大日志；另一台电脑若要重新运行实验，需要另行提供 `/home/cnc/SSD_1T/xzw/IsaacLab-main` 及相应运行环境。文档同时明确禁止把尚未完成的三训练种子复验写成最终统计结论，不把早期 share-actor 不同的 AMS-only 失败当作干净因果消融，不把当前 Reward Teacher 宣称为完整二阶元梯度，不把稳定性表述为无条件全局安全证明。
- 执行静态检查：AMS 主文件、BibTeX、四份交接文档和 manifest 均存在且非空；`ams_paper.tex` 的 7 个 `includegraphics` 全部有对应图片；所有 citation key 在 `references/ams.bib` 中存在；LaTeX begin/end 环境计数平衡。主机仍没有 `pdflatex`、`latexmk` 或 `tectonic`，因此未在本机生成 PDF。
- 已执行打包命令：`cd /home/cnc/Desktop && zip -qr -FS /home/cnc/Desktop/AMS.zip AMS`。归档 `/home/cnc/Desktop/AMS.zip` 已生成，大小约 1.4 MB，共 19 个条目，SHA-256=`d0f15fef87aea16940e6539bd32eb12f275b3fb2e92131acc8f99c7044d62980`；通过 `unzip -l` 核对内容。
- 打包和静态检查未停止或修改 GPU 任务。复核时 launcher PID=1886606、s0 follow-up PID=1886611 仍在运行；s0 当前状态仍为 `rendering`，方法 `happo_base_shared`、级别 `light`，输出目录为 `.../ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/light_r1/evidence`，尚无新的四格最终指标。期间存在该阶段的 Isaac Sim 视频/轨迹导出子进程，未终止。
- 下一步：将 `/home/cnc/Desktop/AMS.zip` 复制到另一台电脑，先阅读 `GPT_HANDOFF.md`、`CODE_LOGIC.md`、`THEORY_WRITING_GUIDE.md`、`EXPERIMENT_HANDOFF.md`，再在有 IEEEtran 的 TeX 环境编译 `ams_paper.tex`；本机继续等待 s0→s1→s2 复验完成后，依据最终 CSV 更新论文统计和图表。

### 2026-10-12T13:52:29 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/light_r1`。

### 2026-10-12T14:28:45 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/easy_0_r1`。

### 2026-10-12T15:04:15 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/easy_a_r1`。

### 2026-10-12T15:39:31 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/easy_b_r1`。

### 2026-10-12T16:15:22 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/med_a_r1`。

### 2026-10-12T16:51:12 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/med_b_r1`。

### 2026-10-12T17:26:39 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/hard_a_r1`。

### 2026-10-12T18:02:19 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_base_shared/hard_b_r1`。

### 2026-10-12T18:16:20 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/off`。

### 2026-10-12T18:42:57 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/light`。

### 2026-10-12T19:09:44 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/easy_0`。

### 2026-10-12T19:36:22 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/easy_a`。

### 2026-10-12T20:02:57 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/easy_b`。

### 2026-10-12T20:30:22 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/med_a`。

### 2026-10-12T20:58:07 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/med_b`。

### 2026-10-12T21:26:07 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/hard_a`。

### 2026-10-12T21:53:55 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_base_shared/hard_b`。

### 2026-10-12T22:45:17 AMS后续实验

- 六算法 `happo_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/light_r1`。

### 2026-10-12T23:22:09 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/easy_0_r1`。

### 2026-10-12T23:59:11 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/easy_a_r1`。

### 2026-10-13T00:36:19 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/easy_b_r1`。

### 2026-10-13T01:13:20 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/med_a_r1`。

### 2026-10-13T01:50:37 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/med_b_r1`。

### 2026-10-13T02:27:59 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/hard_a_r1`。

### 2026-10-13T03:05:06 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta/hard_b_r1`。

### 2026-10-13T03:19:41 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/off`。

### 2026-10-13T03:47:32 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/light`。

### 2026-10-13T04:15:04 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/easy_0`。

### 2026-10-13T04:42:44 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/easy_a`。

### 2026-10-13T05:10:19 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/easy_b`。

### 2026-10-13T05:37:57 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/med_a`。

### 2026-10-13T06:05:29 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/med_b`。

### 2026-10-13T06:33:11 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/hard_a`。

### 2026-10-13T07:01:01 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta/hard_b`。

### 2026-10-13T07:51:10 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/light_r1`。

### 2026-10-13T08:26:34 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/easy_0_r1`。

### 2026-10-13T09:01:11 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/easy_a_r1`。

### 2026-10-13T09:35:47 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/easy_b_r1`。

### 2026-10-13T10:10:23 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/med_a_r1`。

### 2026-10-13T10:44:59 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/med_b_r1`。

### 2026-10-13T11:19:45 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/hard_a_r1`。

### 2026-10-13T11:54:22 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_ams_only/hard_b_r1`。

### 2026-10-13T12:08:08 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/off`。

### 2026-10-13T12:34:13 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/light`。

### 2026-10-13T12:48:52 AMS后续实验

- 发现外部GPU任务，排队而非并发启动；等待PID=[893011]，待运行命令 `/home/cnc/Downloads/ENTER/bin/python3`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/easy_0/validation/attack_env777_seed17011_validator.log`。未终止任何外部进程。

### 2026-10-13T12:58:57 三条论文主线的补充资料与协议事实审计

- 本轮用户询问“物理一致性—价值学习稳定性—能力保持”三点创新是否还缺资料。通过 rg/sed 核对桌面 AMS 文档、ams_paper.tex、参考文献、源代码 ams.py/runner.py/actor.py/teacher.py/attacker.py、verified-followup 启动器、20261003 Meta-AMS hard-b 归档 env.yaml 及 20261012 s0 AMS-only 的归档配置和 validation_gate.json；未修改 Isaac Lab 源码、门禁或训练命令，未启动新实验。
- 发现并修正部分论文/交接协议表述：实际 matrix 为 attack_mode=profile，不是在线训练 CA-GAN；实际 AMS 权重/warm-up/ramp 为 0.02/5/15，不是类默认 0.10/50/100；固定 BC=10、KL=1、batch=1024，训练 clean 环境比例=0.5；retention_curriculum_enabled 与 happo_retention_schedule_enabled 均 false，expert blend=0。因此现有矩阵支持 BC+固定保持+混合攻击课程，不能直接支持渐进专家退出的有效性主张。历史渐进退出若引用，仍须单独绑定当时运行配置和实验产物。
- 查询并核对基础 AMS 一手来源 https://proceedings.mlr.press/v306/lin26a.html 与官方 https://github.com/Lin-Zhihao98/AMS ，添加 references/ams.bib 的 lin2026actionmanifold 和正文引用。应将原创定位为 on-policy 多智能体适配与协同机制，而不是基础 AMS 本身。其余 BibTeX 条目本轮未做全库真实性核查。
- 新增 /home/cnc/Desktop/AMS/SUPPLEMENTAL_EVIDENCE_PLAN.md，按三点贡献列出待补证据：四格独立训练种子统计与机制指标；五车逐车/逐对路径、转角、dwell、视频证据；能力保持/专家退出日程、clean回归曲线与保持机制对照。另列理论边界、代码快照/CSV/配置可移植附件、风险项：双Q最小值与求和次序公式尚需对齐；当前 actor 使用逐车条件局部优势；shield 仍启用，不能把结果写成无shield部署；Taylor/Laplacian说明要处理投影破坏对称性、min不可微和局部方向非正交；真实轨迹是仿真执行轨迹，并非实车试验。
- 已更新 ams_paper.tex、CODE_LOGIC.md、EXPERIMENT_HANDOFF.md、DATA_PROVENANCE.md、GPT_HANDOFF.md、README.md、FILE_MANIFEST.txt 及 references/ams.bib。没有宣称这些文档已完成全部理论/公式/文献审计，也未生成新实验结果。静态检查为15个引用key无缺失、7张图片均存在、环境计数平衡；本机仍无TeX编译验证，静态检查不等于可编译证明。
- s0 最新阶段证据：AMS-only light passed=True、完成率均值/最差=1/1；hard-a passed=True、均值0.958333、最差0.9375；hard-b新候选 passed=False、clean_ok=True、safe=True，均值0.5、最差0.375，未替代hard-a安全模型。base shared hard-b新候选失败(均值0.479167/最差0.4375)，但旧安全模型最终全强度hard-b评估通过(0.625/0.5625)；Meta hard-b新候选失败(0.520833/0.4375)，旧安全模型最终hard-b也失败(0.416667/0.375)。这些分属候选门禁与安全模型事后评估，不能混合成同一结果表。
- 12:57复核 launcher PID=1886606、s0 PID=1886611 仍存活；状态 waiting_for_exclusive_gpu，等待继续 AMS-only final_ladders/easy_0 env777/attack17011。外部 IsaacLab-cagan-redesign 训练 PID=934991 占用GPU，未干扰；s0 Meta-AMS尚未启动，s1/s2仍preflight，独立训练多种子四格尚未完成。
- 更新压缩命令：cd /home/cnc/Desktop 后 zip -qr -FS /home/cnc/Desktop/AMS.zip AMS；unzip -tq 校验无压缩数据错误。最终ZIP约1.4MB，SHA256=fad089d7c71c3a1bb6f90c9eeb1e4fab30dfe06f7af1e0f310109d7bcfa9de44。ZIP仍主要为论文/图/文献/说明，不包含可移植源码快照、原始CSV和权重。下一步优先补可移植证据与主张-公式-代码-数据对照，再依据完成的四格复验和保持机制实验更新论文，不预先断言协同收益或保证不遗忘。

### 2026-10-13T13:40:00 AMS综合ZIP扩充归档

- 按用户要求将当前 AMS 资料从“论文文档包”扩充为单一综合交接包，仍只读复制、不修改 Isaac Lab 运行源码、不终止 GPU 任务。新增 `source_snapshot/`（Ackermann、trajectory、AMS、HAPPO actor/runner、teacher、router、platoon config/task teacher 等相关源码）、`scripts_snapshot/`（verified follow-up、正交多种子、六算法比较、验证与测试脚本）、`experiment_snapshots/`（主矩阵、正交 pilot、s0/s1/s2 多种子目录中的 CSV/JSON/YAML/runs.txt，过滤重复 trajectory/eval_steps 与模型权重）。
- 新增 `evidence_snapshot/`：仿真执行专家轨迹、Ackermann 图片和视频、BC checkpoint、Meta-AMS hard-b final-ladder 的验证摘要/门禁/三种评估轨迹/俯视图/奖励安全图/视频；新增 `reference_material/`：桌面 MGRS 参考论文 PDF 和从 PMLR 官方来源下载的 AMS ICML 2026 PDF；新增 `project_notes/ams_debug_notes.md`。新增/更新 `PACKAGE_SCOPE.md`、`FILE_MANIFEST.txt` 和 README，明确综合包包含项、排除项和可复现性警告。
- 综合 AMS 目录打包前约 162 MB；其中实验快照约 63 MB、证据约 79 MB、参考论文约 18 MB、源码/脚本/文档约 1 MB。完整 Isaac Lab/Isaac Sim、USD/assets、Python 环境、绝大多数阶段模型权重和多 GB 原始日志仍不放入 zip，防止交接包失控膨胀；它们的外部绝对路径保留在 FILE_MANIFEST/DATA_PROVENANCE。
- 打包过程中复核：主矩阵 CSV 存在，s0 compact snapshot 含 44 个 validation_gate.json，五类关键输出均存在；未读取或复制正在写入的重复大轨迹，未改变外部文件。新的 `AMS.zip` 将在把本条记录同步到 `project_notes/` 后生成并用 `unzip -tq` 验证。
- GPU 状态保持不变：launcher PID=1886606、s0 follow-up PID=1886611 仍在运行；s0 等待独占 GPU 后继续 AMS-only final ladder/easy_0 env777/attack17011，外部 PID=934991 的 IsaacLab-cagan-redesign CAGAN 训练仍占用 GPU；未终止任何冲突进程。s0 的 Meta-AMS 尚未启动，s1/s2 仍 preflight，不能把综合包中的中间 JSON 当最终三种子统计。
- 下一步：同步最新 notes 到 `AMS/project_notes/ams_debug_notes.md`，执行 `cd /home/cnc/Desktop && zip -qr -FS /home/cnc/Desktop/AMS.zip AMS`，执行 `unzip -tq` 与 SHA-256，最后只向用户报告综合包路径和当前运行未完成状态。

### 2026-10-13T13:42:30 AMS综合ZIP最终校验

- 已将本条综合归档记录同步到 `/home/cnc/Desktop/AMS/project_notes/ams_debug_notes.md`，执行 `cd /home/cnc/Desktop && zip -qr -FS /home/cnc/Desktop/AMS.zip AMS`，并执行 `unzip -tq /home/cnc/Desktop/AMS.zip`；压缩数据校验通过。
- 最终归档 `/home/cnc/Desktop/AMS.zip` 大小约 80 MB，共 5677 个 zip 条目；最终 SHA-256=`e1b7c8ed52e3689249969a5802698d370821fe6ac74b96b72c1143bb5258c59b`。归档包括源码/脚本快照、实验摘要与配置、参考 PDF、BC checkpoint、Ackermann 证据、Meta-AMS hard-b 全部验证轨迹/视频/图像和论文文档。
- 归档检查未发现实验快照中的模型权重和重复 eval_steps 文件；Meta-AMS hard-b 证据目录按要求保留证明用途所需的 trajectory/eval/video。完整 Isaac Lab/Isaac Sim 和多 GB 原始日志仍按 PACKAGE_SCOPE 明确排除。GPU 任务未被停止；launcher/s0 仍运行且等待外部 GPU 冲突释放。

### 2026-10-13T14:26:06 AMS后续实验

- 外部Isaac/诊断协调进程已退出，继续串行AMS任务。

### 2026-10-13T14:40:05 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/easy_0`。

### 2026-10-13T15:06:19 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/easy_a`。

### 2026-10-13T15:32:15 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/easy_b`。

### 2026-10-13T15:58:14 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/med_a`。

### 2026-10-13T16:24:31 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/med_b`。

### 2026-10-13T16:51:46 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/hard_a`。

### 2026-10-13T17:19:01 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_ams_only/hard_b`。

### 2026-10-13T18:07:43 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/light_r1`。

### 2026-10-13T18:42:33 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-13T19:17:27 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-13T19:52:26 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-13T20:27:32 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/med_a_r1`。

### 2026-10-13T21:02:25 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/med_b_r1`。

### 2026-10-13T21:39:03 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-13T22:15:40 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-13T22:30:00 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/off`。

### 2026-10-13T22:57:31 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/light`。

### 2026-10-13T23:25:05 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/easy_0`。

### 2026-10-13T23:52:15 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/easy_a`。

### 2026-10-14T00:18:18 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/easy_b`。

### 2026-10-14T00:44:29 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/med_a`。

### 2026-10-14T01:10:48 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/med_b`。

### 2026-10-14T01:37:05 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/hard_a`。

### 2026-10-14T02:03:14 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/hard_b`。

### 2026-10-14T02:03:31 AMS后续实验

- 后续实验mode=matrix已完成，产物 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0`。

### 2026-10-14T02:03:32 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_status.json`。

### 2026-10-14T02:53:01 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/light_r1`。

### 2026-10-14T03:31:44 AMS后续实验

- 后续服务中止：RuntimeError: Command exit=1; log=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_0_r1/validation/attack_env42_seed3407_validator.log；状态/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1`。已完成验证保留，未宣告后续矩阵成功。

### 2026-10-14T16:24:00 当前进度只读核验（用户询问进度并要求继续核查）

- 以磁盘为准检查 ps、nvidia-smi、systemd 系统/用户 running services、s0/s1/s2 followup_status.json、method_outcomes.json、matrix CSV、validation_gate、evidence/ladder_complete 标记、s1 validator/evaluation 日志和 AMS.zip。未修改训练代码、未终止任何进程、未重启训练、未修改/重打包 ZIP；本轮仅追加工作记录。
- s0 于 2026-10-14 02:03:31 完成，产物根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0`；`six_algorithm_attack_matrix.csv` 有 4 方法×9 强度=36 行，4 个 ladder_complete、36 个 evidence_complete、36 个 final-ladder MP4 均已落盘。36 行 collision_rate、road_boundary_violation_rate、road_termination_rate_max 的最大值均0。
- s0 最终保留模型 hard-b 全强度评估（均值/最差评估种子）：base shared=0.625/0.5625；Meta-only=0.4166667/0.375；AMS-only=0.5625/0.50；Meta-AMS=0.75/0.75。上述是同一训练种子下的评估结果，不是三独立训练种子均值。只有 Meta-AMS 的 hard-b 新训练候选通过并被晋级；另三格 hard-b 训练候选被拒绝，最终评估使用此前安全模型，base/AMS-only 安全模型在事后 hard-b 评估通过。不能把“事后评估通过”写为“hard-b 新候选晋级”。
- s1 在 10月14日03:31:44 stopped_at_error：happo_base_shared/easy_0_r1/validation/attack_env42_seed3407 评估等待1800秒超时；日志显示 subprocess.TimeoutExpired，不是完成率或安全门禁失败。evaluation.log 只有9833字节，最后更新时间03:02:05，停在环境构建/物理材质绑定附近；目录只有 request.json 和 evaluation.log，无 verified.json，底层卡住原因尚未确定。easy_0_r1 的 training_complete.json 和训练 model_final.pt 已存在；clean_env42/verified.json 已存在，可以在处理遗留进程/超时清理后优先续评估，无需先重训。此前 s1 light_r1 已过门禁，攻击均值0.9791667、最差0.9375，clean/safe均通过。
- 原 launcher PID1886606 和协调器 PID1886611 已退出，不能再报告训练正在正常推进。残留 s1 Isaac 评估 PID1906515（父PID2770 user systemd）仍存活约13小时20分钟，占GPU约1886MiB；GPU瞬时利用率约7%，同时存在ToDesk显存约384MiB。系统和用户 running service 列表未见在运行的 AMS/车队/CAGAN 续跑服务；外部 CAGAN 训练本轮未发现。旧 failed 服务没有擅自启动。s2 仍preflight_passed，训练未开始。
- 桌面 `/home/cnc/Desktop/AMS.zip` 保持10月13日13:41快照，83383424 bytes（约80MB），`unzip -tq`通过，SHA256=e1b7c8ed52e3689249969a5802698d370821fe6ac74b96b72c1143bb5258c59b。它不包含今天才完成的 s0 完整矩阵与新增视频；本轮状态问答未更新归档。
- 下一步建议：先诊断并处理 PID1906515 的遗留评估与进程组超时清理逻辑，保留失败日志，再复用s1已完成的训练/clean验证继续easy-0攻击验证，随后续跑s1/s2；结束后才做多训练种子统计和ZIP更新。当前结果支持Meta-AMS在s0四格中表现最好，不足以支持跨训练种子的显著优越性结论。

### 2026-10-15T00:20:52 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_status.json`。

### 2026-10-15T00:20:56 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/light_r1`。

### 2026-10-15T00:22:30 接管清理、s1续评估与补充训练队列

- 以磁盘实际状态复核：旧 s1 残留评估 PID `1906515` 的完整命令确实指向 `happo_base_shared/easy_0_r1/validation/attack_env42_seed3407`，进程组为 `PGID/SID=1906510`；AMS/systemd 无运行服务，旧 launcher/coordinator 已退出，GPU 无其它 Isaac 训练/评估。仅保留未触碰的 ToDesk 图形进程。
- 已在最终身份复核后终止精确残留进程组 `-1906510`（先SIGTERM等待20秒，仍存活再SIGKILL），未删除任何评估目录；原始超时目录、`request.json`、`evaluation.log`均保留。清理后 `nvidia-smi --query-compute-apps` 无 Isaac 进程。
- 修复 `scripts/tools/ackermann_validation.py` 与 `scripts/tools/run_ackermann_verified_followup.py` 的超时清理：使用 `os.getpgid(process.pid)`/`os.getpgid(child.pid)` 后再杀进程组，并容忍进程已退出；避免 wrapper PID 与 Kit 子进程组不同导致 GPU 任务遗留。两文件已 `py_compile`，shell/差异检查通过。
- 增加仅限上述“启动/超时基础设施文件”的兼容复用：既有 `training_complete.json`、`verified.json` 在物理/策略/评估源不变时继续有效，不因清理修复重新训练或重复已完成仿真；其它源变化仍严格拒绝复用。旧 s1 manifest 已通过 `--resume-coordinator-update` 迁移并保留为 `followup_manifest.coordinator_previous_*.json`。
- s1 恢复预检命令（已通过）：`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/run_ackermann_verified_followup.py --output logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1 --reference logs/rsl_rl/platoon_happo/ams_bridge_refined_20260922/stages/hard_b_r2 --bc-checkpoint logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt --methods happo_base_shared happo_meta happo_ams_only happo_meta_ams --rounds 1 --seed-index 1 --mode matrix --resume-coordinator-update --preflight-only`；注意用户给出的 `ams_gap_stability_20260907/.../hard_bridge_3_r1` 当前只有训练/门禁产物、没有 `validation/clean_env42/verified.json`，不能直接作为该 follow-up runner 的 hard-b reference，因此未强行替换既有 s1 协议。
- 已启动串行恢复队列 PID `1959276`（session/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_resume_queue.log`），脚本 `/home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/run_ackermann_resume_and_supplement.sh`：先恢复 s1，再恢复尚未开始的 s2；每个 seed 使用 `--resume-coordinator-update`，已完成训练/clean/light evidence 按签名复用。当前 s1 状态为 evaluating，正在只补跑缺失的 `happo_base_shared/easy_0_r1/attack_env42_seed3407`；训练进程尚未重启，GPU 当前仅该评估约2.8GiB。
- 已增加补充训练入口 `/home/cnc/SSD_1T/xzw/IsaacLab-main/scripts/tools/run_ackermann_resume_and_supplement.sh`，并扩展 `run_ackermann_retention_recovery.sh` 的可选渐进能力保持课程（默认关闭，队列在 s1/s2 完成后显式开启）：`happo_retention_schedule_enabled=true`、BC/KL 系数从 `1.0/0.10` 渐降至 `0.05/0.01`，专家控制比例 `1→0`，warmup=10、decay=80 updates。补充训练结果将写入 `logs/ackermann_retention_recovery/orthogonal_ablation_retention_schedule_20261015`，不与当前 s1/s2 并发。
- 当前指标未新增门禁结论：沿用已确认的 s1 `light_r1` clean通过、攻击最低完成率 `0.9375`；easy-0 缺失评估正在恢复，尚未形成新 gate。s0 四格 hard-b 既有结果仍为 Meta-AMS `0.75/0.75`（均值/最差，单训练种子），其它三格分别 `0.625/0.5625`、`0.4167/0.375`、`0.5625/0.50`，不能冒充三种子统计。
- 下一步：保持队列运行，监控 s1 `followup_status.json`、`verified.json`/`validation_gate.json` 和 GPU 进程；s1 完成后自动进入 s2，二者完成后自动启动渐进能力保持补充训练。新结果通过 clean、完成率、碰撞、越界门禁后再更新桌面 AMS 资料包/ZIP。

### 2026-10-15T00:23:20 接管队列实时核验

- 队列 PID `1959276` 与 s1 follow-up PID `1959283` 仍在运行；当前唯一 Isaac 子任务是恢复的 `ackermann_validation.py`/Kit 评估（PGID `1959352`，目标仍为 `easy_0/attack_env42_seed3407`），无训练并发、无第二个评估。该评估已持续约2分16秒，`trajectory_env0.csv` 正在增长（约2.95MB），`evaluation.log`约21KB，未超时。
- s1 `followup_status.json` 仍为 `evaluating`，暂无新 gate/完成率结论；旧 clean/light 训练与验证继续按签名复用。队列日志仍记录 `s1 attempt=1`，未触发重试或补充训练启动。

### 2026-10-15T00:27:20 当前任务目的说明核验

- 当前 GPU 上运行的不是新训练，而是恢复 s1 的严格攻击评估：验证已有 `happo_base_shared/easy_0_r1` 模型在 easy-0 混合攻击下能否通过三组环境/攻击种子；这是补齐中断证据、决定是否允许继续更高攻击强度课程的门禁，不会改变模型参数。
- 本轮整体队列的研究目的分三层：1）s1/s2 为正交四格方法（base、Meta、AMS-only、Meta+AMS）补充独立训练种子，检验组合效果是否稳定而非单种子偶然；2）每个阶段都先检查 clean、完成率、碰撞和越界，避免攻击适应破坏基础协同变道能力；3）两组多种子矩阵结束后运行渐进能力保持补充训练，验证真实轨迹暖启动、BC/KL 约束和专家控制 `1→0` 退出是否能抑制攻击适应中的技能退化。
- 因此当前阶段不产生“算法更优”的新结论；只有 s1/s2 的各阶段 gate 和补充训练的 clean/攻击对照完成后，才用于论文中的多种子统计和能力保持消融。

### 2026-10-15T01:20:55 s1门禁兼容修复后恢复

- 00:39 的停止原因已定位为门禁比较逻辑遗漏：clean 旧证据与新补跑 attack 证据只在 `ackermann_validation.py` 清理实现哈希上不同。已在 `ackermann_validation.py` 的 gate 中加入与 follow-up runner 一致的基础设施源兼容比较；物理、策略、评估和渲染源仍严格比较。已通过 `py_compile`/shell 语法检查。
- 原停止时三个 easy-0 attack `verified.json` 均已存在，未删除、未重跑；修复后 s1 队列于 `2026-10-15 01:20:45` 原地恢复，当前已通过 easy-0 验证并进入 `happo_base_shared/easy_0_r1/evidence` 渲染阶段。队列 PID=`1976041`，follow-up PID=`1976048`；s2 尚未启动，补充训练尚未启动。
- 当前下一步是完成 easy-0 的视频/轨迹证据导出，再继续 s1 其余攻击课程和三种正交方法；完成 s1 后自动恢复 s2，最后才启动渐进能力保持补充训练。没有重复已完成训练和已完成 easy-0 仿真。

### 2026-10-15T视频产物核验

- 已定位主实验 Meta-AMS（s0、固定最终安全模型、hard-b）变道视频：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0/final_ladders/happo_meta_ams/hard_b/video/rl-video-step-0.mp4`。
- 文件大小约 `2,609,633` bytes；ffprobe核验为 `1280×720`、`70.000 s`、`3500`帧，属于已通过 hard-b 的 Meta-AMS 固定模型证据，可用于直观查看五车完整变道。

### 2026-10-15T00:39:42 AMS后续实验

- 后续服务中止：ValueError: source code changed between clean and attack evaluations；状态/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1`。已完成验证保留，未宣告后续矩阵成功。

### 2026-10-15T01:20:46 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_status.json`。

### 2026-10-15T01:20:50 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/light_r1`。

### 2026-10-15T01:28:45 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_0_r1`。

### 2026-10-15T02:04:32 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_a_r1`。

### 2026-10-15T02:40:10 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_b_r1`。

### 2026-10-15T03:15:53 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/med_a_r1`。

### 2026-10-15T03:51:25 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/med_b_r1`。

### 2026-10-15T04:26:56 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/hard_a_r1`。

### 2026-10-15T05:02:30 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/hard_b_r1`。

### 2026-10-15T05:16:28 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/off`。

### 2026-10-15T05:43:06 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/light`。

### 2026-10-15T06:09:42 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/easy_0`。

### 2026-10-15T06:36:22 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/easy_a`。

### 2026-10-15T07:02:55 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/easy_b`。

### 2026-10-15T07:29:38 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/med_a`。

### 2026-10-15T07:56:25 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/med_b`。

### 2026-10-15T08:23:13 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/hard_a`。

### 2026-10-15T08:49:59 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/hard_b`。

### 2026-10-15T09:39:37 AMS后续实验

- 六算法 `happo_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/light_r1`。

### 2026-10-15T10:15:10 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/easy_0_r1`。

### 2026-10-15T10:18:20 AMS后续实验

- 后续服务中止：RuntimeError: Command exit=1; log=/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/easy_a_r1/validation/clean_env42_validator.log；状态/日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1`。已完成验证保留，未宣告后续矩阵成功。

### 2026-10-15T21:22:25+08:00 磁盘空间只读检查

- 系统根分区 `/dev/nvme1n1p2` 总计约 468G，已用约 444G（100%），仅剩约 234MB；`/home/cnc/SSD_1T` 为独立分区，尚有约 456G 可用，不是当前满盘来源。
- 最大异常来源为 `/tmp/IsaacLab`，约 319G，包含 14,906 个 `usd_*` 临时目录和约 104,340 个文件，最早来自 2026-07-31，最近到 2026-10-15 10:18。未发现进程 cwd 或 fd 指向该目录，也未发现正在运行的 Isaac/CARLA/AMS 进程；该目录列为最高优先级清理候选，但本次未删除。
- 其他主要占用和候选：`/home/cnc/Downloads` 约 59G（其中 `CARLA` 21G、`CarlaUE4` 17G、`ENTER` 8.5G、`HDMaps` 1.6G，存在疑似重复副本，需确认后再删）；`/home/cnc/.cache/pip` 约 12G（缓存，可重建）；`/home/cnc/.codex/packages` 约 5.1G（当前 Codex 依赖，暂不建议直接删）；`/var/log/journal` 约 4.1G（可按保留周期压缩/清理）；`/home/cnc/.julia` 约 5.6G（编译缓存约 2.1G，运行时和包不要盲删）。
- 发现 2G `/swapfile`，以及若干大型 CARLA/OpenCDA 压缩包和调试符号文件；它们不是误占用，是否删除需结合后续复现实验需求确认。
- 本轮只读检查，没有执行删除、清空或修改实验文件；待用户确认后，优先处理 `/tmp/IsaacLab`，再处理 pip 缓存和已确认的 CARLA 重复文件。

### 2026-10-15T21:49:05+08:00 缓存清理与实验路径迁移

- 用户确认先清除缓存并将所有实验保存路径迁移到1T盘；清理前确认无 Isaac/CARLA/AMS 训练、评估或仿真进程，也没有进程 cwd/fd 使用 `/tmp/IsaacLab`。
- 已清理 `/tmp/IsaacLab` 约319G的14,906个临时USD目录、`/home/cnc/.cache`约16G可重建缓存，以及遗留的 `/tmp/isaaclab` 约113M数据集缓存；未删除项目 `logs`、checkpoint、视频、轨迹或模型。系统盘从100%降到约25%，剩余约334G。
- 新增共享路径策略 `IsaacLab-main/scripts/tools/experiment_env.sh`：实验根目录默认为 `/home/cnc/SSD_1T/xzw/IsaacLab-main`；运行时临时目录为 `/home/cnc/SSD_1T/xzw/ams_runtime/{IsaacLab,tmp}`；XDG、pip、CUDA/GL缓存均在 `/home/cnc/SSD_1T/xzw/ams_runtime/cache`。
- 修改 IsaacLab 资产转换器，生成的默认USD不再写入 `/tmp/IsaacLab`；地形缓存默认写入1T runtime/cache；Recorder数据集、Ray调参、robomimic评估及预训练checkpoint查找路径均改为1T项目 logs。
- RSL-RL、skrl、RL-Games、SB3、HARL、benchmark、sim2sim 和AMS验证脚本的默认日志/视频/轨迹输出均改为由 `ISAACLAB_EXPERIMENT_ROOT` 解析，默认落在 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs`；AMS主启动器会强制继承1T runtime临时/缓存目录。
- 回归检查通过：修改脚本 `bash -n`、Python `py_compile`、`git diff --check`；源码中不再有实验默认路径 `/tmp/IsaacLab`、`/tmp/isaaclab`、`storage_path=/tmp` 或相对 `logs` 默认。当前 `/tmp`约53M、`~/.cache`约1.3M，根分区无遗留IsaacLab临时缓存。

### 2026-10-15T21:50:17 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/followup_status.json`。

### 2026-10-15T21:52:00 AMS实验继续核验

- s1 之前已完成并通过：`happo_base_shared` 从 light 到 hard-b 全部阶段，hard-b 最低完成率 `0.50`、均值 `0.5208333`；`happo_meta` 已通过 light/easy-0，easy-a 训练完成但 clean 评估因 `No space left on device` 中止，不是策略门禁失败。
- 磁盘路径迁移/缓存清理已落盘：`/tmp/IsaacLab` 临时USD目录已清除，当前 `/` 可用约334GB、`/home/cnc/SSD_1T` 可用约456GB；正式 logs/checkpoint/视频/轨迹未删除。队列脚本增加20GB空间预检和磁盘写满可重试。
- 由于21:35同步落盘的AMS协议代码更新，resume兼容集合已显式登记评估/播放/训练路径、Ackermann参数、legacy router和启动脚本；其它未登记物理/策略源仍严格拒绝。s1 manifest预检已通过，旧manifest保留。
- 已重新启动串行队列 PID `2137687`、follow-up PID `2137697`；当前在导出 `happo_base_shared/clean_seed` 证据，Isaac play PID `2137715`，约6.1GiB GPU，占用正常。随后会复用已完成阶段并从 `happo_meta/easy_a_r1` clean 评估恢复，不重复已完成训练。
- 当前无新增算法结论；下一步为完成证据导出、恢复 easy-a clean，再继续 s1 的 happo_meta/AMS-only/Meta-AMS 四格矩阵，随后自动进入 s2 和能力保持补充训练。

### 2026-10-15T22:06:17 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/light_r1`。

### 2026-10-15T22:14:11 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_0_r1`。

### 2026-10-15T22:22:04 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_a_r1`。

### 2026-10-15T22:29:57 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/easy_b_r1`。

### 2026-10-15T22:37:48 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/med_a_r1`。

### 2026-10-15T22:45:42 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/med_b_r1`。

### 2026-10-15T22:53:36 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/hard_a_r1`。

### 2026-10-15T23:01:38 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_base_shared/hard_b_r1`。

### 2026-10-15T23:01:38 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/off`。

### 2026-10-15T23:01:39 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/light`。

### 2026-10-15T23:01:41 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/easy_0`。

### 2026-10-15T23:01:42 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/easy_a`。

### 2026-10-15T23:01:43 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/easy_b`。

### 2026-10-15T23:01:44 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/med_a`。

### 2026-10-15T23:01:45 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/med_b`。

### 2026-10-15T23:01:46 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/hard_a`。

### 2026-10-15T23:01:48 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_base_shared/hard_b`。

### 2026-10-15T23:17:31 AMS后续实验

- 六算法 `happo_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/light_r1`。

### 2026-10-15T23:25:27 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/easy_0_r1`。

### 2026-10-15T23:58:10 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/easy_a_r1`。

### 2026-10-16T00:36:52 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/easy_b_r1`。

### 2026-10-16T01:12:21 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/med_a_r1`。

### 2026-10-16T01:47:53 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/med_b_r1`。

### 2026-10-16T02:23:31 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/hard_a_r1`。

### 2026-10-16T02:59:08 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta/hard_b_r1`。

### 2026-10-16T03:13:15 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/off`。

### 2026-10-16T03:39:45 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/light`。

### 2026-10-16T04:06:31 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/easy_0`。

### 2026-10-16T04:37:52 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/easy_a`。

### 2026-10-16T05:09:19 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/easy_b`。

### 2026-10-16T05:40:49 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/med_a`。

### 2026-10-16T06:12:00 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/med_b`。

### 2026-10-16T06:43:26 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/hard_a`。

### 2026-10-16T07:14:58 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta/hard_b`。

### 2026-10-16T08:12:52 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/light_r1`。

### 2026-10-16T08:52:11 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/easy_0_r1`。

### 2026-10-16T09:31:22 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/easy_a_r1`。

### 2026-10-16T10:10:47 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/easy_b_r1`。

### 2026-10-16T10:50:13 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/med_a_r1`。

### 2026-10-16T11:29:34 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/med_b_r1`。

### 2026-10-16T12:09:15 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/hard_a_r1`。

### 2026-10-16T12:48:47 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_ams_only/hard_b_r1`。

### 2026-10-16T13:07:29 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/off`。

### 2026-10-16T13:38:10 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/light`。

### 2026-10-16T14:08:57 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/easy_0`。

### 2026-10-16T14:39:47 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/easy_a`。

### 2026-10-16T15:10:27 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/easy_b`。

### 2026-10-16T15:41:34 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/med_a`。

### 2026-10-16T16:12:32 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/med_b`。

### 2026-10-16T16:43:21 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/hard_a`。

### 2026-10-16T17:14:35 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_ams_only/hard_b`。

### 2026-10-16T18:12:43 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/light_r1`。

### 2026-10-16T18:52:05 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-16T19:32:03 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-16T20:11:54 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-16T20:51:42 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/med_a_r1`。

### 2026-10-16T21:31:29 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/med_b_r1`。

### 2026-10-16T22:11:19 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-16T22:51:08 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-16T23:09:21 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/off`。

### 2026-10-16T23:40:03 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/light`。

### 2026-10-17T00:04:00 当前进度只读核验

- 本轮应用户“当前进度”要求，读取工作记录尾部、s0/s1/s2状态/方法结论/阶段门禁/最终全强度证据标记、队列脚本和日志，检查 ps、nvidia-smi、systemctl running services、df/du；使用 `ackermann_validation.load_verified` 复核 s1 四格 hard_b_r1 的 clean及三组攻击原始CSV/产物哈希和checkpoint一致性。本轮没有修改代码、重启/终止进程、启动新实验或删除文件，仅追加本条记录。
- s0状态complete，32个training_complete、4个ladder_complete、36个evidence_complete。s1四格各8级攻击训练均已执行完，32个training_complete；其中Base与Meta-AMS hard-b候选通过，Meta-only与AMS-only候选被拒绝并回退先前安全模型。s1最终固定模型全强度评估/展示尚未全部结束，已完成3/4个ladder、29/36个evidence_complete及29个对应MP4。
- s1 hard-b新训练候选（不是回退后的最终保留模型）三组评估完成率均值/最差：Base=0.5208333/0.50（通过），Meta-only=0.5833333/0.4375（未通过），AMS-only=0.50/0.375（未通过），Meta-AMS=0.7916667/0.75（通过）。四格候选clean completion/instant/dwell均1.0，碰撞/越界/越界终止均0；三攻击种子checkpoint与对应clean一致，原始CSV与记录哈希复核通过。Meta-AMS三个攻击完成率分别0.8125、0.8125、0.75，阶段通过及视频证据完成记录时间为10月16日22:51:08。
- 注意区分最终安全模型：s1最终固定模型hard-b已完成的三格分别为Base=0.5208333/0.50、Meta-only=0.4375/0.375、AMS-only=0.4166667/0.375。Meta-AMS最终全强度hard-b尚未运行到，不能将其候选验证数值冒充已完成的final_ladders数据。
- 当前s1状态rendering，正在 `final_ladders/happo_meta_ams/easy_0` 导出真实变道视频；00:03:48前轨迹已写入约9005行（五车×1801步/3500步，36.02秒仿真），不是训练停滞。队列PID2137687与协调PID2137697仍运行约26小时，唯一Isaac播放PID2348010；GPU瞬时利用率46%，该进程显存6147MiB，另有无关ToDesk约366MiB。未见AMS/Isaac系统或用户running service，当前由nohup队列运行，未发现并发训练冲突。
- s2（第三组训练随机种子）仍preflight_passed，0个training_complete；能力保持补充训练尚未启动。现有队列顺序仍为完成s1剩余全强度评估/视频→s2四格→能力保持补充试验，不把排队任务报为已启动。
- 存储正常：系统盘可用约335G，1T项目盘可用约427G，`/home/cnc/SSD_1T/xzw/ams_runtime`约25G。产物根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1`；当前日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_resume_queue.log`。
- 下一步：让已运行队列完成s1剩余证据，再进入s2；完整多种子统计之前还需核对协议/源码迁移一致性。现有s1候选数据支持组合方案本轮hard-b表现较好，但尚不能宣称三独立训练种子显著最优。桌面AMS.zip本轮未更新。

### 2026-10-17T00:08:00 用户确认继续后的实时核验

- 未启动第二个任务；已有串行队列继续运行。s1状态仍为 `rendering`，当前导出 `final_ladders/happo_meta_ams/easy_0` 真实变道证据。Isaac play PID `2348010` 持续高CPU运行，GPU显存约6147MiB；视频文件已增长至约1.05MB、轨迹约9.3MB，说明渲染仍在推进，不是遗留僵死进程。
- 队列PID `2137687`、follow-up PID `2137697` 保持运行；s2仍未启动，补充能力保持训练也未启动。磁盘可用空间仍正常（系统盘约335GB、1T盘约427GB），未进行删除或重复训练。

### 2026-10-17T00:07:55 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/easy_0`。

### 2026-10-17T00:34:05 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/easy_a`。

### 2026-10-17T01:00:10 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/easy_b`。

### 2026-10-17T01:26:10 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/med_a`。

### 2026-10-17T01:52:19 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/med_b`。

### 2026-10-17T02:18:22 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/hard_a`。

### 2026-10-17T02:44:26 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1/final_ladders/happo_meta_ams/hard_b`。

### 2026-10-17T02:44:43 AMS后续实验

- 后续实验mode=matrix已完成，产物 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1`。

### 2026-10-17T02:44:44 AMS后续实验

- 开始后续服务mode=matrix，hard-b参考SHA=faa7d1dc8e4377c0fcc5db56db12e73b6e4707639bb2ccf347344930debe6bb7；32训练/16评估环境、3500步、六算法同BC初始化、各攻击stage最多3轮×40更新、actor LR=5e-07、训练clean混合0.50；失败clean仅导出攻击关闭的诊断证据，不晋级。协议 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/followup_manifest.json`，状态 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/followup_status.json`。

### 2026-10-17T03:33:58 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/light_r1`。

### 2026-10-17T04:14:05 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/easy_0_r1`。

### 2026-10-17T04:54:15 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/easy_a_r1`。

### 2026-10-17T05:34:30 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/easy_b_r1`。

### 2026-10-17T06:14:43 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/med_a_r1`。

### 2026-10-17T06:54:51 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/med_b_r1`。

### 2026-10-17T07:34:46 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/hard_a_r1`。

### 2026-10-17T08:15:00 AMS后续实验

- 六算法 `happo_base_shared` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_base_shared/hard_b_r1`。

### 2026-10-17T08:32:52 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/off`。

### 2026-10-17T08:59:20 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/light`。

### 2026-10-17T09:26:03 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/easy_0`。

### 2026-10-17T09:52:28 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/easy_a`。

### 2026-10-17T10:19:01 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/easy_b`。

### 2026-10-17T10:45:44 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/med_a`。

### 2026-10-17T11:12:18 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/med_b`。

### 2026-10-17T11:39:03 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/hard_a`。

### 2026-10-17T12:05:59 AMS后续实验

- happo_base_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_base_shared/hard_b`。

### 2026-10-17T12:55:20 AMS后续实验

- 六算法 `happo_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/light_r1`。

### 2026-10-17T13:30:54 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/easy_0_r1`。

### 2026-10-17T14:06:37 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/easy_a_r1`。

### 2026-10-17T14:42:09 AMS后续实验

- 六算法 `happo_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/easy_b_r1`。

### 2026-10-17T15:17:54 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/med_a_r1`。

### 2026-10-17T15:53:40 AMS后续实验

- 六算法 `happo_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/med_b_r1`。

### 2026-10-17T16:29:14 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/hard_a_r1`。

### 2026-10-17T17:05:11 AMS后续实验

- 六算法 `happo_meta` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.5625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta/hard_b_r1`。

### 2026-10-17T17:19:10 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/off`。

### 2026-10-17T17:45:57 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/light`。

### 2026-10-17T18:12:25 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/easy_0`。

### 2026-10-17T18:39:04 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/easy_a`。

### 2026-10-17T19:05:33 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/easy_b`。

### 2026-10-17T19:32:15 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/med_a`。

### 2026-10-17T19:58:53 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/med_b`。

### 2026-10-17T20:25:33 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/hard_a`。

### 2026-10-17T20:52:29 AMS后续实验

- happo_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta/hard_b`。

### 2026-10-17T21:41:01 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/light_r1`。

### 2026-10-17T22:15:41 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/easy_0_r1`。

### 2026-10-17T22:50:22 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/easy_a_r1`。

### 2026-10-17T23:25:18 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/easy_b_r1`。

### 2026-10-18T00:00:15 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/med_a_r1`。

### 2026-10-18T00:35:19 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/med_b_r1`。

### 2026-10-18T01:10:10 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/hard_a_r1`。

### 2026-10-18T01:45:43 AMS后续实验

- 六算法 `happo_ams_only` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.3125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_ams_only/hard_b_r1`。

### 2026-10-18T01:59:37 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/off`。

### 2026-10-18T02:25:45 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/light`。

### 2026-10-18T02:52:04 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/easy_0`。

### 2026-10-18T03:18:23 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/easy_a`。

### 2026-10-18T03:44:25 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/easy_b`。

### 2026-10-18T04:10:32 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/med_a`。

### 2026-10-18T04:36:40 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/med_b`。

### 2026-10-18T05:02:46 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/hard_a`。

### 2026-10-18T05:28:50 AMS后续实验

- happo_ams_only 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_ams_only/hard_b`。

### 2026-10-18T06:17:16 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/light_r1`。

### 2026-10-18T06:52:00 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-18T07:27:11 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-18T08:02:16 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-18T08:37:11 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/med_a_r1`。

### 2026-10-18T09:12:09 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/med_b_r1`。

### 2026-10-18T09:47:16 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-18T10:22:07 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-18T10:35:59 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/off`。

### 2026-10-18T11:02:08 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/light`。

### 2026-10-18T11:28:04 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/easy_0`。

### 2026-10-18T11:54:17 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/easy_a`。

### 2026-10-18T12:20:33 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/easy_b`。

### 2026-10-18T12:47:01 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/med_a`。

### 2026-10-18T13:13:12 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/med_b`。

### 2026-10-18T13:39:34 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/hard_a`。

### 2026-10-18T14:05:56 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/final_ladders/happo_meta_ams/hard_b`。

### 2026-10-18T14:06:13 AMS后续实验

- 后续实验mode=matrix已完成，产物 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2`。

### 2026-10-18T23:18:00 AMS状态接管复核

- 磁盘实际状态：s1、s2 的 `followup_status.json` 均为 `state=complete, mode=matrix`；当前未发现训练、play 或评估冲突进程。RTX 3080 当前约 14% 利用率、769 MiB 显存，主要为桌面 ToDesk，不是 AMS 训练任务。
- s1 四格结果：`happo_base_shared`、`happo_meta_ams` 通过 hard-b；`happo_meta`、`happo_ams_only` 在 hard-b 门禁停止。s2 四格结果：`happo_meta`、`happo_meta_ams` 通过 hard-b；`happo_base_shared`、`happo_ams_only` 在 hard-b 门禁停止。两轮均保留了 clean/碰撞/越界门禁结果与全强度证据（各阶段验证、轨迹、视频、图片）。
- s2 完成后的渐进能力保持补充已实际启动并完成 `light_block1`：50 updates，HAPPO+Meta+AMS，采用真实轨迹 BC/KL、攻击样本混合和专家渐退课程。训练 checkpoint：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-10-18_14-06-28_ackermann_retention_recovery_20261018_140613_light_block1_happo_meta_ams/model_final.pt`；训练/评估/图表/打包目录：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261018_140613/light_block1`。
- light_block1 攻击评估原始指标：`collision_rate=0`、`road_boundary_any_rate=0`、`min_pair_gap=0.9402`、`centerline_error_abs_mean=0.0540`、`heading_error_abs_mean=0.0192`；但当前短块的 `platoon_completion_ever_rate_final=0`，因此不能视为通过 light 门禁，仍需 clean 回归和后续修复/延长训练。
- 原队列在最后 tar 打包步骤遇到 `tar: light_block1: 文件发生变化`，以退出码 1 中止于 `run_medium_algorithm_comparison.sh`，尚未执行 clean_eval 和 stage checkpoint 选择；没有重复训练已完成的 light_block1。
- 已从现有 `model_final.pt` 接续 clean 回归（不重新训练），命令为 `isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/eval_happo_platoon.py --task Isaac-Marl-Platoon-HAPPO-v0 --num_envs 16 --seed 3407 --eval_steps 3500 --warmup_steps 0 ... env.algorithm.enable_attack=false env.algorithm.attack_level=off`；PID `2652158`，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261018_140613/clean_eval_resume.log`，输出目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261018_140613/clean_eval/happo_meta_ams`。
- 下一步：等待 clean 回归完成，读取 clean 门禁；若 clean 通过，修复 tar 竞争为非致命打包告警并从现有产物继续 light 门禁/阶段晋级；若 light 仍不通过，只延长或调整 retention 训练，不覆盖 s1/s2 安全最佳模型。
- 23:19 左右首次 clean 启动因当前 shell 残留 conda 变量仅输出 `running in conda env` 后退出，未生成评估产物；已清除 `CONDA_* / VIRTUAL_ENV / PYTHONHOME` 后从相同 checkpoint 重启，当前 PID `2652400`，日志仍为 `.../ackermann_retention_recovery_20261018_140613/clean_eval_resume.log`。
- 最小前台诊断确认 `python.sh` 与 HAPPO 评估入口正常；此前后台 wrapper 的问题来自启动链路而非代码。改用 `env -u CONDA_* -u VIRTUAL_ENV -u PYTHONHOME setsid .../python.sh` 后 clean 回归已真正运行，主 wrapper PID `2653167`、Isaac Python PID `2653173`，GPU 约 3.2 GiB；日志目前推进至约 step 440/3500，结果尚未生成。
- 2026-10-19T00:55 复核：上述 clean 评估已在 2026-10-18T23:28 完成并正常退出，当前没有训练或评估进程，GPU 仅约 769 MiB 桌面占用。clean 结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261018_140613/clean_eval/happo_meta_ams/eval_summary.csv`：`collision_rate=0`、`road_boundary_any_rate=0`、`min_pair_gap=0.9379`，但 `platoon_completion_ever_rate_final=0`、`platoon_min_path_progress_final=0`，因此 clean 门禁未通过。light_block1 同样完成率为 0；当前未启动新的训练，避免在未定位失败原因前重复浪费 GPU。下一步先诊断 retention checkpoint/空间变道完成判据与 BC 暖启动链路，再决定修正后的短训练。
- 2026-10-19T01:15 阶段确认：s1、s2 正交矩阵仍为 `complete`；retention `light_block1` 训练和 clean 回归均已结束，但 clean/light 完成率均为 0，未进入 promotion 或 hard-b 追加训练。当前阶段正式标记为“retention 能力保持失败后的根因诊断/修复准备”，无活动进程、无新 checkpoint；GPU 仍为空闲桌面占用。下一步优先对比安全最佳 `happo_meta_ams` 与 retention checkpoint 的动作/路径进度以及变道完成统计实现，确认是模型退化还是评估配置不一致后再重启训练。
- 2026-10-19T01:23 根因修复后续：修改 `run_ackermann_retention_recovery.sh`，将 retention 父模型改为 s2 hard-b 通过 checkpoint，并修正 `RESUME_RUNS_FILE` 作用域及 Isaac 可解析的父模型别名；修改 `run_medium_algorithm_comparison.sh`，tar 文件变化告警不再中断有效结果。新实验 `ackermann_retention_recovery_20261019_safe_resume` 已启动，日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261019_safe_resume/light_block1.log` 明确显示 `resume ... ams_retention_parent_s2_meta_ams/model_final.pt`，且 HAPPO 从计数 312 继续到 362，证明不是从头训练。50 次续训已完成并生成 checkpoint；当前进入该 checkpoint 的 light 攻击评估（正在评估 `model_350.pt` 与 `model_final.pt`），进程仍在运行，尚未有门禁结论。
- 01:28 复核：修复后 `model_350.pt` 的 light 评估已完成，原始指标为 return=70154.688、leader/platoon speed=0.470/0.424、speed_err=0.048、center=0.021、min_gap=1.615、collision=0、road_out=0，但当前 whole_done/min_progress 仍为 0；`model_final.pt` 仍在评估，不能提前判定门禁。当前训练已结束、评估进程仍在运行。

### 2026-10-20T00:20:47 AMS retention 诊断修复与受保护短块

- 磁盘复核：`ackermann_retention_recovery_20261019_safe_resume` 已完整结束，无遗留训练/评估进程；`light_gate.json` 明确 `passed=false`，light 攻击完成率/驻留完成率均为 0，`model_final.pt` 的 `min_pair_gap=1.1120`、collision=0、road_any=0；clean 回归 `eval_summary.csv` 的完成率同样为 0，`min_pair_gap=0.6853`。这不是进程冲突或错误父模型问题，而是续训后队形间距退化。
- 对照证据：s2 安全最佳 `happo_meta_ams` 在 final ladder 的 clean completion ever/dwell=1.0、`min_pair_gap≈1.833`，light completion=1.0、`min_pair_gap≈1.776–1.785`，故评估器与原安全模型正常。
- 代码修复：`scripts/tools/run_ackermann_retention_recovery.sh` 新增 `EVAL_EVERY`、`RETENTION_ACTOR_LR`、`RETENTION_BC_COEF`、`RETENTION_KL_COEF`、`RETENTION_CLEAN_FRACTION`、`RETENTION_ATTACK_RAMP_UPDATES` 参数化；`bash -n` 通过。`run_medium_algorithm_comparison.sh` 的 tar 文件变化告警已保持为非致命，避免有效结果被打包竞争中断。
- 1-update 参数链路诊断已完成（仅 2 env、50 steps，不作为论文结果）：产物 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261020_guarded_5u_test`，tar 与 clean/light 原始摘要均生成，随后进程已退出。
- 正式受保护短块已启动并确认 GPU 正在工作：命令使用 `TAG=ackermann_retention_recovery_20261020_guarded_5u BLOCK_UPDATES=5 EVAL_EVERY=1000 NUM_ENVS=32 EVAL_NUM_ENVS=16 RETENTION_ACTOR_LR=1e-6 RETENTION_BC_COEF=5 RETENTION_KL_COEF=1 RETENTION_CLEAN_FRACTION=0.80 RETENTION_ATTACK_RAMP_UPDATES=50`，从 s2 hard-b 安全父模型 `ams_retention_parent_s2_meta_ams/model_final.pt` 恢复，BC=5→4、KL=1→0.8、专家比例保持 1.0；当前 HAPPO 训练进程 PID 2742469/2742474，GPU 约 3.3 GiB。
- 当前阶段：正式 5-update 受保护训练进行中，尚无新门禁结论；完成后只读取该短块的 light 与 clean 结果，若仍失败则不替换 s2 安全最佳模型。

### 2026-10-20T02:07:29 AMS guarded-5u 结果

- 正式 `ackermann_retention_recovery_20261020_guarded_5u` 已完成，当前没有训练、评估或比较进程；GPU 回到桌面占用约 769 MiB。
- light 门禁：`passed=true`，完成率/瞬时完成/驻留完成均为 1.0，collision=0，road_any=0；原始摘要 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261020_guarded_5u/light_block1/evaluation/happo_meta_ams/eval_summary.csv`，`min_pair_gap_mean=1.4825`，`trajectory_progress_mean=0.7332`，`platoon_min_path_progress_final=0.6213`。
- clean 门禁：`passed=true`，完成率/瞬时完成/驻留完成均为 1.0，collision=0，road_any=0；原始摘要 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261020_guarded_5u/clean_eval/happo_meta_ams/eval_summary.csv`，`min_pair_gap_mean=1.4805`，`trajectory_progress_mean=0.7332`，`platoon_min_path_progress_final=0.6213`。
- 候选模型 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-10-20_00-20-32_ackermann_retention_recovery_20261020_guarded_5u_light_block1_happo_meta_ams/model_final.pt` 已生成，并分别写入 `promoted_light_runs.txt`、`promoted_clean_runs.txt`；该模型通过 clean/light 门禁，但最小车距仍低于 s2 原安全模型约 1.78–1.83 m，因此暂作为“通过门禁的保持候选”，不覆盖 s2 安全最佳。
- 完整产物（训练、曲线、评估、图像、tar、门禁报告）位于 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_recovery_20261020_guarded_5u`。
- 下一步：在不改动 s2 安全最佳的前提下，基于该候选继续做更高攻击强度的逐级验证；若进入 hard-b 前出现 clean/车距退化，则回退到 s2 安全模型并调整保持课程，不重复已完成训练。

### 2026-10-20T02:08:51 多训练种子状态复核

- 正交消融的三组独立训练种子 `s0/s1/s2` 均已完成，三个 `followup_status.json` 都是 `state=complete, mode=matrix`：
  - s0：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0`
  - s1：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1`
  - s2：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2`
- 每个种子均完成四配置（base_shared、meta、ams_only、meta_ams）的逐级攻击矩阵和 hard-b 门禁，含训练/验证/轨迹/视频/图像产物；每个矩阵有 32 个训练 checkpoint（四配置 × 八个攻击训练阶段）。
- 不是“所有配置都通过 hard-b”：s0 仅 `meta_ams` 通过；s1 的 `base_shared`、`meta_ams` 通过；s2 的 `meta`、`meta_ams` 通过。其余配置按门禁在 hard-b 停止，这是实验结果而非未完成任务。
- 当前 2026-10-20 的 retention guarded-5u 只是一条单独的能力保持候选训练链（训练从 s2 hard-b 父模型恢复，单一训练种子），clean/light 已通过，但它还不是多训练种子复验；后续如要把 retention 机制作为论文主结果，需要另补至少 3 个独立训练种子。

### 2026-10-20T02:11:48 关于 retention 多种子尚未启动的说明

- 用户询问为何尚未开始 retention 多训练种子。实际原因是此前用户明确要求“多训练种子的实验后续会补充，我们先不管这个”，当时优先处理 AMS-only light 失败、恢复路径错误和 retention 单种子能力保持失败；因此没有擅自占用 GPU 启动新的多种子训练。
- 当前已具备启动条件：guarded-5u 单种子候选已通过 clean/light，且目前没有训练/评估冲突进程。但这轮用户消息是询问原因，并未明确授权立即启动新的多种子训练，所以本轮不擅自开跑；若收到启动指令，将以同一 s2 hard-b 父模型、固定协议和至少 3 个独立训练种子 sequential 复验，并保留 s2 安全最佳不被覆盖。

### 2026-10-20T02:14:00 retention 多种子复验已启动

- 用户已明确同意开启实验。先将 `run_ackermann_retention_recovery.sh` 参数化：新增 `TRAIN_SEED`、`EVAL_SEED`，嵌套 medium comparison 使用训练 seed，clean 回归使用指定评估 seed；`bash -n` 通过。
- 已启动顺序多种子复验 PTY 任务，复用既有正交实验 seed `42 / 3407 / 777`，每个 seed 使用同一 s2 hard-b 父模型和 guarded-5u 协议：5 updates、actor lr=1e-6、BC=5→4、KL=1→0.8、clean fraction=0.80、专家比例=1.0；每个 seed 独立保存训练、light、clean、门禁、图表和 tar。
- 任务顺序：`s0 (42) → s1 (3407) → s2 (777)`，当前已进入 `s0`；实验根目录模式为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_multiseed_20261020_s{0,1,2}`。不覆盖 s2 正交消融安全最佳模型。

### 2026-10-20T12:36:57 retention 多种子复验完成复核

- 用户观察时 GPU 已空闲，是因为顺序任务已在凌晨全部完成，并非未启动或中途无产物：s0 于约 02:27、s1 于约 02:39、s2 于约 02:51 完成；当前无训练/评估进程，GPU 约 776 MiB 桌面占用。
- 三个 retention seed (`42/3407/777`) 的 light 与 clean 门禁均 `passed=true`：完成率=1.0、瞬时完成=1.0、驻留完成=1.0、collision=0、road_any=0。
- 三个 seed 的 light `min_pair_gap_mean` 分别为 1.4825、1.4805、1.4805 m；clean 分别为 1.4825、1.4805、1.4805 m；`trajectory_progress_mean≈0.7332`、`platoon_min_path_progress_final≈0.6213`。
- 结果目录：
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_multiseed_20261020_s0`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_multiseed_20261020_s1`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_multiseed_20261020_s2`
- 三种子均通过 clean/light，但这是 guarded-5u 能力保持复验，不等同于 hard-b 多种子复验；下一步应在不覆盖 s2 安全最佳的前提下，选择通过门禁的候选继续逐级提高攻击强度。

### 2026-10-20T12:39:00 easy_0 课程续训已启动

- 已确认当前没有冲突进程；不重复 light/clean，直接使用三个 seed 已通过 light 的候选模型作为父模型，进入 `easy_0` 攻击级别。
- 为此将 `run_ackermann_retention_recovery.sh` 参数化为可指定阶段名、攻击预算和晋级标签；`bash -n` 已通过。easy_0 使用 `FDI_pos=1.00, FDI_acc=0.18, DoS=0.035`，每个 seed 继续 5 updates，保持 actor lr=1e-6、BC=5→4、KL=1→0.8、clean fraction=0.80。
- 已按 `s0(42) → s1(3407) → s2(777)` 顺序启动，当前进入 s0；父模型来自各自 `ackermann_retention_multiseed_20261020_s{0,1,2}/promoted_clean/happo_meta_ams/model_final.pt`。每个阶段独立输出 `easy_0_gate.json`、`clean_gate.json`、曲线、视频/轨迹（若评估器生成）及 tar，不覆盖 s2 正交消融安全最佳。
- 当前任务 PTY session=48808；产物目录模式 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easy0_20261020_s{0,1,2}`。

### 2026-10-20T12:42:00 easy_0 运行状态复核

- s0 的 easy_0 训练已生成 `model_best.pt`、`model_final.pt`、`platoon_metrics.csv`，当前正在进行 easy_0 攻击评估；GPU 显存约 1.7 GiB，训练/评估进程存在且无并发冲突。
- s1、s2 尚未开始训练，仍按顺序排队；当前没有重复 light/clean，也没有覆盖任何安全最佳模型。
- s0 当前产物根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easy0_20261020_s0/easy_0_block1`；待 easy_0 和 clean 双门禁结束后，才继续 s1，再继续 s2。

### 2026-10-21T01:34:00 easy_0 clean 门禁完成并启动 easy_a

- 磁盘复核：无遗留 Isaac 训练/评估进程、无相关 systemd 服务占用；easy_0 三个 seed 的攻击评估已存在，未重复训练。随后补做 3500 步 clean 回归，评估输出分别为：
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easy0_20261020_s0/easy_0_block1/clean_eval/happo_meta_ams/eval_summary.csv`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easy0_20261020_s1/easy_0_block1/clean_eval/happo_meta_ams/eval_summary.csv`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easy0_20261020_s2/easy_0_block1/clean_eval/happo_meta_ams/eval_summary.csv`
- s0/s1/s2 clean 结果均为 completion/dwell/instant=1.0、collision=0、road_boundary_any=0，最小车间距分别约 1.482/1.480/1.480 m；攻击门禁也用 `select_ackermann_stage_checkpoint.py` 重新核对，三者均通过。
- 已创建三种子的 `easy_0_gate.json`、`clean_gate.json`、`promoted_easy_0_runs.txt`、`promoted_clean_runs.txt`，保留旧模型与旧目录不覆盖。
- 已从各自 easy_0 clean promoted checkpoint 顺序启动 easy_a 训练，攻击预算为 `FDI_pos=1.25, FDI_acc=0.25, DoS=0.05`；每 seed 5 updates、16 eval envs、3500 步攻击评估，保持真实轨迹 retention、HAPPO+AMS、Ackermann 空间路径和 5 车队形配置。训练根目录：
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easya_20261021_s0/easy_a_block1`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easya_20261021_s1/easy_a_block1`
  - `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easya_20261021_s2/easy_a_block1`
- 当前状态：s0 已完成训练并正在进行 easy_a 攻击评估；s1、s2 尚未开始，按顺序排队；当前 PTY session=58072。s0 训练 run 为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-10-21_01-41-18_ackermann_retention_easya_20261021_s0_easy_a_block1_happo_meta_ams`。
- 下一步：等待 s0/s1/s2 easy_a 攻击评估完成，随后为每个 seed 补做 clean 回归并执行 attack+clean 双门禁；只有三者均通过才晋级 easy_b。

### 2026-10-21T13:17:00 easy_a 三种子训练完成复核

- 磁盘与进程复核：easy_a s0/s1/s2 的 `training/runs.txt`、`model_final.pt` 和攻击评估 `evaluation/happo_meta_ams/eval_summary.csv` 均已生成；训练/攻击评估进程已全部退出，GPU 回到桌面占用水平。
- easy_a 三个攻击评估均为 completion/dwell/instant=1.0、collision=0、road_boundary_any=0；`platoon_min_path_progress_final` 分别约 0.621302、0.621264、0.621275，`min_pair_gap_mean` 分别约 1.482485、1.480461、1.480492 m。
- easy_a 当前仍未完成阶段门禁：三个 `clean_eval`、`easy_a_gate.json`、`clean_gate.json` 尚未生成。已启动同协议 3500 步 clean 回归，当前为 s0，PTY session=94017，s1/s2 按顺序排队。
- easy_a 训练产物目录：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easya_20261021_s{0,1,2}/easy_a_block1`；只有三种子 clean+attack 双门禁都通过后，才允许晋级 easy_b。
- 2026-10-21T当前实验结果汇总复核：本轮仅只读检查日志、CSV、gate 文件和进程状态，未启动/修改实验。当前无训练或评估进程。
  - 已确认的正式 verified six-algorithm 矩阵：`logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/six_algorithm_attack_matrix.csv`。Meta-AMS 从 off 至 hard-a 全部 whole-platoon completion=1.0，hard-b mean/worst=0.75/0.75，collision/road violation/road termination 均为 0；hard-b 对比 MAPPO 0.625/0.5625、HAPPO-no-Meta 0.5833/0.50、HARL-HAA2C 0.50/0.4375、HARL-MAPPO-shared 0.1042/0、HARL-HATRPO 0.0208/0。该矩阵是单训练链、三评估种子，不等同于三独立训练种子。
  - 已确认正交消融三独立训练种子：`logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s{0,1,2}/six_algorithm_attack_matrix.csv`。hard-b Meta+AMS 三种子 mean/worst 分别为 s0=0.7500/0.7500、s1=0.7917/0.7500、s2=0.7500/0.6875，3/3 通过；跨种子均值 0.7639/0.7292。base_shared 2/3 通过，Meta 1/3，AMS-only 1/3；三种子全部矩阵碰撞、越界和道路终止均为 0。每个种子均保留训练曲线、CSV、轨迹、top-down 图和 MP4。
  - retention 能力保持复验：`ackermann_retention_multiseed_20261020_s{0,1,2}` 的 clean/light 均通过，随后 easy_0 三种子 clean+attack 均通过；easy_a 三种子攻击和 clean 原始 CSV 已生成，completion/dwell/instant=1.0、collision=0、road_any=0，min_pair_gap 约 1.480--1.482 m，但当前尚未生成 easy_a_gate.json/promotion 文件，因此只能记为原始评估通过、阶段正式晋级待确认。easy_a 当前每种子仅有 CSV 和三张汇总图，尚未形成完整视频/top-down/轨迹证据包。
  - 后续必须完成：easy_a 正式双门禁闭环；easy_b、med_a、med_b、hard_a、hard_b 三种子 retention 续训与 clean/attack 双门禁；为 retention 阶段补齐真实变道 MP4、俯视轨迹和轨迹 CSV；若论文要声称六算法统计泛化，还需在统一当前协议下补充六算法独立训练种子及跨训练种子统计。既有 light/easy_0 训练不应重复。

### 2026-10-22T22:37--22:50 retention 课程续训启动

- 磁盘检查确认无 Isaac 训练/评估进程冲突；easy_a 三个 seed 的 clean/attack 原始 CSV 均完整（3500 步），随后用 `scripts/tools/select_ackermann_stage_checkpoint.py` 补齐正式门禁。
- easy_a s0/s1/s2 的 `easy_a_gate.json` 与 `clean_gate.json` 均为 `passed=true`，完成率/瞬时完成/驻留完成均为 1.0，collision=0，road_any=0；分别保存在 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easya_20261021_s{0,1,2}/easy_a_block1/`，并生成 `promoted_easy_a` 与 `promoted_clean`。
- 已按既定优先级从 easy_a promoted checkpoint 启动 easy_b 三种子顺序续训，s0(42)→s1(3407)→s2(777)，不重复 light/easy_0/easy_a 训练。攻击预算为 FDI_pos=1.50、FDI_acc=0.35、DoS=0.07；保持 guarded-5u 协议：actor_lr=1e-6、BC=5→4、KL=1→0.8、clean_fraction=0.80、attack ramp=50、专家比例 1.0→1.0。
- 当前运行批次：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/easyb_batch.log`，PTY session=48759；s0 已完成训练并进入 clean 回归，s1/s2 按顺序排队。s0 目录为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easyb_20261022_s0/easy_b_block1`。
- 当前尚无 easy_b 门禁结论；必须等待每个 seed 的 attack+clean 双门禁，三者均通过后才启动 med_a。若阶段失败，保留失败证据且不将失败模型晋级。
- easy_b s0 已完成 attack+clean 双门禁并通过：`easy_b_gate.json` 与 `clean_gate.json` 均为 `passed=true`，completion/instant/dwell=1.0、collision=0、road_any=0；结果目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easyb_20261022_s0`。批次已自动进入 s1（seed=3407），当前仍在运行。
- easy_b s0/s1/s2 已全部完成 attack+clean 双门禁且均 `passed=true`；三者 completion/instant/dwell 均为 1.0，collision=0、road_any=0。结果目录为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_easyb_20261022_s{0,1,2}`，批次日志 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/easyb_batch.log` 显示 `all seeds complete`。
- easy_b 全部通过后，下一阶段按顺序进入 med_a，使用各自 `promoted_easy_b/happo_meta_ams/model_final.pt`，攻击预算 FDI_pos=1.75、FDI_acc=0.45、DoS=0.09，仍保持 guarded-5u retention 协议；不得重复 easy_b 或更低级别训练。
- 已启动 med_a 三种子顺序批次 s0(42)→s1(3407)→s2(777)，PTY session=15958；s0 当前训练/后续攻击评估运行中，批次日志为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/meda_batch.log`。本阶段必须完成 attack+clean 双门禁后才允许进入 med_b。
- 最新进程复核：med_a s0 已完成 5-update 训练并进入 3500 步 attack 评估，实际命令使用 `attack_level=medium, FDI_pos=1.75, FDI_acc=0.45, DoS=0.09`；当前无其他并发 Isaac 任务。s0 结果目录为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s0/med_a_block1`，尚未产生 med_a gate 结论。

### 2026-10-23T14:46 当前进度复核及 retention 门禁结论更正

- 本轮用户询问进度；仅检查进程、systemd 服务、原始 CSV、配置及源码，除本工作记录外未修改文件，未启动新训练或评估。检查命令包括 `ps -eo pid,etime,comm,args`、`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`、system/user `systemctl list-units`、`tail meda_batch.log`，以及 Python csv/json 只读统计、`rg`/`nl` 源码定位。
- 当前没有 Isaac 训练/评估进程，也无相关运行中的 systemd 服务；GPU compute 列表仅 ToDesk_Session（366 MiB）。med_a 批次正常结束，日志含 `[MED_A_BATCH] all seeds complete`。三个 seed 门禁文件时间分别为 2026-10-22 23:36:21、23:48:18、2026-10-23 00:00:16。已执行的批次只覆盖 med_a 三种子，并未建立自动衔接 med_b/hard_a/hard_b 的任务；retention 根目录无这些后续阶段目录。
- med_a 三个 seed 均存在训练 checkpoint、各 3500 行 attack/clean 评估及 `med_a_gate.json`、`clean_gate.json`，文件标记 passed=true，completion/instant/dwell=1、collision/road_any/road_termination=0。路径：`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s{0,1,2}`。每种子 MP4/topdown/trajectory CSV 数量均为 0。
- 重要更正：逐行核对上述 attack `eval_steps_model_final.csv`，三个 seed 全部 3500 步均记录 `attack_enabled=1`，但 `attack_strength_scale=0`、`attack_clean_env_fraction=1`、`shield_expert_blend_ratio=1`；clean 也全程 expert_blend_ratio=1。这不是无专家、全强度攻击验证，不能把现有 passed=true 作为自主 Meta-AMS 通过 med_a 的证据。
- 扩展检查该 retention 分支的 light/easy_0/easy_a/easy_b/med_a 共 15 个 attack CSV（各 s0/s1/s2），均出现同样的三项协议值（0/1/1）。因此先前把该分支 easy_a/easy_b 称为正式严格通过的表述不充分，现更正为：训练与评估产物已完成、宽松汇总门禁标记通过，但自主攻击有效性未通过严格协议核验。此检查不直接否定独立的 verified 主实验/正交消融；它们本轮未重新审计。
- 代码依据：`source/my_exts/marl_platoon/algorithms/pipeline.py:254-274` 按 curriculum 覆盖专家比例与 clean fraction；med_a env.yaml:6513/6517/6518 设置 curriculum=true、expert_start=end=1；`shield.py:213-221` 在 blend=1 时以 expert_actions 替代 guarded 动作。`scripts/tools/select_ackermann_stage_checkpoint.py:60-68` 只检汇总完成/安全指标，没有校验专家比例、攻击缩放或 clean fraction；现有严格工具 `scripts/tools/ackermann_validation.py:66-87` 则明确要求 expert=0，攻击时 strength=1、clean_fraction=0。
- 下一步应优先纠正 retention 评估协议和门禁覆盖，保留全部现有结果，先复用已完成 checkpoint 做无专家 clean 与全强度多攻击种子验证；根据验证决定回退或续训，不应直接把现有 promotion 当作 med_b 的有效晋级依据。待协议有效后再推进 med_b→hard_a→hard_b，并补齐视频/轨迹，随后进行尚缺的六算法多训练种子实验。当前无新实验在后台排队或运行。

### 2026-10-25T20:08:24 AMS后续实验

- 启动 retention 已有 checkpoint 严格复验（不训练），新目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/strict_revalidation_20261025`；无专家 clean 全部通过才解锁低到高攻击；全过程保持原始模型与旧产物。

### 2026-10-25T20:10 retention 验证协议修复与实机复验启动

- 启动前检查：没有训练/评估冲突任务，旧 AMS systemd failed units 未重启；数据盘可用 373 GiB。保留原有 dirty worktree、模型、CSV、gate 和论文。本轮未修改 AMS/HAPPO 核心算法、环境物理模型、奖励或论文，未执行任何新训练。
- 新增 `scripts/tools/revalidate_ackermann_retention.py`：读取各 med_a 现有 `training/runs.txt` 与 env.yaml；保留该分支原来的空间路径、完成阈值与 safety_shield 参数，显式关闭评估时的 retention curriculum/schedule、专家接管（expert=0），固定攻击 strength=1、ramp=0、clean_fraction=0；先对所有候选做严格 clean，任何候选 clean 不通过则不解锁攻击；全部通过后按 light→easy_0→easy_a→easy_b→med_a 逐级进行三攻击种子验证。复用 `ackermann_validation.py` 和 `Followup` 的逐行核验、内容哈希缓存、限时执行与证据导出，失败不晋级。
- 修改 `scripts/tools/select_ackermann_stage_checkpoint.py`：补充原始 3500 步/16 环境核验，强制无专家且攻击开启时完整强度/零 clean 混入；安全指标改用原始逐步最大值而非时均值；缺文件/缺字段/不完整评估 fail closed。该旧 selector 仅用于单次评估，多攻击种子晋级由严格 driver 完成。
- 修改 `scripts/tools/run_ackermann_retention_recovery.sh`：训练后的旧混合协议评估禁用（SKIP_EVAL=1），改为调用严格 driver，先 clean 后攻击，拒绝失败候选；不再使用原来允许 road_any=0.10/collision=0.01 的弱门禁。历史训练没有重跑。
- 新增 `scripts/tools/test_retention_validation.py`。测试命令 `/home/cnc/Downloads/ENTER/bin/python3 -m unittest discover -s scripts/tools -p test_retention_validation.py -v`（15 项运行通过，含继承的基础用例），同目录 `test_ackermann_followup.py` 11 项通过；`bash -n scripts/tools/run_ackermann_retention_recovery.sh` 通过。真实旧 med_a s0/s1/s2 CSV 经新 selector 检查全部被拒绝：step 1 `shield_expert_blend_ratio != 0`。
- preflight 和启动参数：`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/revalidate_ackermann_retention.py --output logs/ackermann_retention_recovery/strict_revalidation_20261025 --stage-root logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s0/med_a_block1 --stage-root logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s1/med_a_block1 --stage-root logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s2/med_a_block1 --render`；preflight 使用同参数加 `--preflight-only`。
- 持续运行服务：`systemd-run --user --unit=ams-retention-strict-20261025 --property=WorkingDirectory=/home/cnc/SSD_1T/xzw/IsaacLab-main --property=KillMode=control-group --property=TimeoutStopSec=30 --property=RuntimeMaxSec=12h <上述命令>`。20:08:23 启动，MainPID=3094432；已确认 active/running，Isaac eval PID=3094444，占约2862 MiB；candidate_0/s0 正在执行3500步无专家 clean，trajectory_env0.csv 持续增长（20:10约3871行），尚无最终完成率结论。每次评估有1800秒超时，服务cgroup约束辅助防止遗留进程。
- 新结果根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/strict_revalidation_20261025`；包含 `revalidation_manifest.json`、`source_snapshot/`、`followup_status.json`，评估请求/日志/轨迹位于 `candidate_0/clean/validation/clean_env42/`；后续生成 `verified.json`、`validation_gate.json`、plots 和真实视频/俯视轨迹。s0 checkpoint SHA256=`69af5df579c2f5c0f16adf7772f5edaa1c108a32e1767c89e32a06eedddbe50c`。
- 新 driver SHA256=`e33feb7c06a29a120d99da58049298b28efae9c42ac166488e51d29c34cef586`，selector SHA256=`32c962554361bfa5ebf0ad9b9e92be4853ec11d489365ea3bf302036ac489c2b`，recovery wrapper SHA256=`72d6850dd94f4efd8ce92ee353c0f1c8ee56e69c94cd3a345a0a26b69520f4c4`。
- 下一步：依据新严格结果判断当前候选是否可用；若 clean 失败，先保留诊断视频、定位退化并选择有效父模型，不直接提升到 med_b。当前自动队列仅负责严格复验与证据，不会擅自重复训练或自动开始六算法矩阵。
- 回复前再次核查：`systemctl --user show ams-retention-strict-20261025.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus` 显示 active/running、MainPID=3094432；`followup_status.json` 仍为 s0 clean evaluating，evaluation.log 已推进至 step=1520/3500。尚未产生本轮最终门禁结果；s1/s2 由同一服务顺序执行。

### 2026-10-25T20:14:29 AMS后续实验

- retention 严格无专家 clean 复验 candidate_0: passed=False, completion=0.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/strict_revalidation_20261025/candidate_0/clean`。

### 2026-10-25T20:20:19 AMS后续实验

- retention 严格无专家 clean 复验 candidate_1: passed=False, completion=0.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/strict_revalidation_20261025/candidate_1/clean`。

### 2026-10-25T20:26:14 AMS后续实验

- retention 严格无专家 clean 复验 candidate_2: passed=False, completion=0.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/strict_revalidation_20261025/candidate_2/clean`。

### 2026-10-25T22:37 当前进度复核：三种子严格 clean 未通过，诊断证据完整

- 用户询问当前进度。本轮只读检查服务、进程、源码及实验产物，除追加本记录外没有修改代码/配置/实验结果，也未启动新训练或重复评估。
- 检查命令：`systemctl --user show ams-retention-strict-20261025.service -p ActiveState -p SubState -p MainPID -p ExecMainStatus -p Result`、`journalctl --user -u ams-retention-strict-20261025.service -n 45 --no-pager`、`ps -eo pid,ppid,etime,pcpu,pmem,args`、`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv`；Python 只读调用 `ackermann_validation.load_verified()` 重新核验模型和 CSV 哈希/逐行协议，并检查渲染产物 SHA256。
- 三种子严格 clean 分别于20:14:29、20:20:19、20:26:14结束；全部视频导出后服务于20:49:34正常结束。当前 `followup_status.json.state=complete_clean_rejected`、`attacks_unlocked=false`，systemd inactive/dead、Result=success，MainPID=0；没有 Isaac 训练/评估进程。GPU 总占用776MiB不代表训练仍在进行。服务正常完成与候选门禁通过是不同含义。
- s0/s1/s2 每份原始评估均3500步、16环境，逐步 expert_blend_ratio=0、attack_enabled=0；全车队 completion/instant/dwell 均0，collision/road_any/road_termination 均0。协议核验通过但任务完成门禁失败；没有解锁任何新攻击评估、med_b续训或后续强攻击晋级。
- 进一步只读诊断：三份评估重置前step3499的 `platoon_min_path_progress=1.0`，因此不能把完成率0解释为车辆完全没有移动/没有走完路径。此时 `platoon_completion_max_gap_error` 分别3.260821、3.267228、3.274504 m，明显超过当前0.35m车距误差阈值；横向中心误差分别0.03257、0.03496、0.03539m，航向误差0.01002、0.00820、0.00779rad。该 gap 指标是每环境最大相邻车距误差再跨16环境取均值，并非所有环境的全局最大值（`algorithms/router.py:1005-1008`）；完成判据见 `trajectory.py:324-352`。路径进度均值>=0.99期间的该 gap 指标约1.527--3.275m，车距保持是已发现的明显问题；尚未确认其训练/控制根因。最后step3500含终止重置，不用该行进度0推断停滞。
- 产物根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/strict_revalidation_20261025`：各 `candidate_{0,1,2}/clean/validation/clean_env42/` 有 verified.json、eval_summary.csv、eval_steps_model_final.csv、trajectory_env0.csv；各 `clean/evidence/off/` 有 `video/rl-video-step-0.mp4`、`topdown_true_xy.png`、轨迹CSV和reward_attack_safety.png，三份render_complete.json中所有产物SHA256均匹配且 diagnostic_only=true。这些是失败诊断证据，不是通过视频。
- 下一步优先诊断无专家时纵向速度/车距控制和训练中专家退出的问题，核实可用父模型并恢复满足严格clean门禁的自主编队能力；门禁不降低，未通过前不推进更强攻击训练。既有独立verified主实验与正交消融不因本次retention失败而自动作废，本轮未重新审计它们。

### 2026-10-25T22:47 retention 追赶速度配置修复及隔离复验启动

- 用户授权“先修复然后再进行下一步实验”。启动前确认无 Isaac 训练/评估冲突，旧 failed 服务未重启，数据盘可用372GiB；未找到适用AGENTS.md，保留现有dirty工作区和全部历史结果。
- 直接代码与轨迹证据：`algorithms/shield.py:250-267` 对追赶车辆固定赋值catchup_action；`wrappers.py:99-108` 前后裁剪归一化动作至0.4；`ackermann_action.py:process_actions` 将速度乘1.2m/s。retention沿用了通用比较脚本 `run_medium_algorithm_comparison.sh:41` 的0.355，物理追赶速度仅0.426m/s，而头车0.4对应0.48m/s，相差0.054m/s。旧s0轨迹step1000头车/第一跟车位置37.9023/35.2972m，step3499为61.3738/56.3424m，第一间距从2.6051增至5.0314m；追赶执行速度长期0.426，符合配置失配导致持续掉队的表现。
- 参考验证：重新调用 `load_verified()` 审核 `logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2/stages/happo_meta_ams/hard_b_r1/validation/clean_env42/verified.json`，clean completion/instant/dwell=1且安全事件0，命令明确catchup_action=0.40。其shield源码哈希与当前相同（7ddb7f9fdfef21c971cc0dd35c249916d36bc54fe859f1d52b1adc3029a84baf），因此本轮首先恢复该已验证追赶设置，不引入新的控制律。
- 修改 `scripts/tools/revalidate_ackermann_retention.py`：新增显式 `--catchup-action`，只允许有限正值且不超过原happo_action_clip；将原值与新值写入新manifest的explicit_repairs，未覆盖旧env.yaml。评估和视频共用同一修复参数；clean全通过后也保存clean视频，再执行逐级攻击。修改 `run_ackermann_retention_recovery.sh`：后续训练默认显式catchup_action=0.40，避免掉回通用0.355默认值。未改动AMS/HAPPO、奖励、完成判据、专家控制算法、前后轮结构或历史实验结果。
- 测试：`/home/cnc/Downloads/ENTER/bin/python3 -m unittest discover -s scripts/tools -p test_retention_validation.py -v` 共16项运行通过（含继承/基础用例），新增用例验证只变更追赶参数、保持原配置与所有门禁、拒绝越限/NaN。`bash -n scripts/tools/run_ackermann_retention_recovery.sh`通过。
- 新运行命令：`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/revalidate_ackermann_retention.py --output logs/ackermann_retention_recovery/catchup_repair_20261025 --stage-root logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s0/med_a_block1 --stage-root logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s1/med_a_block1 --stage-root logs/ackermann_retention_recovery/ackermann_retention_meda_20261022_s2/med_a_block1 --catchup-action 0.4 --render`；先以同参数加`--preflight-only`通过。
- 22:44:51由 `systemd-run --user --unit=ams-retention-catchup-20261025 --property=WorkingDirectory=/home/cnc/SSD_1T/xzw/IsaacLab-main --property=KillMode=control-group --property=TimeoutStopSec=30 --property=RuntimeMaxSec=12h <上述命令>` 启动，MainPID3121269、active/running；22:46:58 s0轨迹已到1031步，尚无最终门禁。本次复用三份已有checkpoint、不重复训练，隔离验证配置修复效果；全3个严格clean（16环境/3500步）通过后才解锁light→easy_0→easy_a→easy_b→med_a，每级三个攻击评估种子，任一级失败阻止更强级别。
- 新产物根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025`，已保存catchup_diagnosis.json、revalidation_manifest.json、source_snapshot及s0评估请求/日志/流式轨迹；仍保留失败基线strict_revalidation_20261025用于同模型对照。driver SHA256=e1278488c7d134e9e27ce9bc388e03ed2e43587d69b4089de2c8ac7d04c46c62；wrapper SHA256=89aa4d4c033c96458c3507e610e4f19f45f154d45cf6553c29d804ba1277b256。
- 附加只读核查：s2安全父模型与s0 med_a checkpoint中HAPPO actor optimizer的实际lr均5e-7，配置曾写1e-6；runner.load_state_dict会恢复优化器参数。此事实未作为本轮失败根因，尚未修改恢复逻辑；后续新训练前需明确记录实际lr，不能仅按配置宣称训练步长。
- 下一步：等待严格clean验证确认修复效果；通过后由现有服务执行低到高攻击验证，失败保留证据再诊断，不降低0.35m车距误差/完成/碰撞/越界门禁。该单参数修复的提升若获验证，只能归于控制配置修正，不能直接当作AMS学习增益或新训练成果。

### 2026-10-25T22:51 修复后s0严格clean通过，s1正在评估

- s0已完成16环境/3500步严格无专家clean；再次调用 `ackermann_validation.load_verified()` 验证模型/产物哈希和原始协议，completion/instant/dwell全部1.0，collision/road_any/road_termination全部0，`candidate_0/clean/validation_gate.json`为passed=true。checkpoint SHA256仍69af5df579c2f5c0f16adf7772f5edaa1c108a32e1767c89e32a06eedddbe50c，与修复前相同，未重训。
- 重置前step3499，platoon_min_path_progress=1，车距误差指标（每环境相邻对最大误差的16环境均值）从修复前3.260821m降至0.171499m；0.35m阈值保持不变。除显式catchup_action从0.355恢复0.4外，沿用同一评估设置/模型/核心源码。该对照支持追赶配置失配是本次s0 clean失败的关键原因，不能据此宣称训练收益或全部种子已通过。
- 服务 `ams-retention-catchup-20261025.service`保持active/running、MainPID3121269；22:50:56自动进入candidate_1/s1 clean，Isaac PID3122079正在初始化，s2随后执行。当前尚未解锁攻击；三种子都通过后继续已排定的clean视频及light→med_a多攻击种子评估，任何失败停止更强级别。
- 本轮最终结果根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025`；s0门禁/原始CSV/轨迹/进度安全图已生成，catchup_diagnosis.json补充实测对照。视频由服务在三份clean结束后生成。当前仍未启动新训练；无需重复旧训练来验证此次配置修复。

### 2026-10-25T22:44:51 AMS后续实验

- 启动 retention 已有 checkpoint 严格复验（不训练），新目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025`；无专家 clean 全部通过才解锁低到高攻击；全过程保持原始模型与旧产物。

### 2026-10-25T22:50:56 AMS后续实验

- retention 严格无专家 clean 复验 candidate_0: passed=True, completion=1.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/clean`。

### 2026-10-25T22:57:03 AMS后续实验

- retention 严格无专家 clean 复验 candidate_1: passed=True, completion=1.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/clean`。

### 2026-10-25T23:03:00 AMS后续实验

- retention 严格无专家 clean 复验 candidate_2: passed=True, completion=1.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/clean`。

### 2026-10-25T23:51:54 AMS后续实验

- retention candidate_0/light 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/light`。

### 2026-10-26T00:17:41 AMS后续实验

- retention candidate_1/light 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/light`。

### 2026-10-26T00:43:24 AMS后续实验

- retention candidate_2/light 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/light`。

### 2026-10-26T01:09:08 AMS后续实验

- retention candidate_0/easy_0 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/easy_0`。

### 2026-10-26T01:35:03 AMS后续实验

- retention candidate_1/easy_0 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/easy_0`。

### 2026-10-26T02:00:43 AMS后续实验

- retention candidate_2/easy_0 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/easy_0`。

### 2026-10-26T02:26:27 AMS后续实验

- retention candidate_0/easy_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/easy_a`。

### 2026-10-26T02:52:19 AMS后续实验

- retention candidate_1/easy_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/easy_a`。

### 2026-10-26T15:29 进度复核与初始化崩溃后的断点恢复

- 用户询问当前进度。通过 `systemctl --user show/cat ams-retention-catchup-20261025.service`、journalctl、ps、nvidia-smi和Python只读核验CSV/JSON/SHA256确认：服务于02:58:46以exit=1中断，检查时没有Isaac进程。失败发生在candidate_2/s2的easy_a、env_seed3407/attack_seed9917初始化，`evaluation.log`报告 `free(): invalid pointer` 和native abort；目录只有request.json/evaluation.log，没有任何评估CSV，不能记作算法门禁失败。原生崩溃具体根因尚未确定。
- 截至恢复前：三种子clean全部严格通过；light和easy_0均3/3模型通过，每模型含三个攻击评估种子；easy_a的s0/s1完整通过，s2已完成env42/attack3407一个评估，第二个初始化崩溃、第三个未开始。easy_b/med_a尚未开始。共28份独立完成评估（3 clean+25 attack）通过`load_verified()`重新校验，所有completion=1且collision/road_any/road_termination=0；攻击逐行协议满足expert=0、strength=1、clean_fraction=0。
- clean step3499车距误差指标（每环境最大误差的跨环境均值）s0/s1/s2分别0.171499/0.166621/0.167503m。门禁未放宽，本次是已有模型修正追赶配置后的复验，不是新训练成果。
- 11份视频/轨迹/俯视图证据包已经完成（s0四级、s1四级、s2三级）；全部render_complete.json记录的产物哈希复核一致。运行源码相对于revalidation_manifest.json无变化，模型/命令/协议与原记录一致，可安全复用完成结果。结果根目录仍为 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025`，数据盘剩余368GiB。
- 沿用此前“修复后继续实验”授权，在确认无进程冲突后执行 `systemctl --user start ams-retention-catchup-20261025.service` 断点恢复，未改动任何算法/配置/验证代码，未重复训练。MainPID3172283，active/running；已复用前28份完成评估和11份视频，15:29:12回到s2/easy_a/env3407/attack9917，Isaac PID3173018启动。失败尝试完整保存在 `candidate_2/easy_a/validation/attack_env3407_seed9917.superseded_1792999752934642685/`，原validator日志追加保留。
- 下一步：完成s2剩余easy_a评估与证据；三种子通过后按原队列继续easy_b→med_a，失败停止晋级。若再次出现初始化原生崩溃则进一步定位，不能将重启描述为已根治该底层问题。当前没有新训练。
- 回复前确认已越过初始化并实际推进仿真：s2/easy_a/env3407/attack9917的trajectory_env0.csv已2151行（约430步的5车记录），GPU中Isaac PID3173018占2862MiB，服务仍active/running。

### 2026-10-26T15:28:38 AMS后续实验

- 启动 retention 已有 checkpoint 严格复验（不训练），新目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025`；无专家 clean 全部通过才解锁低到高攻击；全过程保持原始模型与旧产物。

### 2026-10-26T15:28:40 AMS后续实验

- retention 严格无专家 clean 复验 candidate_0: passed=True, completion=1.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/clean`。

### 2026-10-26T15:28:41 AMS后续实验

- retention 严格无专家 clean 复验 candidate_1: passed=True, completion=1.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/clean`。

### 2026-10-26T15:28:42 AMS后续实验

- retention 严格无专家 clean 复验 candidate_2: passed=True, completion=1.0, collision=0.0, road_any=0.0；已有模型未重训，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/clean`。

### 2026-10-26T15:28:46 AMS后续实验

- retention candidate_0/light 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/light`。

### 2026-10-26T15:28:50 AMS后续实验

- retention candidate_1/light 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/light`。

### 2026-10-26T15:28:54 AMS后续实验

- retention candidate_2/light 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/light`。

### 2026-10-26T15:28:57 AMS后续实验

- retention candidate_0/easy_0 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/easy_0`。

### 2026-10-26T15:29:01 AMS后续实验

- retention candidate_1/easy_0 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/easy_0`。

### 2026-10-26T15:29:05 AMS后续实验

- retention candidate_2/easy_0 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/easy_0`。

### 2026-10-26T15:29:08 AMS后续实验

- retention candidate_0/easy_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/easy_a`。

### 2026-10-26T15:29:12 AMS后续实验

- retention candidate_1/easy_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/easy_a`。

### 2026-10-26T15:48:54 AMS后续实验

- retention candidate_2/easy_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/easy_a`。

### 2026-10-26T16:14:54 AMS后续实验

- retention candidate_0/easy_b 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/easy_b`。

### 2026-10-26T16:40:44 AMS后续实验

- retention candidate_1/easy_b 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/easy_b`。

### 2026-10-26T17:06:24 AMS后续实验

- retention candidate_2/easy_b 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/easy_b`。

### 2026-10-26T17:32:06 AMS后续实验

- retention candidate_0/med_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_0/med_a`。

### 2026-10-26T17:57:53 AMS后续实验

- retention candidate_1/med_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_1/med_a`。

### 2026-10-26T18:23:40 AMS后续实验

- retention candidate_2/med_a 严格复验 passed=True，attack_min=1.0，结果 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025/candidate_2/med_a`。

### 2026-10-26T20:38 进度复核：追赶修复后三种子clean至med_a全部严格通过

- 本轮用户询问进度；执行date、systemctl --user show、journalctl、ps、nvidia-smi，以及Python只读调用`ackermann_validation.load_verified()`和证据包SHA256核验。除追加本记录外无修改，未启动新训练/评估。
- 服务 `ams-retention-catchup-20261025.service` 于18:23:40正常结束，当前inactive/dead、Result=success、ExecMainStatus=0、MainPID=0；followup_status.json为complete_revalidated、training_performed=false。当前无Isaac训练/评估进程，GPU compute列表仅ToDesk_Session368MiB；不是再次卡住或异常退出。
- s0/s1/s2均完成clean、light、easy_0、easy_a、easy_b、med_a，18个模型/阶段门禁全部passed=true。排除复用clean的符号链接后共有48份独立原始评估（clean3份，五个攻击强度各9份）；逐份重新核验模型/原始CSV哈希及3500步/16环境协议，全部completion/instant/dwell=1.0，collision/road_any/road_termination=0。攻击均按无专家、strength=1、clean_fraction=0协议验证。
- 18份完整证据包（3模型×6阶段）内的视频、真实轨迹CSV、俯视轨迹图和奖励/安全图哈希全部匹配render_complete.json；历史失败尝试及旧数据完整保留。根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/catchup_repair_20261025`；各 `candidate_{0,1,2}/{clean,light,easy_0,easy_a,easy_b,med_a}/validation_gate.json`可独立复核。
- 范围说明：本轮完成的是已有三份retention模型在追赶配置修复后的严格复验，未新增训练，不能据此把旧专家接管阶段重新认定为有效攻击训练。该修复分支的med_b/hard_a/hard_b尚未启动，也没有排队中的新训练；既有独立主实验/正交消融的hard_b属于另一实验记录。
- 下一步优先在当前已通过的模型/修复配置基础上推进med_b→hard_a→hard_b，仍采用clean回归+三攻击评估种子门禁；启动实际续训前核对实际优化器lr与攻击/专家课程状态，避免再次由resume或课程覆盖配置。全部强度完成后再推进尚缺的统一协议对照实验。

### 2026-10-26T20:53 修复分支真实续训启动：med_b→hard_a→hard_b

- 用户授权“修复分支然后继续开始训练”。启动前确认无Isaac训练/评估冲突；三个父模型均来自catchup_repair_20261025的已通过med_a记录，重新校验其clean和三个攻击种子原始数据、模型与评估源码哈希。未改动旧模型/论文/核心AMS、HAPPO、车辆物理或门禁。
- 新增 `scripts/tools/continue_ackermann_retention.py`，按med_b→hard_a→hard_b逐级、每级s0/s1/s2顺序续训20次更新（32环境），每个新模型均做16环境3500步clean+三攻击种子评估，并导出clean及攻击视频/俯视轨迹/CSV和训练奖励图；任一种子未通过该级门禁则不进入更强攻击，失败模型不替代该种子的安全父模型。训练完成标记绑定父/子模型哈希、原始训练日志、更新次数、参数实际变化及实际优化器lr；完成训练可复用，不盲目重跑部分失败训练。
- 配置修复：从原env.yaml保留全部算法参数和已验证空间/安全配置，显式catchup_action=0.4；关闭retention_curriculum和专家接管，关闭依赖累计update_count的BC/KL课程，使用原课程终值BC=4、KL=0.8；攻击profile按阶段增加预算，阶段内strength=1、clean_fraction=0.5、warmup/ramp=0。父模型实际actor lr均5e-7，本次配置显式与之相同，并在训练后校验恢复优化器与配置一致，不再把1e-6误报为实际值。AMS权重沿用父模型0.1，未擅自改用其他实验的0.02。
- med_b攻击预算：FDI_pos=2.0、FDI_acc=0.5、DoS=0.10；hard_a=2.75/0.75/0.12；hard_b=4.0/1.3/0.18。学习种子42/3407/777，验证env/attack种子对仍(42,3407)/(3407,9917)/(777,17011)。新训练不重新执行已完成的clean至med_a训练/评估；各级新模型仍必须重新做clean回归。
- 启动排障发现并修复通用入口问题：`run_medium_algorithm_comparison.sh`原先`rm -rf RESULT_ROOT`删除放在该目录内的resume_runs.txt，且缺少resume文件时静默跳过`--resume`。20:48首次新启动的实际argv缺少`--resume`，已立即停止`ams-retention-train-20261026.service`及其cgroup；保存打开但被删除的日志至`verified_continuation_20261026/startup_failure_unlinked.log`，并写`invalid_startup.json`明确该尝试无效、禁止promotion。其未完成新run `2026-10-26_20-48-44_verified_continuation_20261026_s0_med_b_happo_meta_ams`仅有model_0/model_best，无model_final，保留作诊断；没有将其当作续训结果。
- 实际修改 `run_medium_algorithm_comparison.sh`：不再清空结果目录；保留预先准备的request/resume文件，已有manifest或training/runs.txt则拒绝覆盖；显式指定的resume文件缺失/空文件时报错退出，不再从头训练。新增 `scripts/tools/test_retention_continuation.py`：6项CPU测试通过，包括原始训练协议/学习率/模型变化审计、失败停止更强阶段、真实shell目录准备保留resume、缺resume文件必须退出；`bash -n scripts/tools/run_medium_algorithm_comparison.sh`通过。
- 正式新命令：`/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/continue_ackermann_retention.py --reference logs/ackermann_retention_recovery/catchup_repair_20261025 --output logs/ackermann_retention_recovery/verified_continuation_20261026_r1 --updates 20 --render`；同参数`--preflight-only`已通过。服务启动：`systemd-run --user --unit=ams-retention-train-20261026-r1 --property=WorkingDirectory=/home/cnc/SSD_1T/xzw/IsaacLab-main --property=KillMode=control-group --property=TimeoutStopSec=30 --property=RuntimeMaxSec=12h <上述命令>`。
- 已核查正式Isaac PID3202180实际argv含`--resume --load_run verified_continuation_20261026_r1_s0_med_b_parent --checkpoint model_final.pt`；20:51:14开始s0 med_b，当前训练CSV已有10行，update从父模型345推进至355，expert=0、attack_enabled=1、strength=1、clean_fraction=0.5，最新action_attack_delta_abs_mean=0.0488803、BC_loss=0.00364491、reference_KL=0.223996、AMS_Q_loss=0.359618。训练仍在进行，尚无本轮新门禁结论。
- 正式输出 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1`，含continuation_manifest.json/source_snapshot/训练请求与实际日志。s0 run=`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-10-26_20-51-32_verified_continuation_20261026_r1_s0_med_b_happo_meta_ams`。driver SHA256=535c1d9b56bd038e529f0739583d540db5f2b5c1aedd4d7c42747f4f52d24f66；修复后shell SHA256=a86285d10c67123debf088d3a4bfb75747ab16430df6d791934780b03ec99bd8。
- 下一步：s0训练结束后先核对有效更新与攻击数据，再进入严格clean/med_b验证和证据导出；随后s1/s2，三者通过才晋级hard_a/hard_b。当前自动队列已包含全部剩余三级，无需手动衔接；若门禁失败则保留结果并停止更强级别。

### 2026-10-26T20:54 s0 med_b真实续训审计通过，已进入严格clean

- 正式服务MainPID3202098，active/running。s0已完成20次HAPPO更新，计数345→365；模型参数确实变化，实际actor_lr=5e-7与配置一致，全部20行训练日志满足expert=0/attack_enabled=1/strength=1/clean_fraction=0.5，action_attack_delta_abs_mean最大0.0761453。`candidate_0/med_b/training_complete.json`已生成，训练与打包exit=0。
- 新checkpoint SHA256=d96f8923e434bb1785ab64fa76423d6a6899f2fc79edf852a7d17f99dda00781；父模型SHA256=69af5df579c2f5c0f16adf7772f5edaa1c108a32e1767c89e32a06eedddbe50c；训练CSV SHA256=41d4ef0b6286eebd66785bc269b2035439140ac5937dd753ad42fa57171a7e6c。模型路径为 `logs/rsl_rl/platoon_happo/2026-10-26_20-51-32_verified_continuation_20261026_r1_s0_med_b_happo_meta_ams/model_final.pt`。
- 20:52:58已自动转入该新模型的严格clean评估，输出 `verified_continuation_20261026_r1/candidate_0/med_b/validation/clean_env42/`。这是新训练已经完成后的验证阶段，尚无clean或med_b门禁结论；s1/s2和后续hard_a/hard_b仍按原门禁顺序排队。

### 2026-10-26T20:48:26 AMS后续实验

- 启动修复分支真实续训med_b→hard_a→hard_b，三种子逐级门禁；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026`。

### 2026-10-26T20:51:14 AMS后续实验

- 启动修复分支真实续训med_b→hard_a→hard_b，三种子逐级门禁；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1`。

### 2026-10-26T20:52:58 AMS后续实验

- retention s0/med_b真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_0/med_b`。

### 2026-10-26T21:32:16 AMS后续实验

- retention s0/med_b续训后门禁passed=True，clean=True，attack_min=1.0；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_0/med_b`。

### 2026-10-26T21:33:57 AMS后续实验

- retention s1/med_b真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_1/med_b`。

### 2026-10-26T22:13:26 AMS后续实验

- retention s1/med_b续训后门禁passed=True，clean=True，attack_min=1.0；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_1/med_b`。

### 2026-10-26T22:15:07 AMS后续实验

- retention s2/med_b真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_2/med_b`。

### 2026-10-26T22:54:15 AMS后续实验

- retention s2/med_b续训后门禁passed=True，clean=True，attack_min=1.0；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_2/med_b`。

### 2026-10-26T22:55:56 AMS后续实验

- retention s0/hard_a真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_0/hard_a`。

### 2026-10-26T23:35:15 AMS后续实验

- retention s0/hard_a续训后门禁passed=True，clean=True，attack_min=1.0；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_0/hard_a`。

### 2026-10-26T23:36:55 AMS后续实验

- retention s1/hard_a真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_1/hard_a`。

### 2026-10-27T00:16:23 AMS后续实验

- retention s1/hard_a续训后门禁passed=True，clean=True，attack_min=1.0；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_1/hard_a`。

### 2026-10-27T00:18:04 AMS后续实验

- retention s2/hard_a真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_2/hard_a`。

### 2026-10-27T00:57:32 AMS后续实验

- retention s2/hard_a续训后门禁passed=True，clean=True，attack_min=1.0；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_2/hard_a`。

### 2026-10-27T00:59:12 AMS后续实验

- retention s0/hard_b真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_0/hard_b`。

### 2026-10-27T01:38:53 AMS后续实验

- retention s0/hard_b续训后门禁passed=True，clean=True，attack_min=0.875；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_0/hard_b`。

### 2026-10-27T01:40:36 AMS后续实验

- retention s1/hard_b真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_1/hard_b`。

### 2026-10-27T02:20:08 AMS后续实验

- retention s1/hard_b续训后门禁passed=True，clean=True，attack_min=0.8125；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_1/hard_b`。

### 2026-10-27T02:21:49 AMS后续实验

- retention s2/hard_b真实续训完成，updates=20，实际actor_lr=5e-07，expert=0/strength=1/clean_fraction=0.5及参数更新核验通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_2/hard_b`。

### 2026-10-27T03:01:33 AMS后续实验

- retention s2/hard_b续训后门禁passed=True，clean=True，attack_min=0.9375；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1/candidate_2/hard_b`。

### 2026-10-27T18:26 进度复核：修复续训分支三种子通过hard_b

- 用户询问当前进度。本轮读取systemd/journal、ps/nvidia-smi、门禁及原始产物，并以 `continue_ackermann_retention.audit_training()` 重新审计9个训练块、`ackermann_validation.load_verified()`核验36份原始评估、SHA256核验18份渲染证据包；除追加本记录外未修改文件，未启动新实验。
- `ams-retention-train-20261026-r1.service`于03:01:33正常结束，当前inactive/dead、Result=success、ExecMainStatus=0、MainPID=0，followup_status.json.state=complete；没有Isaac训练/评估进程，GPU compute仅ToDesk_Session384MiB。服务CPU累计时间不是运行时长或超时证据，本次正常完成。
- s0/s1/s2均完成med_b→hard_a→hard_b，每级20次HAPPO更新，每条链345→365→385→405，总计9个有效续训块/180次更新。实际actor_lr=5e-7，所有新模型参数有变化，训练日志expert=0、strength=1、clean_fraction=0.5且非零攻击扰动；原始训练CSV与模型哈希均匹配training_complete.json。
- med_b、hard_a三个分支各自三攻击种子的全车队累计完成率均100%，所有阶段clean均100%；全部36份评估collision/road_any/road_termination均0，攻击评估保持无专家、完整强度、零clean混入及16环境/3500步。
- hard_b详细结果（每项为三个攻击评估种子的累计完成率）：s0=0.875/0.9375/0.875，mean=0.8958333333、worst=0.875；s1=0.9375/0.8125/0.875，mean=0.875、worst=0.8125；s2=0.9375/0.9375/0.9375，mean=worst=0.9375。九项合并均值0.9027777778，全局最差0.8125；三个hard_b门禁均passed=true，原0.50最低完成率门限未降低。累计完成率不同于同一时刻的驻留完成率，不能将hard_b记成100%驻留或始终保持完成。
- 产物根目录 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1`：36份完整验证、18套视频/真实轨迹/俯视图和9张训练奖励曲线均齐备。最终hard_b模型为：s0 `logs/rsl_rl/platoon_happo/2026-10-27_00-57-50_verified_continuation_20261026_r1_s0_hard_b_happo_meta_ams/model_final.pt`（SHA256 d032547856825ad4014ee435cc36f891c646c071987d89355db9dd39966491a0）；s1 `2026-10-27_01-39-11_verified_continuation_20261026_r1_s1_hard_b_happo_meta_ams/model_final.pt`（524e5f091c79b58a71d61b4af496e2a31c0bd03dd8f0cbaf8ca3830fdc9bcaa5）；s2 `2026-10-27_02-20-25_verified_continuation_20261026_r1_s2_hard_b_happo_meta_ams/model_final.pt`（d2464b00286adb60a7d604ae3ef1417509671fb6e3c7ad4e6ad9618cbfac9aa0），后两者同在logs/rsl_rl/platoon_happo/下。
- 范围/下一步：当前已完成修复分支本次三阶段续训与逐级门禁，不等于整个论文实验全部完成。此前低强度验证使用的是较早checkpoint；应先对三个最终hard_b模型补齐统一的全部攻击强度回归矩阵，再推进尚缺的六算法统一协议/多训练种子对照与统计。不能直接将新分支90.28%与旧主实验75%作严格因果比较，协议/配置和训练链需统一核对。当前没有后续任务在后台运行。

### 2026-10-27T18:41:39 AMS后续实验

- 用户授权“继续进行下一步补齐和训练”。复核实际磁盘、训练/评估进程及systemd：前一轮verified_continuation_20261026_r1已完成，服务inactive/success，无冲突Isaac任务，数据盘可用356GiB。三个最终hard_b模型及其12份原始clean/hard_b评估、6套视频均重新核对哈希/源码/命令，协议一致，可复用；其余7个攻击强度需补63份评估、21套视频。
- 六算法缺口：旧ams_verified_matrix_repair_20261003仅旧协议一组训练链；已完成的orthogonal s0/s1/s2为四格消融，不等于六算法。旧协议lateral_turn_gain=0.30、centerline_turn_gain=0.25、d_drop=1.50、pair3/pair4=1.0，与当前0.32/0.28/1.45/0.85/0.90及retention设置不同，不能直接拼接或给基线恢复攻击训练后的AMS模型。本次新增统一协议六算法×3个后续训练种子队列，旧训练/评估均保留。
- 新增scripts/tools/run_ackermann_completion_queue.py（SHA256 dc3e00a9398fc63c921ea9a5b5bfa7d7b96a28a037b3717392e7d75d0f7a5cd0）和test_ackermann_completion_queue.py；未改动核心算法/物理实现/论文/门禁。队列先完成三个固定最终模型的全强度回归；随后从共同真实轨迹BC checkpoint model_bc.pt（SHA256 6b8c15acf17b7e250a65fc85609c304ba2cc14f8d0a65d8f34b4188f6e801615）分别训练mappo/happo_no_meta/happo_meta_ams/harl_mappo_shared/harl_haa2c/harl_hatrpo，每方法seed=42/3407/777。三个种子指同一BC暖启动条件下的独立适应训练，不是三个独立BC训练。
- 新对照固定每攻击级40次更新、每级最多1个候选，32训练/16评估环境，actor_lr=5e-7，专家0、完整阶段攻击强度1、训练clean_fraction=0.5、BC=4、KL=0.8；AMS沿用当前配置weight=0.1/warmup=50/ramp=100，无额外参数搜索。先严格clean，不通过仅允许无攻击clean能力获取（最多240更新）；攻击阶段每个候选clean+3攻击种子通过才晋级。失败候选不作父模型，最终矩阵保留last accepted模型和停止级别；固定模型全强度评估不代表更强训练晋级。
- 新增训练审计检查实际更新计数、actor参数变化、恢复优化器学习率、保存配置及方法开关、所有训练行真实攻击/专家比例、BC参考策略身份；部分训练若不能明确验证完成则停止诊断，禁止盲目重训。延续共享GPU锁/进程检查/进程组超时清理，运行中发现源码漂移立即阻止后续启动。新增结果按训练种子均值统计样本标准差，不将9个评估种子当9个独立训练样本。
- 验证：新增6项测试、已有retention continuation 6项、followup 11项均通过；preflight以完整原始文件+语义命令核验复用12评估/6证据包，通过且未运行仿真。一次预检后修正了已创建validation目录时视频缓存导入条件，旧预检元数据归档preflight_archive/20261027_184030，最终预检18:40:41通过。所有旧实验结果未覆盖。
- 输出根目录logs/ackermann_completion_20261027；queue_manifest.json、source_snapshot、uniform_six/s{0,1,2}/matrix_manifest.json已保存。即将启动ams-completion-20261027.service，命令：/home/cnc/Downloads/ENTER/bin/python3 scripts/tools/run_ackermann_completion_queue.py --reference /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_retention_recovery/verified_continuation_20261026_r1 --output /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027。当前尚未产生新训练或新回归指标；先运行s0/light回归，完成全部回归后自动接入六算法训练。

### 2026-10-27T18:41:44 AMS后续实验

- 启动最终模型回归→统一六算法三种子队列；复用12份严格评估和6套视频，新增63份评估/21套视频后进入全新统一BC起点训练；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027`。

### 2026-10-27T18:41:47 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/off`。

### 2026-10-27T18:43:54 AMS后续实验

- 启动确认：ams-completion-20261027.service于18:41:39进入active/running，MainPID=3282965；实际Isaac GPU评估PID=3283040，评估器完成环境初始化并开始执行s0最终hard_b模型的light（env_seed=42/attack_seed=3407）。日志确认16环境、5车、动作(16,5,2)、AMS K=8/radius=0.25/weight=0.1；18:43 GPU利用率29%、显存3243MiB，未发现冲突任务。不是新增训练已完成，也不是六算法训练已开始。
- 39份当前源码与source_snapshot逐一SHA256核对一致。新队列预检缓存12份评估/6套视频均可复用；最终回归还需实际运行63份低于hard_b攻击评估与21套视频。首个light尚未生成verified.json，不能提前报告通过率。正式对照训练及按训练种子统计已在同一服务顺序队列中，无需人工再衔接；clean/碰撞/道路/完成率门限未降低。
- 监控：systemctl --user status ams-completion-20261027.service；总状态logs/ackermann_completion_20261027/followup_status.json，总日志service.log。回归产物final_model_regressions/final_ladders/s{0,1,2}/；后续训练uniform_six/s{0,1,2}/；最终统计uniform_six/three_training_seed_summary.csv（尚未生成）。下一步保持单GPU顺序运行，等待最终模型回归；对照阶段任何不明确的训练/源码漂移/协议异常停止诊断，已完成产物不重跑、不覆盖。

### 2026-10-27T19:07:43 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/light`。

### 2026-10-27T19:33:24 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/easy_0`。

### 2026-10-27T19:59:12 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/easy_a`。

### 2026-10-27T20:24:50 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/easy_b`。

### 2026-10-27T20:50:37 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/med_a`。

### 2026-10-27T21:16:36 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/med_b`。

### 2026-10-27T21:42:23 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/hard_a`。

### 2026-10-27T21:42:24 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s0/hard_b`。

### 2026-10-27T21:42:27 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/off`。

### 2026-10-27T22:08:09 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/light`。

### 2026-10-27T22:34:00 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/easy_0`。

### 2026-10-27T22:59:44 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/easy_a`。

### 2026-10-27T23:25:24 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/easy_b`。

### 2026-10-27T23:51:17 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/med_a`。

### 2026-10-28T00:17:21 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/med_b`。

### 2026-10-28T00:43:04 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/hard_a`。

### 2026-10-28T00:43:05 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s1/hard_b`。

### 2026-10-28T00:43:07 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/off`。

### 2026-10-28T01:09:00 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/light`。

### 2026-10-28T01:34:42 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/easy_0`。

### 2026-10-28T02:00:41 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/easy_a`。

### 2026-10-28T02:26:30 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/easy_b`。

### 2026-10-28T02:52:19 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/med_a`。

### 2026-10-28T03:12:27 AMS后续实验

- 用户询问当前进度，检查时间2026-10-28 03:11–03:12 CST。本轮只做进度核查及追加记录，未修改源码/配置，未启动或重跑实验。命令包括systemctl --user show ams-completion-20261027.service、nvidia-smi、ps、service.log/evidence_export.log读取；通过ackermann_validation.load_verified()复核72份唯一原始评估，通过SHA256核验25套完整视频证据与39份当前源码/source_snapshot，均一致。
- 后台服务active/running，MainPID3282965，自2026-10-27 18:41:39连续运行；当前GPU PID3343315是play.py渲染进程（约6208MiB），正在s2/med_b生成真实变道视频，日志步数持续推进。没有超时重启/partial/superseded目录；数据盘剩余346GiB。
- 最终模型回归：s0、s1均完成clean至hard_b全部9级验证及视频，ladder_complete.json已生成；s2已完成clean和light/easy_0/easy_a/easy_b/med_a/med_b全部数值验证，med_b视频进行中，hard_a的3评估及视频尚未执行；s2 hard_b的3评估及视频已经从同checkpoint历史验证复用，待export走到hard_b时生成本次汇总门禁。
- 计数：新增评估60/63（95.24%），另复用12份，总72/75；新增视频/轨迹/俯视图证据19/21，另复用6套，总25/27。尚缺3份s2 hard_a评估及2套证据（s2 med_b正在生成，s2 hard_a待执行）。三个分支的clean至med_b完成率全部100%，s0/s1 hard_a各3种子也均100%；72份已核验评估的collision/road_any/road_termination最大值均0。
- 复用hard_b结果未改变：s0均值0.8958333333、最差0.875；s1均值0.875、最差0.8125；s2均值/最差0.9375；全部9评估合并均值0.9027777778。此处为全车队累计完成率，不是持续驻留100%，也不是新增hard_b训练效果。
- 六算法统一协议三训练种子尚未开始：uniform_six下training_request.json=0、training_complete.json=0；当前GPU用于回归证据录制。固定模型回归完成后，同一服务自动进入统一BC起点的六算法流程，先严格clean，门禁通过才开始攻击训练，失败不晋级；无需另起并发进程。
- 产物根目录/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027；回归final_model_regressions/final_ladders/s{0,1,2}/；状态followup_status.json；后续训练uniform_six/s{0,1,2}/。下一步继续现有队列完成s2 med_b视频→hard_a评估/视频→全强度汇总→六算法统一BC clean及训练；当前无需修复或重启。

### 2026-10-28T03:18:17 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/med_b`。

### 2026-10-28T03:44:01 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/hard_a`。

### 2026-10-28T03:44:02 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions/final_ladders/s2/hard_b`。

### 2026-10-28T03:44:15 AMS后续实验

- 三个最终模型全强度回归完成，all_gates_passed=True；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/final_model_regressions`。对照训练从独立BC起点开始，不使用任何失败回归模型晋级。

### 2026-10-28T04:00:50 AMS后续实验

- 统一对照s0/mappo/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/light_r1`。

### 2026-10-28T04:33:24 AMS后续实验

- 六算法 `mappo` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/light_r1`。

### 2026-10-28T04:35:58 AMS后续实验

- 统一对照s0/mappo/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/easy_0_r1`。

### 2026-10-28T05:08:41 AMS后续实验

- 六算法 `mappo` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/easy_0_r1`。

### 2026-10-28T05:11:16 AMS后续实验

- 统一对照s0/mappo/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/easy_a_r1`。

### 2026-10-28T05:43:58 AMS后续实验

- 六算法 `mappo` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/easy_a_r1`。

### 2026-10-28T05:46:36 AMS后续实验

- 统一对照s0/mappo/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/easy_b_r1`。

### 2026-10-28T06:19:04 AMS后续实验

- 六算法 `mappo` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/easy_b_r1`。

### 2026-10-28T06:21:37 AMS后续实验

- 统一对照s0/mappo/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/med_a_r1`。

### 2026-10-28T06:54:06 AMS后续实验

- 六算法 `mappo` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/med_a_r1`。

### 2026-10-28T06:56:40 AMS后续实验

- 统一对照s0/mappo/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/med_b_r1`。

### 2026-10-28T07:29:10 AMS后续实验

- 六算法 `mappo` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/med_b_r1`。

### 2026-10-28T07:31:45 AMS后续实验

- 统一对照s0/mappo/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/hard_a_r1`。

### 2026-10-28T08:04:14 AMS后续实验

- 六算法 `mappo` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/hard_a_r1`。

### 2026-10-28T08:06:48 AMS后续实验

- 统一对照s0/mappo/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/hard_b_r1`。

### 2026-10-28T08:39:42 AMS后续实验

- 六算法 `mappo` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/mappo/hard_b_r1`。

### 2026-10-28T08:47:37 AMS后续实验

- mappo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/off`。

### 2026-10-28T09:13:51 AMS后续实验

- mappo 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/light`。

### 2026-10-28T09:40:11 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/easy_0`。

### 2026-10-28T10:06:39 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/easy_a`。

### 2026-10-28T10:33:00 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/easy_b`。

### 2026-10-28T10:59:22 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/med_a`。

### 2026-10-28T11:25:55 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/med_b`。

### 2026-10-28T11:25:58 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/hard_a`。

### 2026-10-28T11:52:24 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/mappo/hard_b`。

### 2026-10-28T12:08:48 AMS后续实验

- 统一对照s0/happo_no_meta/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/light_r1`。

### 2026-10-28T12:41:11 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/light_r1`。

### 2026-10-28T12:43:45 AMS后续实验

- 统一对照s0/happo_no_meta/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/easy_0_r1`。

### 2026-10-28T13:16:22 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/easy_0_r1`。

### 2026-10-28T13:18:55 AMS后续实验

- 统一对照s0/happo_no_meta/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/easy_a_r1`。

### 2026-10-28T13:51:20 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/easy_a_r1`。

### 2026-10-28T13:53:56 AMS后续实验

- 统一对照s0/happo_no_meta/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/easy_b_r1`。

### 2026-10-28T14:26:31 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/easy_b_r1`。

### 2026-10-28T14:29:05 AMS后续实验

- 统一对照s0/happo_no_meta/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/med_a_r1`。

### 2026-10-28T15:01:46 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/med_a_r1`。

### 2026-10-28T15:04:21 AMS后续实验

- 统一对照s0/happo_no_meta/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/med_b_r1`。

### 2026-10-28T15:36:54 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/med_b_r1`。

### 2026-10-28T15:39:27 AMS后续实验

- 统一对照s0/happo_no_meta/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/hard_a_r1`。

### 2026-10-28T16:12:06 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/hard_a_r1`。

### 2026-10-28T16:14:44 AMS后续实验

- 统一对照s0/happo_no_meta/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/hard_b_r1`。

### 2026-10-28T16:47:17 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_no_meta/hard_b_r1`。

### 2026-10-28T16:55:05 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/off`。

### 2026-10-28T17:21:22 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/light`。

### 2026-10-28T17:48:10 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/easy_0`。

### 2026-10-28T18:14:32 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/easy_a`。

### 2026-10-28T18:40:56 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/easy_b`。

### 2026-10-28T18:44:23 AMS后续实验

- 用户询问进度；核查时间2026-10-28 18:41–18:44 CST。本轮只读检查并追加记录，未修改源码/配置、未重启服务或重跑实验。执行systemctl --user show ams-completion-20261027.service、nvidia-smi、ps、df、service.log及分支状态读取；读取回归complete.json/27行汇总CSV，核验本次新增s2 hard_a三份原始评估和s2 med_b/hard_a两套视频产物哈希，并用run_ackermann_completion_queue.audit_new_training()独立重新审计全部16个新训练块。
- 原三个最终Meta-AMS模型全强度回归于03:44完成，complete.json.all_gates_passed=true，27/27模型-强度门禁通过；75份唯一verified评估（63新增+12复用）、27套完整视频/轨迹/俯视图（21新增+6复用）已齐。三个分支clean/light/easy_0/easy_a/easy_b/med_a/med_b/hard_a完成率均100%，hard_b仍为s0均值89.5833%、s1 87.5%、s2 93.75%，九评估总体均值90.2778%、最差81.25%；全队累计完成率，不能解释为始终驻留完成。此前72份及本轮新增3份验证的碰撞/道路越界/道路终止均0。
- 六算法统一协议已于03:44接续开始，目前seed index=0（训练seed42）。MAPPO和happo_no_meta各已完成light→hard_b的8块×40更新=320真实更新，共16块/640更新；本轮重审计全部通过，确认实际参数变化、5e-7优化器学习率、方法开关、共同BC参考、专家0/真实攻击strength1/clean_fraction0.5和训练CSV哈希。16张训练奖励曲线已存在。
- MAPPO：light至med_b各攻击seed完成率100%，hard_a最差87.5%，hard_b候选三seed=37.5%/37.5%/37.5%，未达到原50%最差seed门槛，clean通过且无碰撞/越界。候选拒绝晋级，保留hard_a模型SHA256 36e2b4de6c273d2f1a8f87757e7fc57eec9d4e028d0298a56da49e0bef35f55c；该保留模型全强度评估/9级视频已完成，其hard_b评估=50%/43.75%/50%、均值47.9167%，门禁仍不通过。区分拒绝候选37.5%与保留模型47.9167%，不可混报。
- happo_no_meta：light至med_b各seed100%，hard_a最差93.75%，hard_b候选=43.75%/50%/75%，最差43.75%未达到50%门槛；clean通过且无碰撞/越界。保留hard_a模型SHA256 617d93c4ef6683b4c4c03e6b50250528ecc324eab09c81a6de8f56ed6e46db00，当前在该模型的最终全强度评估med_a/env42/attack3407，off至easy_b五级完整证据已完成。其最终hard_b评估尚未执行，不将候选数据当最终模型结果。
- 服务仍active/running（MainPID3282965）；GPU PID3451757为eval_happo_platoon.py，显存约2826MiB，日志步数持续推进；无partial/superseded目录、无停止报错，39份当前源码与快照哈希一致，磁盘剩余326GiB。根followup_status.json.time=03:44是进入s0矩阵的时间，不表示停滞；实时细项在uniform_six/s0/followup_status.json及service.log（当前18:40:56开始med_a）。
- 待完成：当前happo_no_meta全强度评估与视频；随后s0的happo_meta_ams、harl_mappo_shared、harl_haa2c、harl_hatrpo；s1/s2全部六算法尚未启动。以上保持同一自动队列，失败候选回退，不降低clean/完成率/安全门禁。新统一对照的Meta-AMS臂尚未训练，不能直接把旧修复分支90.28%当作新公平对照的Meta-AMS成绩。
- 产物根目录/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027；已完成回归表final_model_regressions/six_algorithm_attack_matrix.csv及三张heatmap；新训练uniform_six/s0/stages/{mappo,happo_no_meta}/；最终对照评估uniform_six/s0/final_ladders/。下一步让现有评估正常完成后自动进入Meta-AMS s0同BC起点训练，本轮不并发起新任务。

### 2026-10-28T19:07:23 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/med_a`。

### 2026-10-28T19:33:46 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/med_b`。

### 2026-10-28T19:33:48 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/hard_a`。

### 2026-10-28T20:00:23 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_no_meta/hard_b`。

### 2026-10-28T20:16:50 AMS后续实验

- 统一对照s0/happo_meta_ams/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/light_r1`。

### 2026-10-28T20:48:41 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/light_r1`。

### 2026-10-28T20:51:30 AMS后续实验

- 统一对照s0/happo_meta_ams/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-28T21:23:30 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-28T21:26:20 AMS后续实验

- 统一对照s0/happo_meta_ams/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-28T21:58:03 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-28T22:00:52 AMS后续实验

- 统一对照s0/happo_meta_ams/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-28T22:32:43 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-28T22:35:33 AMS后续实验

- 统一对照s0/happo_meta_ams/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/med_a_r1`。

### 2026-10-28T23:07:24 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/med_a_r1`。

### 2026-10-28T23:10:13 AMS后续实验

- 统一对照s0/happo_meta_ams/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/med_b_r1`。

### 2026-10-28T23:42:14 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/med_b_r1`。

### 2026-10-28T23:45:04 AMS后续实验

- 统一对照s0/happo_meta_ams/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-29T00:16:51 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-29T00:19:40 AMS后续实验

- 统一对照s0/happo_meta_ams/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-29T00:51:17 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-29T00:59:18 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/off`。

### 2026-10-29T01:25:05 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/light`。

### 2026-10-29T01:50:58 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/easy_0`。

### 2026-10-29T02:17:04 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/easy_a`。

### 2026-10-29T02:42:49 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/easy_b`。

### 2026-10-29T03:08:36 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/med_a`。

### 2026-10-29T03:34:15 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/med_b`。

### 2026-10-29T04:00:02 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/hard_a`。

### 2026-10-29T04:00:04 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/happo_meta_ams/hard_b`。

### 2026-10-29T04:16:16 AMS后续实验

- 统一对照s0/harl_mappo_shared/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/light_r1`。

### 2026-10-29T04:48:44 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/light_r1`。

### 2026-10-29T04:51:07 AMS后续实验

- 统一对照s0/harl_mappo_shared/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/easy_0_r1`。

### 2026-10-29T05:23:34 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/easy_0_r1`。

### 2026-10-29T05:25:55 AMS后续实验

- 统一对照s0/harl_mappo_shared/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/easy_a_r1`。

### 2026-10-29T05:58:32 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/easy_a_r1`。

### 2026-10-29T06:00:57 AMS后续实验

- 统一对照s0/harl_mappo_shared/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/easy_b_r1`。

### 2026-10-29T06:33:40 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/easy_b_r1`。

### 2026-10-29T06:36:02 AMS后续实验

- 统一对照s0/harl_mappo_shared/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/med_a_r1`。

### 2026-10-29T07:08:31 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/med_a_r1`。

### 2026-10-29T07:10:55 AMS后续实验

- 统一对照s0/harl_mappo_shared/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/med_b_r1`。

### 2026-10-29T07:43:23 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/med_b_r1`。

### 2026-10-29T07:45:48 AMS后续实验

- 统一对照s0/harl_mappo_shared/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/hard_a_r1`。

### 2026-10-29T08:18:24 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.5；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/hard_a_r1`。

### 2026-10-29T08:20:48 AMS后续实验

- 统一对照s0/harl_mappo_shared/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/hard_b_r1`。

### 2026-10-29T08:53:31 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.0625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_mappo_shared/hard_b_r1`。

### 2026-10-29T09:01:14 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/off`。

### 2026-10-29T09:27:22 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/light`。

### 2026-10-29T09:53:46 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/easy_0`。

### 2026-10-29T10:20:04 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/easy_a`。

### 2026-10-29T10:46:24 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/easy_b`。

### 2026-10-29T11:12:56 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/med_a`。

### 2026-10-29T11:39:11 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/med_b`。

### 2026-10-29T11:39:13 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/hard_a`。

### 2026-10-29T12:05:41 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_mappo_shared/hard_b`。

### 2026-10-29T12:21:58 AMS后续实验

- 统一对照s0/harl_haa2c/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/light_r1`。

### 2026-10-29T12:27:32 AMS后续实验

- 用户询问当前进度，检查时间2026-10-29 12:26–12:27 CST。本轮只读核查及追加记录，未修改任何源码/配置，未重启服务或重复实验。执行systemctl --user show、nvidia-smi、ps、df、service.log/分支状态读取；重新审计自上轮后新增的17个训练块（Meta-AMS 8、shared-MAPPO 8、HAA2C 1），全部audit_new_training()结果与training_complete.json一致；另用load_verified()核验Meta-AMS最终模型25份原始评估、SHA256核验9套完整视频证据及39份源码/快照。
- ams-completion-20261027.service仍active/running，MainPID3282965。GPU PID3578880是HAA2C light_r1训练后的strict-clean评估器，显存约2822MiB，日志已经推进到2540步并继续运行；12:27尚无该clean verified.json，不能提前报告通过。没有partial/superseded目录，未发现队列中断；数据盘可用304GiB。根状态uniform_matrix/seed0的旧时间只是进入此大阶段的时间，实时进度读取uniform_six/s0/followup_status.json。
- 首组统一对照seed42目前完成4种方法全部8攻击阶段候选训练及最终9级评估/视频：mappo、happo_no_meta、happo_meta_ams、harl_mappo_shared。HAA2C作为第5种方法已完成light的40次真实更新，正在clean回归验证；HATRPO尚未开始。s1(seed3407)、s2(seed777)尚未启动。累计训练请求=完成块=33，共1320次真实更新；其中新17块审计确认参数变化、actor_lr=5e-7、方法开关、共同BC参考、专家0/攻击strength1/clean_fraction0.5等有效协议；当前32张奖励曲线，HAA2C light第33张将在本阶段验证结束后自动生成。
- 新统一协议Meta-AMS s0完成全部训练及最终全强度评估。最终模型为logs/rsl_rl/platoon_happo/2026-10-29_00-17-09_ackermann_completion_20261027_s0_happo_meta_ams_hard_b_r1_happo_meta_ams/model_final.pt，SHA256 7956bb7960862b12cd26808cf4c526e5b85fe6dcb2779355cf502ff2ef234026。最终clean的累计/瞬时峰值/驻留峰值均1，light至hard_a各3攻击seed全车队累计完成率均100%；hard_b为100%/93.75%/87.5%，平均93.75%、最差87.5%，全部门禁通过。25份原始评估碰撞/road_any/road_termination最大值均0，9套真实变道视频/轨迹/俯视图哈希通过。该93.75%来自新同BC对照s0，区别于先前修复续训三分支总体90.28%，不可混用。
- 本组已完成方法的“最终门禁保留模型”hard_b结果（三评估seed）：MAPPO=50%/43.75%/50%，均值47.9167%、最差43.75%，评估门禁失败；无Meta HAPPO=56.25%/50%/62.5%，均值56.25%、最差50%，最终评估通过；Meta-AMS=100%/93.75%/87.5%，均值93.75%、最差87.5%，通过；shared-MAPPO=12.5%/0%/12.5%，均值8.3333%、最差0%，失败。
- 区分候选训练晋级与保留模型最终评估：MAPPO、无Meta HAPPO、shared-MAPPO的hard_b新候选都未通过，继续保留各自hard_a模型。无Meta HAPPO的新hard_b候选为43.75%/50%/75%，最差不足50%，但回退hard_a模型的hard_b最终评估最差达到50%，两者不是矛盾或重复训练。shared-MAPPO新hard_b候选为12.5%/6.25%/6.25%，与其保留模型最终数据也应分列。失败候选未替换安全模型，门槛未降低。
- 产物根目录/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027；已完成4方法最终评估位于uniform_six/s0/final_ladders/{mappo,happo_no_meta,happo_meta_ams,harl_mappo_shared}/；HAA2C当前目录uniform_six/s0/stages/harl_haa2c/light_r1/。旧修复分支三模型75份评估/27套视频的完整回归已完成，未改变。
- 下一步继续现有串行队列：HAA2C通过clean后做light三攻击种子验证及门禁；按门禁完成HAA2C与HATRPO，再进入s1/s2。当前数据只是首个共同BC起点适应训练种子的结果，完整三训练种子统计仍未形成，不能据此宣称最终显著最优。本轮无需代码修复或另起并发任务。

### 2026-10-29T12:54:22 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/light_r1`。

### 2026-10-29T12:56:54 AMS后续实验

- 统一对照s0/harl_haa2c/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/easy_0_r1`。

### 2026-10-29T13:29:18 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/easy_0_r1`。

### 2026-10-29T13:31:51 AMS后续实验

- 统一对照s0/harl_haa2c/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/easy_a_r1`。

### 2026-10-29T14:04:32 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/easy_a_r1`。

### 2026-10-29T14:07:05 AMS后续实验

- 统一对照s0/harl_haa2c/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/easy_b_r1`。

### 2026-10-29T14:39:25 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/easy_b_r1`。

### 2026-10-29T14:42:00 AMS后续实验

- 统一对照s0/harl_haa2c/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/med_a_r1`。

### 2026-10-29T15:14:28 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/med_a_r1`。

### 2026-10-29T15:17:00 AMS后续实验

- 统一对照s0/harl_haa2c/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/med_b_r1`。

### 2026-10-29T15:49:36 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/med_b_r1`。

### 2026-10-29T15:52:12 AMS后续实验

- 统一对照s0/harl_haa2c/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/hard_a_r1`。

### 2026-10-29T16:24:45 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/hard_a_r1`。

### 2026-10-29T16:27:18 AMS后续实验

- 统一对照s0/harl_haa2c/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/hard_b_r1`。

### 2026-10-29T16:59:58 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_haa2c/hard_b_r1`。

### 2026-10-29T17:07:42 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/off`。

### 2026-10-29T17:34:05 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/light`。

### 2026-10-29T18:00:35 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/easy_0`。

### 2026-10-29T18:26:54 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/easy_a`。

### 2026-10-29T18:53:25 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/easy_b`。

### 2026-10-29T19:19:53 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/med_a`。

### 2026-10-29T19:46:15 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/med_b`。

### 2026-10-29T19:46:17 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/hard_a`。

### 2026-10-29T20:12:49 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_haa2c/hard_b`。

### 2026-10-29T20:29:18 AMS后续实验

- 统一对照s0/harl_hatrpo/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_hatrpo/light_r1`。

### 2026-10-29T20:43:19 AMS后续实验

- 六算法 `harl_hatrpo` 阶段 `light_r1`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/stages/harl_hatrpo/light_r1`。

### 2026-10-29T20:43:19 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/off`。

### 2026-10-29T21:09:27 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/light`。

### 2026-10-29T21:35:43 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/easy_0`。

### 2026-10-29T22:01:56 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/easy_a`。

### 2026-10-29T22:28:19 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/easy_b`。

### 2026-10-29T22:54:47 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/med_a`。

### 2026-10-29T23:21:10 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/med_b`。

### 2026-10-29T23:47:28 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/hard_a`。

### 2026-10-30T00:14:05 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s0/final_ladders/harl_hatrpo/hard_b`。

### 2026-10-30T00:30:46 AMS后续实验

- 统一对照s1/mappo/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/light_r1`。

### 2026-10-30T01:03:20 AMS后续实验

- 六算法 `mappo` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/light_r1`。

### 2026-10-30T01:05:55 AMS后续实验

- 统一对照s1/mappo/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/easy_0_r1`。

### 2026-10-30T01:38:34 AMS后续实验

- 六算法 `mappo` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/easy_0_r1`。

### 2026-10-30T01:41:10 AMS后续实验

- 统一对照s1/mappo/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/easy_a_r1`。

### 2026-10-30T02:13:58 AMS后续实验

- 六算法 `mappo` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/easy_a_r1`。

### 2026-10-30T02:16:31 AMS后续实验

- 统一对照s1/mappo/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/easy_b_r1`。

### 2026-10-30T02:48:59 AMS后续实验

- 六算法 `mappo` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/easy_b_r1`。

### 2026-10-30T02:51:32 AMS后续实验

- 统一对照s1/mappo/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/med_a_r1`。

### 2026-10-30T03:24:03 AMS后续实验

- 六算法 `mappo` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/med_a_r1`。

### 2026-10-30T03:26:37 AMS后续实验

- 统一对照s1/mappo/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/med_b_r1`。

### 2026-10-30T03:59:13 AMS后续实验

- 六算法 `mappo` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/med_b_r1`。

### 2026-10-30T04:01:49 AMS后续实验

- 统一对照s1/mappo/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/hard_a_r1`。

### 2026-10-30T04:34:31 AMS后续实验

- 六算法 `mappo` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/hard_a_r1`。

### 2026-10-30T04:37:05 AMS后续实验

- 统一对照s1/mappo/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/hard_b_r1`。

### 2026-10-30T05:09:49 AMS后续实验

- 六算法 `mappo` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.4375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/mappo/hard_b_r1`。

### 2026-10-30T05:17:42 AMS后续实验

- mappo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/off`。

### 2026-10-30T05:43:53 AMS后续实验

- mappo 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/light`。

### 2026-10-30T06:10:11 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/easy_0`。

### 2026-10-30T06:36:34 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/easy_a`。

### 2026-10-30T07:02:58 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/easy_b`。

### 2026-10-30T07:29:23 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/med_a`。

### 2026-10-30T07:55:50 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/med_b`。

### 2026-10-30T07:55:53 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/hard_a`。

### 2026-10-30T08:22:23 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/mappo/hard_b`。

### 2026-10-30T08:38:52 AMS后续实验

- 统一对照s1/happo_no_meta/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/light_r1`。

### 2026-10-30T09:11:16 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/light_r1`。

### 2026-10-30T09:13:50 AMS后续实验

- 统一对照s1/happo_no_meta/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/easy_0_r1`。

### 2026-10-30T09:46:43 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/easy_0_r1`。

### 2026-10-30T09:49:20 AMS后续实验

- 统一对照s1/happo_no_meta/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/easy_a_r1`。

### 2026-10-30T10:21:50 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/easy_a_r1`。

### 2026-10-30T10:24:23 AMS后续实验

- 统一对照s1/happo_no_meta/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/easy_b_r1`。

### 2026-10-30T10:56:54 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/easy_b_r1`。

### 2026-10-30T10:59:27 AMS后续实验

- 统一对照s1/happo_no_meta/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/med_a_r1`。

### 2026-10-30T11:31:53 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/med_a_r1`。

### 2026-10-30T11:34:26 AMS后续实验

- 统一对照s1/happo_no_meta/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/med_b_r1`。

### 2026-10-30T12:07:00 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/med_b_r1`。

### 2026-10-30T12:09:32 AMS后续实验

- 统一对照s1/happo_no_meta/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/hard_a_r1`。

### 2026-10-30T12:42:11 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/hard_a_r1`。

### 2026-10-30T12:44:46 AMS后续实验

- 统一对照s1/happo_no_meta/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/hard_b_r1`。

### 2026-10-30T13:17:36 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.25；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_no_meta/hard_b_r1`。

### 2026-10-30T13:25:19 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/off`。

### 2026-10-30T13:51:29 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/light`。

### 2026-10-30T14:17:50 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/easy_0`。

### 2026-10-30T14:44:02 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/easy_a`。

### 2026-10-30T15:10:29 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/easy_b`。

### 2026-10-30T15:36:53 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/med_a`。

### 2026-10-30T16:03:08 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/med_b`。

### 2026-10-30T16:03:10 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/hard_a`。

### 2026-10-30T16:29:30 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_no_meta/hard_b`。

### 2026-10-30T16:45:57 AMS后续实验

- 统一对照s1/happo_meta_ams/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/light_r1`。

### 2026-10-30T17:17:40 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.8125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/light_r1`。

### 2026-10-30T17:20:28 AMS后续实验

- 统一对照s1/happo_meta_ams/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-30T17:52:18 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/easy_0_r1`。

### 2026-10-30T17:55:07 AMS后续实验

- 统一对照s1/happo_meta_ams/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-30T18:26:55 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/easy_a_r1`。

### 2026-10-30T18:29:45 AMS后续实验

- 统一对照s1/happo_meta_ams/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-30T19:01:28 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/easy_b_r1`。

### 2026-10-30T19:04:17 AMS后续实验

- 统一对照s1/happo_meta_ams/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/med_a_r1`。

### 2026-10-30T19:36:05 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/med_a_r1`。

### 2026-10-30T19:38:57 AMS后续实验

- 统一对照s1/happo_meta_ams/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/med_b_r1`。

### 2026-10-30T20:10:51 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/med_b_r1`。

### 2026-10-30T20:13:41 AMS后续实验

- 统一对照s1/happo_meta_ams/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-30T20:45:38 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/hard_a_r1`。

### 2026-10-30T20:48:25 AMS后续实验

- 统一对照s1/happo_meta_ams/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-30T21:20:43 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/happo_meta_ams/hard_b_r1`。

### 2026-10-30T21:28:33 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/off`。

### 2026-10-30T21:54:36 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/light`。

### 2026-10-30T22:20:27 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/easy_0`。

### 2026-10-30T22:46:18 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/easy_a`。

### 2026-10-30T23:12:06 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/easy_b`。

### 2026-10-30T23:37:59 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/med_a`。

### 2026-10-31T00:03:54 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/med_b`。

### 2026-10-31T00:29:59 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/hard_a`。

### 2026-10-31T00:30:02 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/happo_meta_ams/hard_b`。

### 2026-10-31T00:46:10 AMS后续实验

- 统一对照s1/harl_mappo_shared/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_mappo_shared/light_r1`。

### 2026-10-31T01:18:46 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `light_r1`，passed=False，clean_ok=True，攻击最低完成率=0.625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_mappo_shared/light_r1`。

### 2026-10-31T01:18:47 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/off`。

### 2026-10-31T01:45:24 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/light`。

### 2026-10-31T02:11:38 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/easy_0`。

### 2026-10-31T02:37:58 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/easy_a`。

### 2026-10-31T03:04:23 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/easy_b`。

### 2026-10-31T03:30:33 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/med_a`。

### 2026-10-31T03:57:09 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/med_b`。

### 2026-10-31T04:23:43 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/hard_a`。

### 2026-10-31T04:50:07 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_mappo_shared/hard_b`。

### 2026-10-31T05:06:39 AMS后续实验

- 统一对照s1/harl_haa2c/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/light_r1`。

### 2026-10-31T05:38:58 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/light_r1`。

### 2026-10-31T05:41:28 AMS后续实验

- 统一对照s1/harl_haa2c/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/easy_0_r1`。

### 2026-10-31T06:14:04 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=0.75；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/easy_0_r1`。

### 2026-10-31T06:16:36 AMS后续实验

- 统一对照s1/harl_haa2c/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/easy_a_r1`。

### 2026-10-31T06:49:15 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/easy_a_r1`。

### 2026-10-31T06:51:46 AMS后续实验

- 统一对照s1/harl_haa2c/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/easy_b_r1`。

### 2026-10-31T07:24:12 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_haa2c/easy_b_r1`。

### 2026-10-31T07:31:56 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/off`。

### 2026-10-31T07:58:33 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/light`。

### 2026-10-31T08:24:58 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/easy_0`。

### 2026-10-31T08:25:01 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/easy_a`。

### 2026-10-31T08:51:32 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/easy_b`。

### 2026-10-31T09:17:56 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/med_a`。

### 2026-10-31T09:44:25 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/med_b`。

### 2026-10-31T10:10:45 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/hard_a`。

### 2026-10-31T10:37:23 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_haa2c/hard_b`。

### 2026-10-31T10:53:51 AMS后续实验

- 统一对照s1/harl_hatrpo/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_hatrpo/light_r1`。

### 2026-10-31T11:07:46 AMS后续实验

- 六算法 `harl_hatrpo` 阶段 `light_r1`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/stages/harl_hatrpo/light_r1`。

### 2026-10-31T11:07:47 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/off`。

### 2026-10-31T11:34:17 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/light`。

### 2026-10-31T12:00:38 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/easy_0`。

### 2026-10-31T12:26:54 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/easy_a`。

### 2026-10-31T12:53:23 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/easy_b`。

### 2026-10-31T13:19:41 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/med_a`。

### 2026-10-31T13:46:12 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/med_b`。

### 2026-10-31T14:12:34 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/hard_a`。

### 2026-10-31T14:38:53 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s1/final_ladders/harl_hatrpo/hard_b`。

### 2026-10-31T14:55:49 AMS后续实验

- 统一对照s2/mappo/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/light_r1`。

### 2026-10-31T15:28:25 AMS后续实验

- 六算法 `mappo` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/light_r1`。

### 2026-10-31T15:30:59 AMS后续实验

- 统一对照s2/mappo/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/easy_0_r1`。

### 2026-10-31T16:03:27 AMS后续实验

- 六算法 `mappo` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/easy_0_r1`。

### 2026-10-31T16:06:00 AMS后续实验

- 统一对照s2/mappo/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/easy_a_r1`。

### 2026-10-31T16:38:50 AMS后续实验

- 六算法 `mappo` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/easy_a_r1`。

### 2026-10-31T16:41:24 AMS后续实验

- 统一对照s2/mappo/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/easy_b_r1`。

### 2026-10-31T17:14:13 AMS后续实验

- 六算法 `mappo` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/easy_b_r1`。

### 2026-10-31T17:16:48 AMS后续实验

- 统一对照s2/mappo/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/med_a_r1`。

### 2026-10-31T17:49:22 AMS后续实验

- 六算法 `mappo` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/med_a_r1`。

### 2026-10-31T17:51:54 AMS后续实验

- 统一对照s2/mappo/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/med_b_r1`。

### 2026-10-31T18:24:49 AMS后续实验

- 六算法 `mappo` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/med_b_r1`。

### 2026-10-31T18:27:25 AMS后续实验

- 统一对照s2/mappo/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/hard_a_r1`。

### 2026-10-31T19:00:08 AMS后续实验

- 六算法 `mappo` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/hard_a_r1`。

### 2026-10-31T19:02:43 AMS后续实验

- 统一对照s2/mappo/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/hard_b_r1`。

### 2026-10-31T19:35:10 AMS后续实验

- 六算法 `mappo` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.25；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/mappo/hard_b_r1`。

### 2026-10-31T19:42:53 AMS后续实验

- mappo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/off`。

### 2026-10-31T20:09:11 AMS后续实验

- mappo 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/light`。

### 2026-10-31T20:35:23 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/easy_0`。

### 2026-10-31T21:01:39 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/easy_a`。

### 2026-10-31T21:27:57 AMS后续实验

- mappo 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/easy_b`。

### 2026-10-31T21:54:28 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/med_a`。

### 2026-10-31T22:20:44 AMS后续实验

- mappo 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/med_b`。

### 2026-10-31T22:20:46 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/hard_a`。

### 2026-10-31T22:47:14 AMS后续实验

- mappo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/mappo/hard_b`。

### 2026-10-31T23:03:32 AMS后续实验

- 统一对照s2/happo_no_meta/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/light_r1`。

### 2026-10-31T23:36:10 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/light_r1`。

### 2026-10-31T23:38:43 AMS后续实验

- 统一对照s2/happo_no_meta/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/easy_0_r1`。

### 2026-11-01T00:11:18 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/easy_0_r1`。

### 2026-11-01T00:13:49 AMS后续实验

- 统一对照s2/happo_no_meta/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/easy_a_r1`。

### 2026-11-01T00:46:09 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/easy_a_r1`。

### 2026-11-01T00:48:46 AMS后续实验

- 统一对照s2/happo_no_meta/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/easy_b_r1`。

### 2026-11-01T01:21:16 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/easy_b_r1`。

### 2026-11-01T01:23:52 AMS后续实验

- 统一对照s2/happo_no_meta/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/med_a_r1`。

### 2026-11-01T01:56:11 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/med_a_r1`。

### 2026-11-01T01:58:45 AMS后续实验

- 统一对照s2/happo_no_meta/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/med_b_r1`。

### 2026-11-01T02:31:20 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/med_b_r1`。

### 2026-11-01T02:33:57 AMS后续实验

- 统一对照s2/happo_no_meta/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/hard_a_r1`。

### 2026-11-01T03:06:36 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/hard_a_r1`。

### 2026-11-01T03:09:12 AMS后续实验

- 统一对照s2/happo_no_meta/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/hard_b_r1`。

### 2026-11-01T03:41:55 AMS后续实验

- 六算法 `happo_no_meta` 阶段 `hard_b_r1`，passed=False，clean_ok=True，攻击最低完成率=0.3125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_no_meta/hard_b_r1`。

### 2026-11-01T03:49:40 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/off`。

### 2026-11-01T04:16:18 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/light`。

### 2026-11-01T04:42:35 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/easy_0`。

### 2026-11-01T05:09:01 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/easy_a`。

### 2026-11-01T05:35:22 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/easy_b`。

### 2026-11-01T06:01:59 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/med_a`。

### 2026-11-01T06:28:33 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/med_b`。

### 2026-11-01T06:28:35 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/hard_a`。

### 2026-11-01T06:55:12 AMS后续实验

- happo_no_meta 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_no_meta/hard_b`。

### 2026-11-01T07:11:35 AMS后续实验

- 统一对照s2/happo_meta_ams/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/light_r1`。

### 2026-11-01T07:43:28 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/light_r1`。

### 2026-11-01T07:46:16 AMS后续实验

- 统一对照s2/happo_meta_ams/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/easy_0_r1`。

### 2026-11-01T08:18:12 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/easy_0_r1`。

### 2026-11-01T08:20:59 AMS后续实验

- 统一对照s2/happo_meta_ams/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/easy_a_r1`。

### 2026-11-01T08:52:44 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/easy_a_r1`。

### 2026-11-01T08:55:37 AMS后续实验

- 统一对照s2/happo_meta_ams/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/easy_b_r1`。

### 2026-11-01T09:27:31 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/easy_b_r1`。

### 2026-11-01T09:30:24 AMS后续实验

- 统一对照s2/happo_meta_ams/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/med_a_r1`。

### 2026-11-01T10:02:24 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/med_a_r1`。

### 2026-11-01T10:05:13 AMS后续实验

- 统一对照s2/happo_meta_ams/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/med_b_r1`。

### 2026-11-01T10:37:09 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/med_b_r1`。

### 2026-11-01T10:40:01 AMS后续实验

- 统一对照s2/happo_meta_ams/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/hard_a_r1`。

### 2026-11-01T11:11:55 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_a_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/hard_a_r1`。

### 2026-11-01T11:14:44 AMS后续实验

- 统一对照s2/happo_meta_ams/hard_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/hard_b_r1`。

### 2026-11-01T11:46:40 AMS后续实验

- 六算法 `happo_meta_ams` 阶段 `hard_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/happo_meta_ams/hard_b_r1`。

### 2026-11-01T11:54:28 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/off`。

### 2026-11-01T12:20:13 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/light`。

### 2026-11-01T12:45:55 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/easy_0`。

### 2026-11-01T13:11:53 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/easy_a`。

### 2026-11-01T13:37:46 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/easy_b`。

### 2026-11-01T14:03:43 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/med_a`。

### 2026-11-01T14:29:35 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/med_b`。

### 2026-11-01T14:55:38 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/hard_a`。

### 2026-11-01T14:55:41 AMS后续实验

- happo_meta_ams 固定模型全强度证据完成级别 `hard_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/happo_meta_ams/hard_b`。

### 2026-11-01T15:11:55 AMS后续实验

- 统一对照s2/harl_mappo_shared/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_mappo_shared/light_r1`。

### 2026-11-01T15:44:28 AMS后续实验

- 六算法 `harl_mappo_shared` 阶段 `light_r1`，passed=False，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_mappo_shared/light_r1`。

### 2026-11-01T15:44:29 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/off`。

### 2026-11-01T16:10:40 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/light`。

### 2026-11-01T16:36:59 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/easy_0`。

### 2026-11-01T17:03:28 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/easy_a`。

### 2026-11-01T17:29:50 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/easy_b`。

### 2026-11-01T17:56:20 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/med_a`。

### 2026-11-01T18:22:55 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/med_b`。

### 2026-11-01T18:49:24 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/hard_a`。

### 2026-11-01T19:15:58 AMS后续实验

- harl_mappo_shared 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_mappo_shared/hard_b`。

### 2026-11-01T19:32:17 AMS后续实验

- 统一对照s2/harl_haa2c/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/light_r1`。

### 2026-11-01T20:04:35 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `light_r1`，passed=True，clean_ok=True，攻击最低完成率=0.875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/light_r1`。

### 2026-11-01T20:07:11 AMS后续实验

- 统一对照s2/harl_haa2c/easy_0完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/easy_0_r1`。

### 2026-11-01T20:39:33 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_0_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/easy_0_r1`。

### 2026-11-01T20:42:07 AMS后续实验

- 统一对照s2/harl_haa2c/easy_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/easy_a_r1`。

### 2026-11-01T21:14:41 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.9375；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/easy_a_r1`。

### 2026-11-01T21:17:15 AMS后续实验

- 统一对照s2/harl_haa2c/easy_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/easy_b_r1`。

### 2026-11-01T21:49:55 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `easy_b_r1`，passed=True，clean_ok=True，攻击最低完成率=1.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/easy_b_r1`。

### 2026-11-01T21:52:27 AMS后续实验

- 统一对照s2/harl_haa2c/med_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/med_a_r1`。

### 2026-11-01T22:25:18 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `med_a_r1`，passed=True，clean_ok=True，攻击最低完成率=0.6875；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/med_a_r1`。

### 2026-11-01T22:27:50 AMS后续实验

- 统一对照s2/harl_haa2c/med_b完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/med_b_r1`。

### 2026-11-01T23:00:16 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `med_b_r1`，passed=True，clean_ok=True，攻击最低完成率=0.5625；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/med_b_r1`。

### 2026-11-01T23:02:49 AMS后续实验

- 统一对照s2/harl_haa2c/hard_a完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/hard_a_r1`。

### 2026-11-01T23:35:40 AMS后续实验

- 六算法 `harl_haa2c` 阶段 `hard_a_r1`，passed=False，clean_ok=True，攻击最低完成率=0.3125；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_haa2c/hard_a_r1`。

### 2026-11-01T23:43:33 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/off`。

### 2026-11-02T00:10:10 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `light`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/light`。

### 2026-11-02T00:36:42 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_0`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/easy_0`。

### 2026-11-02T01:03:17 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/easy_a`。

### 2026-11-02T01:29:39 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `easy_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/easy_b`。

### 2026-11-02T01:56:09 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/med_a`。

### 2026-11-02T01:56:12 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `med_b`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/med_b`。

### 2026-11-02T02:22:37 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_a`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/hard_a`。

### 2026-11-02T02:48:59 AMS后续实验

- harl_haa2c 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_haa2c/hard_b`。

### 2026-11-02T03:05:37 AMS后续实验

- 统一对照s2/harl_hatrpo/light完成40次真实更新；方法开关、BC参考、有效攻击及学习率审计通过；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_hatrpo/light_r1`。

### 2026-11-02T03:19:30 AMS后续实验

- 六算法 `harl_hatrpo` 阶段 `light_r1`，passed=False，clean_ok=False，攻击最低完成率=0.0；数据 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/stages/harl_hatrpo/light_r1`。

### 2026-11-02T03:19:31 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `off`，门禁passed=True；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/off`。

### 2026-11-02T03:45:39 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `light`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/light`。

### 2026-11-02T04:13:28 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_0`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/easy_0`。

### 2026-11-02T04:41:14 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/easy_a`。

### 2026-11-02T05:09:21 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `easy_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/easy_b`。

### 2026-11-02T05:37:23 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/med_a`。

### 2026-11-02T06:05:14 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `med_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/med_b`。

### 2026-11-02T06:33:16 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_a`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/hard_a`。

### 2026-11-02T07:01:15 AMS后续实验

- harl_hatrpo 固定模型全强度证据完成级别 `hard_b`，门禁passed=False；原始验证/视频/轨迹/绘图 `/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027/uniform_six/s2/final_ladders/harl_hatrpo/hard_b`。

### 2026-11-02T07:01:39 AMS后续实验

- 回归及统一六算法三种子队列完成；`/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027`。

### 2026-11-02T14:40:53 AMS后续实验

- 用户询问当前进度，检查时间2026-11-02 14:35起。本轮仅核查并追加记录，未修改源码/配置/模型/论文，未启动或重复实验。执行systemctl --user show/status、journalctl --user -u ams-completion-20261027.service、nvidia-smi、实际/proc可执行文件扫描、df及结果读取；对全部104个训练块重新调用audit_new_training()，对最终450份评估调用load_verified()，核验162套最终视频证据的全部产物哈希、39份当前源码/source_snapshot，并由原始数据重新计算每种子矩阵及54行多种子汇总的均值/样本标准差/最差seed/门禁通过数，全部一致。
- 整套计划队列于2026-11-02 07:01:39完成。s0完成时间2026-10-30 00:14:29；s1为2026-10-31 14:39:16；s2为2026-11-02 07:01:39。18个方法-训练种子流程全部结束，18份ladder_complete齐备，最终包含clean+8攻击强度的162个方法-种子-强度单元、450份单checkpoint评估、162套视频/真实轨迹/俯视图。共104个真实训练块、4160次更新、104张训练奖励曲线（s0 41、s1 30、s2 33）；强度晋级被门禁拒绝的分支按原协议提前停止，不代表所有方法都训练通过hard_b。
- 进程情况：当前没有Isaac训练/评估进程，也没有omni.telemetry遗留进程；GPU compute仅ToDesk。systemd显示failed/Result=timeout，但主进程于07:01:40正常退出（code=exited,status=0/SUCCESS），所有汇总已经生成。journal明确07:02:40是在stop-sigterm等候60秒超时后SIGKILL遥测子进程4089113（omni.telemetry），不是训练/评估运行超时或任务半途终止；ControlGroup为空、RuntimeMaxSec=infinity。保留此历史状态，不为消除failed标记重跑实验。数据盘剩余193GiB。
- 新统一协议Meta-AMS三个共同BC起点适应训练种子全部完成8级攻击训练并通过hard_b；每条链320次更新。最终clean至hard_a全部100%完成，hard_b各训练种子的三评估seed均值s0=93.75%、s1=93.75%、s2=91.6667%，总体均值93.0556%，训练seed均值的样本标准差1.2028个百分点；全部九次hard_b评估最差87.5%，hard_b通过数3/3。注意本结果来自uniform_six，区别于旧修复续训分支90.2778%。
- 最终保留模型hard_b汇总（均值；训练seed均值的样本标准差；全局最差评估；最终评估通过数）：Meta-AMS=93.0556%；1.2028pp；87.5%；3/3。无Meta HAPPO=47.2222%；7.8874pp；37.5%；1/3。MAPPO=46.5278%；2.4056pp；37.5%；0/3。HAA2C=27.0833%；20.5185pp；6.25%；0/3。共享参数MAPPO=2.7778%；4.8113pp；0%；0/3。HATRPO=0%；0pp；0%；0/3。450份最终评估collision/road_any/road_termination最大值均0；该表为全车队累计完成率，零安全违规不等于全部机动完成。
- 早停/回退解释必须随论文表保留：MAPPO与无Meta HAPPO三条链均在hard_b候选处拒绝，最终保留hard_a模型；无Meta HAPPO s0保留模型hard_b最终评估通过，但不是hard_b新候选晋级。共享MAPPO s0在hard_b拒绝、保留hard_a，s1/s2在light拒绝并回退同一BC；HAA2C s0在hard_b拒绝保留hard_a、s1在easy_b拒绝保留easy_a、s2在hard_a拒绝保留med_b；HATRPO三条链均在light训练后clean失败，保留同一BC。最终保留checkpoint唯一数分别MAPPO=3、无Meta HAPPO=3、Meta-AMS=3、共享MAPPO=2、HAA2C=3、HATRPO=1。因此HATRPO三个0值不是三个成功训练模型的独立鲁棒性证据；共享MAPPO两组也是相同BC回退值。所有方法共享同一次真实轨迹BC拟合，三种子独立性限定于后续适应训练，不能宣称三次从零或三次独立BC训练。
- 原修复分支三模型全强度回归（75评估/27证据）已于10-28完成；本轮新增六算法完整矩阵、每种子54行CSV及热图、54行跨训练种子汇总已齐。本协议下Meta-AMS的hard_b平均完成率最高且3/3通过，但本轮没有执行显著性检验，不能仅凭均值写“显著最优”或泛化为所有设置最优。
- 关键产物：logs/ackermann_completion_20261027/uniform_six/three_training_seed_summary.csv；uniform_six/s{0,1,2}/six_algorithm_attack_matrix.csv；各seed的completion_heatmap.png、completion_min_heatmap.png、return_heatmap.png；原始证据uniform_six/s{0,1,2}/final_ladders/<method>/<level>/。下一步转入论文表图、早停与训练预算披露、统计检验及证据整理；现有实验无需重跑。下次新任务启动时可完善遥测子进程退出回收，本轮已无进程需要清理。
- Meta-AMS最终模型：
- s0: /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-10-29_00-17-09_ackermann_completion_20261027_s0_happo_meta_ams_hard_b_r1_happo_meta_ams/model_final.pt；SHA256=7956bb7960862b12cd26808cf4c526e5b85fe6dcb2779355cf502ff2ef234026
- s1: /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-10-30_20-45-56_ackermann_completion_20261027_s1_happo_meta_ams_hard_b_r1_happo_meta_ams/model_final.pt；SHA256=40c2da2d04223faa0ae308c28e6e7bdd138bd02a0300ccd386a43339ffc76d05
- s2: /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/2026-11-01_11-12-13_ackermann_completion_20261027_s2_happo_meta_ams_hard_b_r1_happo_meta_ams/model_final.pt；SHA256=f0ef96464544314305b024099c951c4b819c45605170641fc5764b1e6dbe636e


### 2026-11-02T17:08:25+08:00 AMS 论文结果补充与训练种子统计核查

- 用户要求“继续进行下一步”。按上一轮记录进入论文表图、训练预算披露、统计检验和证据整理；本轮未启动新训练、仿真或评估，未改变算法、配置、门禁、checkpoint 或原始实验结果。
- 实际状态：uniform_six 六方法 × 三个共同 BC 起点适应种子的既定流程已完成。再次读取 systemd 状态为 failed/Result=timeout、ExecMainStatus=0、ControlGroup 为空；该历史状态来自主任务成功退出后的 telemetry 清理超时，不是实验中断。17:02 GPU compute 查询为空，进程检查无 train_happo/eval_happo/omni.telemetry。没有为消除 failed 标记重跑实验。
- 新增脚本：IsaacLab-main/scripts/tools/build_ackermann_paper_results.py 和 test_ackermann_paper_results.py；只读取已完成队列，在新目录构建结果补充包。收集时对450个最终评估调用 load_verified、核验104个训练块的 checkpoint/metrics CSV、39份队列源码快照、代表性视频产物；导出450评估、104训练阶段、18模型选择结果和4160条奖励记录。派生汇总54行×7项统计与原 three_training_seed_summary.csv 在绝对误差1e-14内一致。
- 当前主结果保持不变：Meta-AMS hard-b 三训练种子均值93.75%、93.75%、91.6667%；总体93.0556%、训练种子均值样本标准差1.2028个百分点、九评估最差87.5%、最终门禁3/3通过；clean到hard-a全部100%。最终450评估碰撞/道路越界/道路终止率最大均0。共104训练块、4160更新、4,259,840环境转移（不含BC拟合/评估/视频）。
- 统计分析以每个训练种子的三个攻击评估均值为单位，不能把九次评估当九次训练。事后探索性 hard-b 配对符号置换枚举2^3=8种组合；Meta-AMS对其余五方法每个配对差均正，但双侧精确p均0.25，五比较Holm校正均1.00。观察效应量可报告，不能写统计显著最优；检验有效性还条件于零假设下配对标签/符号可交换，不由相同整数seed自动保证。三个种子共用一次BC拟合，候选门禁与最终评估重用固定eval seed；不是独立BC拟合或留出泛化测试。
- hard-b Meta-AMS平均完成率相对MAPPO、无Meta HAPPO、共享MAPPO、HAA2C、HATRPO分别高46.5278、45.8333、90.2778、65.9722、93.0556个百分点。完整置换分布40行保存。三个样本的双侧检验最小p=0.25；五比较Holm首阈值0.01对应至少8对才有足够离散分辨率，此为分辨率下限而非功效估算或自动追加训练理由。
- 报告明确保留全部早停、失败、BC回退和实际训练预算：HATRPO三条最终链是相同BC回退，Shared MAPPO两条为相同BC；不把这些称为三个独立成功训练权重。六方法的actor共享/更新机制不同，不能将差异单独归因于AMS或替代匹配2×2消融。当前均有BC/KL保持且专家混合为0，不能从该队列证明渐进专家退出或保持模块的独立因果收益；安全指标是在active shield下的观测，不是数学保证。
- 新产物目录：/home/cnc/Desktop/AMS/results_20261102。包含5页IEEEtran实验补充PDF及TeX、references目录、5张LaTeX表、6张PDF/PNG图、统计与预算CSV、450份原始评估summary+verified、104块训练/门禁记录、39份哈希匹配源码快照、复算脚本、4项统计单测、中文README与PAPER_INTEGRATION_zh.txt。代表性s0 hard-b原始70秒MP4、17500行五车轨迹、俯视图已附；轨迹图只略去最后step3500重置的不连续点，原始CSV未改，五车进度依次达到1的时间23.12/27.94/33.12/38.42/43.52秒（仅该展示环境，不冒充16环境统计）。
- 复现命令：在仓库运行 /home/cnc/Downloads/ENTER/bin/python3 -u scripts/tools/build_ackermann_paper_results.py --source logs/ackermann_completion_20261027 --output /home/cnc/Desktop/AMS/results_20261102；在产物目录运行 python scripts/build_ackermann_paper_results.py --output . --rebuild、python scripts/test_ackermann_paper_results.py、python scripts/verify_package.py。独立CSV复算成功，4项单测通过，33项派生产物哈希及39项源码哈希匹配，1205个交付文件SHA256校验无误。
- 编译器仅安装到任务目录 /home/cnc/SSD_1T/xzw/ams_runtime/paper_tools/tectonic_0.17.0，不影响训练环境；Tectonic官方0.17.0发布归档SHA256=8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7，与官方下载元数据匹配。命令 tectonic --keep-logs --keep-intermediates ams_results_20261102.tex。PDF5页，已逐页渲染检查；无overfull、缺字、未定义字体/引用、??，保留少量underfull空白排版提示；compile.log与validation_report.json保存。
- 交付PDF：/home/cnc/Desktop/AMS/results_20261102/ams_results_20261102.pdf，SHA256=315872a5ea1fea895cd31ce72bf46c6c32dd15c7cf63834c195f85d3cfb707ab。
- 新ZIP：/home/cnc/Desktop/AMS_results_20261102.zip，11,574,237字节，1206个文件（含manifest），CRC testzip通过；SHA256=5065dfa57d43c86668927d8db93145cea4e8a95c5c808775acec3a50a7260adf，同目录.sha256可校验。这是论文结果补充包，非全部原始大数据归档；全部162视频、完整逐步评估CSV和checkpoint仍在服务器已记录路径，不声称本ZIP包含全部。
- 论文版本差异：现存Desktop/AMS/ams_paper.pdf（10月21日）与ams_paper.tex（10月13日）摘要/理论内容不完全对应；中文衔接说明的行号只指现存TeX。原TeX SHA256=36e6f7831e5c38bc44bb5fb1b053b72c7d1c3bbd30a55b6ba54e3b61265e3904，原PDF=353eb13a80df10b47b5cfd89fa79602385e74ec1e1e8d9e4d426c2af5e159140，旧AMS.zip=e1b7c8ed52e3689249969a5802698d370821fe6ac74b96b72c1143bb5258c59b，本轮结束再次核验均未变。没有用旧TeX覆盖新版PDF，也没有声称完成Proposition 1修订。
- 下一步：整稿合并前先定位当前论文PDF对应的准确TeX版本；按中文衔接说明同步摘要、实验表图、预算与结论。新实验方面先固定独立留出评估及训练种子统计计划，再补当前协议的匹配模块消融/保持对照；原队列已完成无需重跑，增加seed的范围应依据研究问题与功效而非只追求显著性。后续新队列可完善telemetry退出回收，当前无残留进程需清理。


### 2026-11-02T18:27:29+08:00 补充资料位置核对

- 用户询问“补充后的完整资料在哪里”。本轮只核对目录和ZIP条目，未修改代码、论文、实验数据，未运行训练。
- 执行ls核对 /home/cnc/Desktop/AMS、AMS/results_20261102、AMS.zip、AMS_results_20261102.zip，并用Python zipfile读取两个归档目录。
- 桌面论文资料总目录为 /home/cnc/Desktop/AMS，现包含原有资料及新增 results_20261102 子目录；新增实验PDF位于该子目录的ams_results_20261102.pdf。
- /home/cnc/Desktop/AMS.zip 为10月13日旧包，5677条目，不包含 results_20261102；/home/cnc/Desktop/AMS_results_20261102.zip 为新增结果补充包，1206条目，约12MB，只包含本轮补充资料。尚未生成同时合并原资料与新补充的单一完整ZIP，不能把补充ZIP称为完整原始归档。
- 全量实验大文件仍位于 /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_completion_20261027；桌面资料包未包含全部checkpoint/逐步CSV/162视频。指标和实验状态未变化。下一步仍为定位当前PDF准确TeX版本后进行整稿合并；本次向用户明确资料位置与两个压缩包范围。


### 2026-11-03T02:38:57+08:00 合并完整论文资料包并准备邮件交付

- 用户明确授权将补充后的完整资料包邮件发送至 m13546208014@163.com。本轮开始合并Desktop/AMS全部原有资料与results_20261102，不改原论文、算法或实验数据。
- 已核验补充包1205个文件SHA256无误；旧资料存在两处绝对路径目录链接（旧hard-b attack评估和clean验证），目标均存在，分别7文件/17129962字节与7文件/16637282字节，将在新ZIP中展开为真实文件，避免跨电脑失效。
- 当前无可用邮件工具；检索到Gmail/Outlook候选但均未连接。服务器无sendmail/msmtp/mail/mailx和已配置SMTP；Thunderbird两份profile均未发现邮件账号配置。已请求用户提供/连接发件通道，不请求聊天提供密码或授权码。尚未发送邮件，未上传第三方存储。
- 计划输出 /home/cnc/Desktop/AMS_complete_20261103.zip，包含原AMS目录、新补充、最新工作记录副本和交付索引/哈希清单；所有实验指标保持上一轮结果（Meta-AMS hard-b 93.0556%、3/3门禁通过）。完整Isaac环境、全量训练checkpoint和全部逐步评估大文件依然在服务器原路径，不将论文交接包冒称整台服务器备份。
