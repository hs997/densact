# Data provenance

The manuscript uses the following disk-resident artifacts.

## Main Meta-AMS ladder

- Matrix: /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/six_algorithm_attack_matrix.csv
- Meta-AMS hard-b top-down evidence: .../final_ladders/happo_meta_ams/hard_b/topdown_true_xy.png
- Meta-AMS hard-b reward/safety evidence: .../final_ladders/happo_meta_ams/hard_b/reward_attack_safety.png
- Meta-AMS hard-b training reward: .../stages/happo_meta_ams/hard_b_r1/plots/happo_meta_ams_training_reward.png

The matrix reports completion means of 1.00 from off through hard-a and 0.75 at hard-b for happo_meta_ams; collision, road-boundary violation, and road termination are zero in all reported Meta-AMS ladder rows.

## Spatial Ackermann evidence

- /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_expert_smoke/evidence/figures/top_down_trajectory.png
- /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_expert_smoke/evidence/figures/tracking_and_front_steering.png
- BC checkpoint: /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt

## Orthogonal pilot

- Matrix: /home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/six_algorithm_attack_matrix.csv
- Base shared hard-b completion mean: 0.5208; worst seed: 0.50.
- Meta-only hard-b completion mean: 0.5208; worst seed: 0.50.
- Early AMS-only hard-b assessment: 0.0417; worst seed: 0.00; training stopped at light.
- The early AMS-only branch used a different actor-sharing setting. It is diagnostic evidence only and is not a final causal ablation result.

## Statistical boundary

The three attack/environment seed pairs in the reported matrices are evaluation repetitions. They are not independent training seeds. The corrected three-training-seed four-cell revalidation is stored under
/home/cnc/SSD_1T/xzw/IsaacLab-main/logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0,
..._s1, and ..._s2, and is intentionally not represented as completed in the manuscript.

## Protocol caveats

- The reported matrix commands use env.algorithm.attack_mode=profile, not the optional CAGAN/generator attack path.
- The verified follow-up launcher overrides AMS to happo_ams_advantage_weight=0.02, happo_ams_warmup_updates=5, and happo_ams_ramp_updates=15; the runner defaults are different.
- The verified orthogonal stages enable fixed BC/KL retention but set both retention schedules and expert blending to disabled/zero. A progressive expert-withdrawal claim requires a separate run.
- At the 2026-10-13 status check, the s0 multiseed run was partial and still active. Do not use its intermediate method outcomes as final four-cell statistics.
