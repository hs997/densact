# AMS experiment handoff and evidence ledger

All numbers in this file are snapshots from disk and must be rechecked before publication. The project root is `/home/cnc/SSD_1T/xzw/IsaacLab-main`.

## Completed single-chain matrix

Main matrix:

    logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003/six_algorithm_attack_matrix.csv

The six methods are MAPPO, HAPPO without Meta/AMS, Meta-AMS (`happo_meta_ams`), HARL-HAA2C, HARL-MAPPO shared, and HARL-HATRPO. At hard-b, the recorded whole-platoon completion means and worst evaluation seed were:

| method | hard-b mean | hard-b worst seed | collision | road violation | road termination |
|---|---:|---:|---:|---:|---:|
| MAPPO | 0.6250 | 0.5625 | 0 | 0 | 0 |
| HAPPO without Meta | 0.5833 | 0.5000 | 0 | 0 | 0 |
| Meta-AMS | 0.7500 | 0.7500 | 0 | 0 | 0 |
| HARL-HAA2C | 0.5000 | 0.4375 | 0 | 0 | 0 |
| HARL-MAPPO shared | 0.1042 | 0.0000 | 0 | 0 | 0 |
| HARL-HATRPO | 0.0208 | 0.0000 | 0 | 0 | 0 |

For `happo_meta_ams`, completion was 1.00 from off through hard-a and 0.75 at hard-b. The hard-b top-down and reward/safety figures in this package come from the corresponding final-ladder artifacts.

These are repeated evaluation environments/seeds for one training chain, not independent training seeds. Use “best completed single-chain result” rather than “statistically significant best method.”

## Orthogonal pilot

Pilot matrix:

    logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261009_v3/six_algorithm_attack_matrix.csv

The pilot hard-b values were:

| cell | hard-b mean | worst seed | interpretation |
|---|---:|---:|---|
| base shared | 0.5208 | 0.50 | shared-actor baseline |
| Meta-only | 0.5208 | 0.50 | teacher only in this pilot |
| AMS-only | 0.0417 | 0.00 | failed at light; early diagnostic only |
| Meta-AMS | not a valid causal conclusion from the pilot | — | use the corrected run |

The early AMS-only branch used `happo_share_actor=false`, so the failure is confounded. Never use this table as proof that AMS is intrinsically harmful. The corrected run forces shared actor settings across the four cells.

## Corrected three-training-seed revalidation

The currently running outputs are:

    logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s0
    logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s1
    logs/rsl_rl/platoon_happo/ams_orthogonal_ablation_20261012_multiseed_s2

The launcher is:

    scripts/tools/run_ackermann_orthogonal_multiseed.sh

The four cells are happo_base_shared, happo_meta, happo_ams_only, and happo_meta_ams. Seed mapping is training/environment seed 42, 3407, and 777. Each cell uses the same real-trajectory BC warm start and clean-to-hard-b gate. At the 2026-10-13 check, s0 was still active: base shared had reached hard-b evidence, Meta and AMS-only had trained through hard-a but their hard-b training gates failed, and the combined cell had not yet started. s1/s2 remained preflight-only. No final multiseed aggregate is available.

When complete, collect for each cell and attack level:

- number of independent training seeds;
- whole-platoon completion mean, standard deviation, and worst seed;
- collision rate, road-boundary violation rate, and road-termination rate;
- completion gate components: progress, lateral, heading, gap, pair-lateral, pair distance, road exceedance, and dwell;
- reward curves and training duration;
- trajectory, top-down, lane-change image, and real-trajectory video paths.

Use bootstrap confidence intervals or a clearly stated t interval only after confirming the number of independent training seeds. Do not mix evaluation repetitions with training-seed uncertainty.

## Stage protocol and budgets

The verified ladder is clean/off, light, easy-0, easy-a, easy-b, med-a, med-b, hard-a, and hard-b. A stage must pass its strict whole-platoon gate before the next attack intensity is attempted. The current orthogonal matrix uses a real trajectory BC warm start plus fixed BC/KL retention; retention curriculum and expert blending are disabled in its actual overrides. A progressive expert-exit schedule is supported by the code but still needs a dedicated experiment. The last passing checkpoint is retained if a later stage fails.

The actual short-stage AMS values in the verified follow-up launcher are correction weight 0.02, warm-up 5 updates, ramp 15 updates, fixed BC coefficient 10, fixed reference-policy KL coefficient 1, and retention batch size 1024. The runner class defaults (0.10, 50, 100) are not the values used by these matrix stages.

The safe reference for the earlier hard-bridge work is recorded in the project notes. The current paper uses the completed `ams_verified_matrix_repair_20261003` matrix for the six-method headline comparison; do not silently substitute a later failed candidate.

## Evidence locations

- Expert spatial path and steering: `logs/ackermann_expert_smoke/evidence/figures/`.
- Real trajectory BC checkpoint: `logs/ackermann_bc_warmstart/real_trajectory_bc/model_bc.pt`.
- Matrix CSVs: each experiment output root, usually `six_algorithm_attack_matrix.csv`.
- Stage status/manifests: each method/level directory under `stages/` and `followup_status.json` at the output root.
- Trajectory CSVs and replay/video: method/level `validation/` directories, often with `trajectory_env0.csv`, `topdown_true_xy.png`, and `reward_attack_safety.png`.

## Reproducibility checks

From the project root, after confirming exclusive GPU availability:

    bash -n scripts/tools/run_ackermann_orthogonal_multiseed.sh
    python3 -m py_compile scripts/tools/run_ackermann_verified_followup.py
    python3 scripts/tools/test_ackermann_followup.py

For a completed matrix, inspect CSV headers and manifests before computing statistics. Check that train and evaluation flags match, especially `happo_share_actor`, `happo_ams_enabled`, `enable_teacher`, attack level, attack seed, BC path, and the whole-platoon completion thresholds.

## What a future paper update should add

1. A four-cell table with three independent training seeds.
2. A plot of hard-b completion versus training seed for all four cells.
3. A safety table showing collision and road violations, not only completion.
4. A trajectory montage showing all five vehicles crossing the same path and completing the dwell gate.
5. A sensitivity plot for AMS neighborhood radius, number of orthogonal directions, and AMS correction weight.
6. If feasible, a wall-clock/critic-overhead comparison and a clean-no-attack robustness check.

