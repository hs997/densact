# AMS handoff package for a second Codex/GPT workstation

## Read this first

This directory is a manuscript and research handoff package for the AMS project. It is not a self-contained Isaac Lab checkout. The runnable source tree remains at:

    /home/cnc/SSD_1T/xzw/IsaacLab-main

The manuscript draft is `ams_paper.tex`. The figures and BibTeX database are local to this folder. The source code, experiment logs, checkpoints, videos, and large simulator assets remain in the project tree and are listed in `FILE_MANIFEST.txt` and `DATA_PROVENANCE.md`.

The working title is **Meta-Guided Action-Manifold Smoothing for Secure Ackermann Vehicle Platoons under Hybrid Attacks**. The intended contribution has two complementary parts:

1. a physics-aware reward teacher that aligns attacked training with the true physical safety objective; and
2. action-manifold smoothing that regularizes local action-value geometry while respecting the two-dimensional Ackermann command manifold.

The strongest truthful current claim is conditional and empirical: the combined Meta-AMS configuration is the best result in the completed single-chain six-algorithm matrix at hard-b, while the independent multi-training-seed orthogonal revalidation is still running and must be added before claiming statistically robust superiority. The current matrix uses the bounded profile attack mode, short-stage AMS overrides (0.02 weight, 5 warm-up, 15 ramp), and fixed BC/KL retention rather than a progressive expert-exit schedule.

## Non-negotiable writing boundaries

- Do not report the 2026-10-12 multi-training-seed run as completed. Inspect its manifests and CSV files before updating the paper.
- Do not call the current teacher implementation an exact full second-order meta-gradient implementation. The code uses a stable first-order/proxy-aligned teacher update based on physical-score improvement and teacher-output consistency. The paper should state this explicitly.
- Do not claim global safety, global stability, or a formal guarantee under arbitrary attacks. The stability statement is a local/conditional ultimate-boundedness result under bounded attacks, bounded hold error, local dissipativity, and bounded optimization residual.
- The early AMS-only orthogonal pilot used a different actor-sharing setting and failed the light gate. It is diagnostic evidence, not a clean causal ablation.
- A completed episode means the whole five-vehicle platoon satisfies the joint progress, lateral, heading, spacing, pairwise-lateral, distance, road-boundary, and dwell conditions. It is not a leader-only lane-change score.
- All five agents use the same spatial lane-change path. The physical actuator model is front-wheel steering plus rear-wheel drive; the rear wheels never receive steering commands.

## Current disk status at handoff

At the time this package was prepared, the corrected orthogonal multi-seed launcher was still running:

    bash scripts/tools/run_ackermann_orthogonal_multiseed.sh

The launcher PID was 1886606 and its current child was a verified-followup process for seed index 0. The active stage was the shared-actor baseline clean/light evaluation chain. Seed indices map to training/environment seeds 0 -> 42, 1 -> 3407, and 2 -> 777. The process must not be killed merely to compile or edit the manuscript. The latest status and exact paths are always authoritative in:

    /home/cnc/SSD_1T/xzw/ams_debug_notes.md

The external Cagan/Isaac process that previously occupied the GPU had exited at the last check. Recheck `ps`, the launcher log, and each `followup_status.json` before starting any new GPU work.

## What the next GPT/Codex should do

1. Read this file, `CODE_LOGIC.md`, `THEORY_WRITING_GUIDE.md`, `EXPERIMENT_HANDOFF.md`, `DATA_PROVENANCE.md`, and the tail of `ams_debug_notes.md`.
2. Inspect the actual source and logs, rather than trusting this snapshot if they disagree.
3. Compile `ams_paper.tex` on a machine with IEEEtran, BibTeX, and a PDF viewer. Fix only evidence-supported wording.
4. When the multi-seed run completes, update the manuscript with mean, standard deviation, worst-seed, and confidence intervals only from the final CSVs. Keep single-chain and independent-training-seed statistics visibly separate.
5. Add any new result figures to `figures/`, add their provenance to `DATA_PROVENANCE.md`, and run the static checks described in `README.md`.
6. Append every important inspection, command, modification, metric, artifact path, and next step to `ams_debug_notes.md`.

## Conceptual origin

The reference paper on the desktop, `Security Control of Vehicle Platoons under Hybrid Attacks via Meta-Gradient Reward Shaping.pdf`, motivates the reward-teacher layer. Its central observation is that hybrid DoS/FDI attacks corrupt local observations, executed commands, and observation-derived rewards at the same time. A student policy can therefore improve its attacked reward while becoming worse in the true physical objective. The original solution uses a conditional attack generator, CTDE/HATRL, privileged physical scoring, and an outer reward-teacher update.

AMS extends that logic in the action space. The teacher determines the direction of the learning objective from clean physical state and executed action. AMS then compares the executed action with feasible nearby Ackermann actions, smooths a conservative local Q value, and supplies a local advantage correction to HAPPO. This division of labor is the main theoretical story: reward shaping corrects objective direction; action-manifold smoothing controls local policy-update geometry.

## Package contents

- `ams_paper.tex`: complete draft manuscript.
- `references/ams.bib`: BibTeX entries.
- `figures/`: figures copied from verified artifacts.
- `CODE_LOGIC.md`: source-level architecture and data flow.
- `THEORY_WRITING_GUIDE.md`: notation, propositions, assumptions, and claim boundaries.
- `EXPERIMENT_HANDOFF.md`: metrics, experiment matrix, stage names, and reproducibility commands.
- `DATA_PROVENANCE.md`: artifact origins and statistical caveats.
- `FILE_MANIFEST.txt`: concise map of this package and the external source tree.

