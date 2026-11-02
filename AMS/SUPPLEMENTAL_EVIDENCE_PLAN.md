# Supplemental material still needed before a final submission

This file separates missing documentation from missing experiments. The manuscript should not claim an item under “needs a run” until the corresponding artifact exists and passes the same whole-platoon gate.

## Corrections that must remain synchronized

1. The official AMS source is Lin, “Action Manifold Smoothing: A Lipschitz Pathway Perspective on High-Dimensional Reinforcement Learning,” ICML 2026, PMLR 306:74070--74088. The BibTeX entry is now in references/ams.bib. Cite the original AMS paper and cite the present work only for the HAPPO/CTDE, Ackermann-feasible, executed-action-anchored adaptation.
2. The reported matrix uses attack_mode=profile. It does not train or evaluate the CAGAN/generator mode. The paper must call the current attack “bounded hybrid profile DoS--FDI” unless a separate CAGAN experiment is added.
3. The code defaults are AMS weight 0.10, warm-up 50, ramp 100. The verified matrix launcher overrides them to weight 0.02, warm-up 5, ramp 15. Every table and configuration appendix must identify which values were actually used.
4. The current orthogonal matrix enables fixed BC and reference-policy retention (BC coefficient=10, KL coefficient=1, batch size 1024), but disables both retention schedules and expert blending. It therefore supports the claim “BC/retention warm-started attack adaptation,” not yet the stronger claim “progressively removes expert control.”

## Experiments required for the three-part research line

### A. Objective alignment plus AMS stability

The corrected four-cell factorial run must finish for at least three independent training seeds:

    HAPPO shared
    HAPPO + Meta Teacher
    HAPPO + AMS
    HAPPO + Meta Teacher + AMS

For every cell and seed, retain clean/off through hard-b, whole-platoon completion, collision, road violation, road termination, dwell, and the final accepted checkpoint. Report mean, standard deviation, worst training seed, and a confidence interval only across independent training seeds. Evaluation environment repetitions must be labelled separately.

Add a mechanism ablation for:

- pointwise target versus symmetric neighborhood target;
- random directions versus QR-orthogonal directions;
- Euclidean clipping versus Ackermann feasibility projection;
- actor action versus executed-action critic anchor;
- single Q versus twin-Q minimum;
- AMS correction weight and neighborhood radius.

### B. Ackermann spatially coupled maneuver

Add machine-readable evidence, not only a top-down image:

- per-vehicle x/y/yaw and path-progress traces;
- per-vehicle centerline, heading, predecessor-gap, pair-lateral, and pair-distance traces;
- the exact dwell interval during which the joint completion gate remains true;
- front-left/front-right steering targets, rear wheel velocity targets, and a rear-steering target audit proving that rear wheels never steer;
- a video and a static montage showing the five vehicles entering and leaving the same quintic path sequentially.

The paper should report the worst vehicle and worst adjacent pair, not only the average. The current gate thresholds are progress 0.99, lateral 0.12 m, heading 0.12 rad, gap 0.35 m, pair lateral 0.20 m, pair distance 0.30 m, zero road exceedance, and 2 s dwell.

### C. Capability retention and progressive attack adaptation

This is the largest missing experimental block. Run the same four-cell comparison with a specified retention schedule, for example:

    BC acquisition -> high retention -> linear BC/KL decay -> attack adaptation

Log the coefficients at every update, expert blend ratio, BC loss, reference-policy KL, clean completion after every stage, and attack completion after every stage. Compare against fixed-retention and no-retention controls. The desired claim is “limits degradation” or “suppresses forgetting,” not “guarantees no forgetting.”

The schedule must be evaluated both during training and after the last attack stage by returning each accepted checkpoint to the clean environment. A method that passes hard-b but loses clean completion should not be counted as capability-preserving.

## Attack-model evidence

The present result is a profile attack. If the paper wants the stronger wording “adaptive/state-conditioned hybrid attack,” add a separate attack_mode=cagan matrix with the same checkpoint, seed, budget, and completion protocol. Compare profile and CAGAN as attack families; do not merge their rows.

For every attack family, record observation FDI, command FDI, DoS masks, hold duration, attack seed, and whether the attack changed the executed command. This is needed to distinguish observation corruption from action corruption and from safety-shield intervention.

## Theory supplements

Add four short technical items to the appendix:

1. A local Taylor expansion under differentiability and an inactive projection boundary: symmetric pairs cancel first-order terms and yield directional second-order terms. A full Laplacian interpretation needs a complete basis or a suitable expectation over random bases. Clipping can break symmetry, and the minimum of twin critics can be nonsmooth; handle these cases separately instead of assuming cancellation everywhere.
2. A statement of the conditions under which the Ackermann projection is locally Lipschitz/non-expansive in the normalized command neighborhood.
3. A bound or diagnostic for the difference between the ideal outer meta-gradient and the implemented physical-score/teacher-output proxy.
4. A local ultimate-bound calculation using logged attack, teacher, and AMS residual magnitudes. The current proposition is conditional; it is not a global safety proof.

## Evidence and reproducibility appendix

### Remaining manuscript/source audit (not yet resolved)

- The manuscript's smoothed-Q equation places the twin-critic minimum around the sum of the plus/minus values; the code takes a minimum at each neighbor and then averages. These operations are not generally equivalent. Align the equation with the archived implementation used for the reported checkpoint.
- The current runner uses per-vehicle conditional local advantages: hold the other cars at their executed actions and perturb one vehicle's action block. Distinguish that actor signal from joint-neighborhood smoothing in the TD target. Joint QR directions need not remain orthogonal after restriction to a two-dimensional action block.
- Safety shields remain enabled in the verified protocol. Separate removal of the training-only teacher/AMS critic from removal of deployment-time safety corrections; the current result is not proof of unshielded policy safety.
- The teacher is a proxy-aligned implementation, not a verified exact meta-gradient. The local dissipativity and shield regularity assumptions still need justification. Bounded tracking error alone does not prove finite-time lane-change completion or string stability.
- The other workstation needs portable source snapshots, raw summary CSVs, selected configuration/gate JSON files, and a data dictionary. Absolute paths in FILE_MANIFEST.txt alone do not transfer this evidence. These portable attachments are not included in the current ZIP.
- “Real trajectory” refers here to simulator-executed expert trajectories; it does not establish physical-vehicle hardware experiments. Keep expert demonstrations separate from learned-policy evidence.

These are writing/audit tasks, not authorization to restart or change the running training protocol.

Archive, for each promoted checkpoint:

- source revision or source-file hashes;
- full Hydra/environment overrides;
- Python and Isaac Sim versions;
- GPU model and simulator seed;
- training seed versus evaluation seed;
- checkpoint SHA-256;
- CSV schema and plot-generation command;
- video/ROS replay path.

The external project root and all current output roots are listed in FILE_MANIFEST.txt. The running status must be refreshed from followup_status.json and the actual CSVs before each paper update.

## Current revalidation snapshot

At the last check on 2026-10-13, seed s0 had completed base-shared, Meta, and AMS-only training stages through hard-a. The hard-b training gate failed for all three of those cells; the last accepted base-shared checkpoint nevertheless passed the separate post-hoc hard-b evidence ladder, while Meta failed that post-hoc hard-b assessment and AMS-only was still being assessed. At 12:57 local time the launcher was waiting for exclusive GPU access before the AMS-only easy-0 final-ladder evaluation could continue. The combined Meta-AMS cell had not yet started in s0, and s1/s2 remained preflight-only. These are partial diagnostics, not final multi-seed statistics.

