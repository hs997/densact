# AMS manuscript package

This folder contains a draft IEEE-style manuscript for the current AMS project:

> Meta-Guided Action-Manifold Smoothing for Secure Ackermann Vehicle Platoons under Hybrid Attacks

The draft is written against the implementation and disk-resident experiment records available on 2026-10-12. The reported Meta-AMS result is the completed single-training-chain happo_meta_ams ladder from
logs/rsl_rl/platoon_happo/ams_verified_matrix_repair_20261003.

The independent multi-training-seed revalidation is intentionally not reported as completed. The manuscript labels the present orthogonal AMS-only result as a diagnostic pilot because the earlier AMS-only branch used a different actor-sharing setting and failed the light gate. The corrected shared-actor multiseed revalidation is running separately.

## Files

- ams_paper.tex: main LaTeX manuscript.
- references/ams.bib: BibTeX database.
- figures/: figures copied from the verified Ackermann and attack-ladder evidence folders.
- GPT_HANDOFF.md, CODE_LOGIC.md, THEORY_WRITING_GUIDE.md: handoff material for another GPT/Codex workstation.
- EXPERIMENT_HANDOFF.md, SUPPLEMENTAL_EVIDENCE_PLAN.md: evidence ledger and items still needed before final claims.
- PACKAGE_SCOPE.md: exact contents and exclusions of the comprehensive ZIP.

## Build

With an IEEEtran LaTeX installation, run pdflatex on ams_paper.tex, run bibtex on ams_paper, and run pdflatex twice more.

The current host does not have a LaTeX executable installed, so compilation must be performed on a TeX Live/MiKTeX machine.
