# Comprehensive AMS package scope

This ZIP is the portable research handoff, not a full Isaac Lab installation.

## Included

- The current LaTeX manuscript, BibTeX references, figures, and all GPT/Codex handoff documents.
- Read-only snapshots of the AMS-relevant Python modules and experiment launcher scripts.
- Selected experiment manifests, gate JSON files, environment/agent YAML files, training metric CSV files, matrix CSV files, and method outcomes.
- The full work log ams_debug_notes.md.
- The original MGRS reference PDF and the official AMS ICML 2026 paper PDF.
- The real-trajectory BC checkpoint and the compact Ackermann expert evidence set.
- The verified Meta-AMS hard-b final-ladder evidence directory, including validation summaries, top-down plot, reward/safety plot, trajectory CSVs, and video.

## Intentionally excluded

- The complete Isaac Lab checkout, Isaac Sim installation, USD/assets, Python environments, and systemd configuration.
- Large training checkpoints from every stage and the multi-gigabyte raw experiment roots.
- Redundant per-environment render logs and repeated evaluation-step CSVs outside the selected Meta-AMS hard-b evidence.

The excluded files remain at the absolute paths listed in FILE_MANIFEST.txt and DATA_PROVENANCE.md. Copy the project checkout and simulator separately if the second machine must rerun training.

## Reproducibility warning

The included s0 multiseed files are an in-progress snapshot. They are evidence of current state, not final three-training-seed statistics. The package preserves both the training gate and post-hoc final-ladder files so that a writer can distinguish promoted checkpoints from failed candidates.

