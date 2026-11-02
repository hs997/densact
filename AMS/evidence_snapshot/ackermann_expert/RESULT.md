# Five-car Ackermann clean lane-change result

- Real Isaac physics, one environment, 0.005 s physics step and 0.02 s control step.
- Five vehicles use the same quintic spatial path and enter it sequentially.
- Commands are normalized rear-wheel speed plus front-knuckle steering; rear-wheel steering is never commanded.
- Strict instantaneous whole-platoon completion first occurred at step 1717 (34.34 s).
- Strict 2 s dwell completion first occurred at step 1816 (36.32 s).
- At dwell completion: minimum path progress 1.000, maximum center error 0.00561 m, maximum heading error 0.00507 rad, maximum gap error 0.26799 m, and maximum pair lateral error 0.00870 m.
- Collision rate, road-boundary violation rate, and maximum road exceedance were all 0.
- The strict road termination allowance remained 0.10 m; it was not relaxed for this result.

Artifacts:

- `videos_topdown/rl-video-step-0.mp4`: 38 s, 1280x720, 50 fps, fixed high-angle whole-platoon video.
- `trajectory.csv`: 2200 steps x 5 vehicles of pose, reference, road clearance, commands, wheel speeds, and physical front steering.
- `figures/top_down_trajectory.png`: overhead trajectory evidence.
- `figures/tracking_and_front_steering.png`: tracking error and front-steering evidence.
- Strict evaluator output is in `../../tune_01/eval_steps_model_final.csv` and `../../tune_01/eval_summary.csv`.

This is a deterministic clean-acquisition expert baseline, not a claim that the failed learned policy has converged. The next training stage should use this rollout for warm start/imitation, then require the same strict clean gate before any attack ladder.
