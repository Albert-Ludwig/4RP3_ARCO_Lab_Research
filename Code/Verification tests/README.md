# Verification Tests

Component checks and individual-policy functional runs belong here. The current stage verifies correctness and integration readiness; it does not rank reconstruction performance. Final experiments and trained real-data kernels remain in [Final_Code](../Final_Code/README.md).

## Return-inclusive ROS closed-loop verification

The MI, EMSE and WIVR runs share a **300 s total budget including return to Home**. GRG retains the existing objective and checks return feasibility using the [calibrated travel-time model](../Final_Code/Trained_timecost/README.md), `d_ij = L_ij / v + delta` with `v = 0.891822 m/s` and `delta = 2.709215 s`. Nav2 handles the return to the fixed Home goal. Continue collecting temperature at 1 Hz on that route and assimilate the readings into the final GP. Three single-seed Gazebo runs passed the return deadline and observation-retention checks. All recorded 1 Hz samples through Home arrival, with zero missed ticks and zero GRG timeouts.

| Policy | Home arrival, old → new (s) | Observations, old → new | Return observations | Reconstruction RMSE, old → new (°C) | GRG compute P95, old → new (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [MI](MI_Test/README.md) | 315.19 → 290.55 | 282 → 290 | 7 | 0.07065 → 0.07097 | 0.77 → 0.51 |
| [EMSE](EMSE_Test/README.md) | 320.98 → 284.11 | 283 → 284 | 8 | 0.07587 → 0.07376 | 3.87 → 2.52 |
| [WIVR](WIVR_test/README.md) | 315.74 → 296.59 | 288 → 296 | 11 | 0.07105 → 0.07209 | 3.67 → 2.34 |

The archived protocol used a separate return and the nominal travel estimate. The revised runs evaluate the calibrated estimate and return policy together on the same fixture.

## Shared Python-fixture configuration

| Item                  | Setting                                                                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scene                 | Temporary 30 x 30 m area; Bunker Mini for the planned ROS stage.                                                                                                              |
| Mission budget        | 120 s total, including online computation, navigation, necessary waiting/recovery and return. Concurrent sensing is not charged twice.                                        |
| Candidate Checkpoints | 30 sensing candidates plus a separate Home, selected from a filtered 1 m base grid. No requirement to visit every candidate.                                                  |
| Measurements          | Continuous 1 Hz sampling at actual positions/timestamps; no fixed checkpoint sensing dwell.                                                                                   |
| Temperature source    | Independent, on-demand GSTools Matern SRF. Current field scales/noise/mask are provisional.                                                                                   |
| Reference resolution  | 0.1 m spatial / 0.5 s temporal for output and offline evaluation; arbitrary-coordinate sensor queries remain valid.                                                           |
| GP                    | Fixed-parameter 3D spatiotemporal RBF; segment-end batch conditioning. Current Python fixtures use 980 synthetic historical observations and saved real-data hyperparameters. |
| Scoring               | MI, area-averaged EMSE, Adaptive WIVR. A separate 10 m scoring grid uses 0.5 s scoring times. WIVR weights refresh after GP updates and freeze within each search.            |
| GRG                   | Recursive depth 2; time-split parameter 2.                                                                                                                                    |

The archived ROS performance tests used a **30 × 20 m arena, 10 waypoints plus Home, and a 300 s sampling window with a separate return**. The WIVR scheduling benchmark also compared the 1 s and 5 s GP/GRG update intervals.

The Python scoring fixture uses Home (3,3) m, a central box at x,y = [13,17] m, and 1 m/s nominal timing. Archived ROS execution and timing are recorded in the [WIVR test](WIVR_test/README.md).

## Test folders and status

| Folder                                                                                         | Purpose / status                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [GP3D_baseline_test](GP3D_baseline_test/README.md)                                             | Historical RBF GP reconstruction and prediction checks; executed.                                                                                                                                                               |
| [Robot_close_loop_system](Robot_close_loop_system/README.md)                                   | Historical Python GP-GRG replay, updates and budget checks; executed.                                                                                                                                                           |
| [Minimal_GP_test](Minimal_GP_test/README.md)                                                   | Initial GP validation on real WELL-E measurements; executed.                                                                                                                                                                    |
| [BunkerMini_drive_test](BunkerMini_drive_test/README.md)                                       | Gazebo drive, odometry, TF and command-timeout checks; passed.                                                                                                                                                                  |
| [Nav2_test](Nav2_test/README.md)                                                               | Navigation around one known static obstacle with ideal localization; passed.                                                                                                                                                    |
| [GRG grading policy compare verification](<GRG grading policy compare verification/README.md>) | Saved preliminary feasibility study of the final experiment workflow.                                                                                                                                                           |
| [SRF_test](SRF_test/README.md)                                                                 | Completed: 38 checks for the added obstacle-free 97 x 64 m / 120 s workload; 97 x 64 m / 900 s and 30 x 30 m / 120 s results retained. This source workload variant does not change the shared integration configuration above. |
| [Continuous_sensing_test](Continuous_sensing_test/README.md)                                   | Completed: 48 moving-sampling, buffering and batch GP checks.                                                                                                                                                                   |
| [Scoring_policy_test](Scoring_policy_test/README.md)                                           | Completed: 71 core checks and 6 independent route-score audits. All 30/50-point searches finished; 50-point EMSE/WIVR has a nominal 120 s budget limitation.                                                                    |
| [MI_Test](MI_Test/README.md), [EMSE_Test](EMSE_Test/README.md), [WIVR_test](WIVR_test/README.md) | Return-inclusive verification passed for each policy. The individual reports compare the saved and revised protocols and include MCAP replays. |
