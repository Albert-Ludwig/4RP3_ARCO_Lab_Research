# Verification Tests

Component checks and individual-policy functional runs belong here. The current stage verifies correctness and integration readiness; it does not rank reconstruction performance. Final experiments and trained real-data kernels remain in [ROS_simulation](../ROS_simulation/README.md).

## Shared configuration

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

The Python scoring fixture uses Home (3,3) m, a central box at x,y = [13,17] m, and 1 m/s nominal timing. Actual ROS velocity, Nav2 path costs and mission timing remain to be verified. See the [ROS verification configuration](ROS_closed_loop_test/testing_code/protocol.json).

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
| [ROS_closed_loop_test](ROS_closed_loop_test/README.md)                                         | Remaining fourth stage: separate functional runs for MI, EMSE and Adaptive WIVR using 30 candidates. Configuration exists; implementation and execution are pending.                                                            |
