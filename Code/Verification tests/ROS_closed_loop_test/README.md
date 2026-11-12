# Bunker Mini: Planning During Navigation

Retained run: **run_20261112T001310Z** (UTC run identifier). Only this run is presented.

## Purpose

Verify that the C++ GP/GRG planner runs while Bunker Mini navigates in Gazebo/Nav2, that 1 Hz continuous temperature acquisition remains complete, and that candidates are handed over at the correct endpoint with an updated GP and actual timestamps. This is one scheduling/performance verification, not a comparison of scoring policies or real-field accuracy.

## Configuration

| Item | Setting |
| --- | --- |
| Scene / robot | Flat 97 x 64 m rectangle, boundary walls, no interior obstacles; Bunker Mini, Gazebo, ROS 2 Jazzy and Nav2 |
| Localization / mapping | Ideal simulator localization; planar translation of real observations, not surveyed map registration |
| Candidates / start | 30 rounded 6 x 5 checkpoints plus start (48.5,32) m; 930 directed Nav2 paths |
| Mission / sensing | 300 s including initial planning; 1 Hz ticks 1-299, no dwell; free endpoint, no mandatory return |
| Initial GP | 980 real observations; fixed ARD RBF, no retraining |
| RBF lengths | x=3.598 m, y=21.094 m, t=21.136 s; all parameters in initial.json |
| Synthetic source | Independent GSTools finite-mode Matern SRF; nu=1.5, 1024 modes, seed=20261109 |
| Field scales / noise | [12, 12, 180] (m,m,s); noise SD 0.03 degC, keyed by measurement ID |
| Field reference resolution | 0.1 m / 0.5 s; arbitrary-position/time queries, no precomputed space-time grid |
| Scoring | Adaptive WIVR; 70 area-clipped 10 m cells x 151 target times at 2 s = 10,570 targets; route events remain 1 Hz |
| GRG / guards | Depth 2, two time splits; 45 s search and 10 s selected-route re-score guards |
| Parallelism | OpenMP/Eigen, 4 matrix threads; event-feature batches of 32 |
| Navigation | RPP, velocity smoother and bounded watchdog; forward/nominal 1.0 m/s, reverse 0.3 m/s, yaw command cap 0.6 rad/s |
| Update | Frozen committed GP during lookahead; actual samples buffered and assimilated at activation |
| Evaluation | 70 fixed locations at 0/75/150/225/300 s; GP committed by each time; History-Only baseline |

## Architecture and execution

| Module | Implementation | Responsibility |
| --- | --- | --- |
| Mission Manager | C++; lookahead_manager.hpp + handover.hpp | Keep one Nav2 goal active; future-endpoint request; request ID/anchor checks; arrival handover |
| Planner | C++/Eigen; nodes.cpp + core.hpp | Frozen GP/WIVR -> GRG candidate; actual batch -> GP update -> candidate re-score/time trimming |
| Sampler | C++; nodes.cpp | Capture actual pose/time every second and query temperature |
| Temperature source | Python/GSTools; srf_monitor.py | Independent fixed synthetic truth plus reproducible noise |
| Monitor / analysis | Python | Resource logs, independent numerical checks, MCAP-derived timing, figures and replay |

Initial search -> actual-data activation -> Nav2 goal. During that goal, LookaheadManager::ask_search requests a candidate anchored at the endpoint and a predicted arrival time. Planner::plan freezes the committed GP and weights; current-leg samples stay buffered. At actual arrival, activate_candidate waits for a candidate if necessary, commits the actual batch, and re-scores/trims the route at actual time before dispatch. Replanning never preempts the active Nav2 goal. At 300 s, sensing/navigation stop; an outstanding calculation may finish without dispatch.

## Recorded results

| Measurement | Recorded value |
| --- | --- |
| Simulation / wall time | 300.046 / 300.052 s |
| Received / missing / duplicate measurements | 299 / 0 / 0 |
| Completed navigation segments | 3 |
| Nonempty GP updates | 4; 3 after initial activation |
| Lookahead searches / arrival activations | 4 / 4 |
| Lookahead planning during physical motion | 93.30 s |
| Checkpoint handover waits | 2.42, 1.54, 1.29 s |
| Initial dispatch wait | 46.89 s |
| Total computation / computation while stationary | 145.22 / 51.40 s |
| Search guard timeouts | 3 |
| Real-time factor | 0.99998 |
| Distance travelled | 220.51 m |
| Sampled summed-process RSS / container peak | 2048 / 2533 MiB |
| SRF / sensor round-trip P95 | 1.124 / 3.262 ms |
| Committed new / terminal pending observations | 252 / 47 |
| End-time synthetic RMSE / History-Only | 0.195787 / 0.196260 degC |
| Maximum C++ / Python score discrepancy | 8.2e-17 |
| Movement-time planning / full-search acceptance | True / False |

Physical motion means pose-derived translation above 0.05 m/s or wrapped rotation above 0.05 rad/s. Overlap is integrated from positive pose intervals. Total computation overlaps navigation and must not be added to mission duration.

[Notebook with figures and replay](test_report.ipynb) | [Replay](result/run_20261112T001310Z/robot_replay.mp4)

The large pairwise Nav2 graph is published losslessly as graph.json.gz; graph.json remains a local working copy. The report reads either form.

## Limits and interpretation

Movement-time planning is verified; strict combined acceptance remains false because 3 searches hit the 45 s guard. Initial planning still waits 46.89 s. Activation checks an existing candidate under the updated GP, without repeating full route selection. ETA errors and late plans can still create waiting.

The trace remains effectively collinear (y-span 0.109 m); area coverage is not established. Synthetic field scales, sensor response and surveyed map registration remain unvalidated.

47 terminal observations remain saved but uncommitted. RMSE uses the committed GP against synthetic truth, without an artificial terminal update; it is not real-field accuracy or policy superiority. Expected endpoint navigation cancellations: 1.

Raw truth-twist yaw outliers: 4; per-run quaternion/pose checks are retained in yaw_incident.json, with no raw data removed. No GUI frame rate was measured; real-time factor and sensing ticks are reported separately.

## Files

| Path | Purpose |
| --- | --- |
| testing_code/src/nodes.cpp | Planner, sampler, search/activation branches and numerical audit |
| testing_code/src/core.hpp | RBF GP, continuous route events, WIVR, caches and GRG |
| testing_code/src/lookahead_manager.hpp, handover.hpp | Navigation/compute scheduling and request/endpoint checks |
| testing_code/src/matrix_threads.hpp, CMakeLists.txt, package.xml | OpenMP setup and ROS C++ build |
| testing_code/check_handover.py, src/handover_test.cpp | Early/late replies, stale IDs, wrong anchors and no-preemption state checks |
| testing_code/protocol.json, field_config.json | Runtime settings and fixed synthetic-source parameters |
| testing_code/prepare.py, scene.launch.py | Freeze inputs and generate the scene/Nav2 configuration |
| testing_code/navigation_profile.py, navigation_profile.json, capstone_nav2_reference.yaml | Navigation limits, smoother and configuration provenance |
| testing_code/srf_monitor.py | Temperature source and resource monitor |
| testing_code/run.py, run.sh, Run-ClosedLoop.ps1 | Preflight, build, launch, MCAP recording, stop/drain and reporting |
| testing_code/audit.py | Independent GP/WIVR checks and synthetic estimation errors |
| testing_code/overlap_metrics.py, report.py | Pose-based overlap, timeline, figures, replay and this executed notebook |
| result/latest.json | Pointer to the retained run |
| result/run_20261112T001310Z/rosbag/, *.jsonl, *.log | Raw MCAP, measurements, manager/planner events and resource logs |
| result/run_20261112T001310Z/initial.json, field.json, protocol.json, map.*, world.sdf | Exact recorded inputs |
| result/run_20261112T001310Z/plan_*.json, gp_*.json, batch_*.json | Candidate/activation records, GP snapshots and actual assimilated batches |
| result/run_20261112T001310Z/source_snapshot/, manifest.json | Original recorded source and hashes; historical evidence, not the live launcher |
| result/run_20261112T001310Z/analysis_source_snapshot/ | Analysis/report code for this presentation |
| result/run_20261112T001310Z/metrics.json, audit_and_estimation.json, artifact_validation.json | Results, independent checks and validation |
| result/run_20261112T001310Z/robot_replay.mp4, *.png | This run's physical-pose replay and figures |

## Reproduction and dependencies

~~~powershell
powershell -ExecutionPolicy Bypass -File .\testing_code\Run-ClosedLoop.ps1
~~~

A new run creates a new timestamped result; it does not overwrite this record. Preflight runs the handover test and checks actual OpenMP/GP operation without requiring discarded experiment results.

Requires Docker container bunker-jazzy, repository mount /workspace, ROS 2 Jazzy and overlay /ros-data/build/10.27_ROS_simulation/Nav2_test/install. C++ requires Eigen3, OpenMP and nlohmann/json. Container SRF Python: /home/ubuntu/.venvs/srf-verification; host reporting Python: C:\Ubuntu\ROS\reporting-venv.

External dependencies retained outside this folder: aligned observations Data/Real_Data_Integration/result/gp_samples.csv; saved RBF kernel Code/ROS_simulation/Trained_Real_Data_Kernel; Nav2/Bunker fixture code ../Nav2_test/testing_code; numerical helpers ../Continuous_sensing_test/testing_code, ../Scoring_policy_test/testing_code and ../SRF_test/testing_code. Exact recorded inputs remain in this run directory.

To rebuild this report from saved MCAP, run testing_code/report.py using the host reporting Python. Set CL_REUSE_VIDEO=1 to reuse the recorded replay.

