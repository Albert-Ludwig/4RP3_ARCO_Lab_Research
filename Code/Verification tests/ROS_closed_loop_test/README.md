# ROS closed loop: GP–GRG update interval

## Purpose

Measure the compute cost and feasibility of GP + GRG updates every 1 s versus every 5 s, with **WIVR fixed in both cases**. 

## Folder layout

```text
ROS_closed_loop_test/
  code/          # Source, configuration, launchers and source snapshots
  results/       # Runtime data, metrics, models and figures
  recordings/    # MP4 replay and rosbag recordings
  README.md
  test_report.ipynb
```

## Configuration and architecture

30 × 20 m flat map, ten candidate waypoints plus Home, Bunker Mini/Gazebo/Nav2, 300 s missions, 1 Hz continuous temperature sampling in BOTH cases. Fixed real-fitted RBF parameters, 980 same-SRF synthetic historical observations; independent Matérn SRF truth. Fixed WIVR scoring method, depth 2/two splits; 24 spatial cells × 151 target times = 3,624 scoring targets. Predicted sensing remains 1 Hz. The only intended configuration change is the GP + GRG update interval: 1 s versus 5 s.

The main C++ node buffers every received temperature record. On each configured update it assimilates all buffered observations, freezes a data-only GP snapshot, and enqueues one complete GRG request. The worker rebuilds that snapshot, evaluates WIVR with area × sample-time posterior variance, and searches from the active goal endpoint at its estimated arrival time. The current Nav2 goal is never replaced mid-leg. At arrival, only a completed result with the matching goal epoch can supply the next goal. A delayed GRG result can use an older GP version; latency/version lag and waiting are recorded. FIFO requests are neither coalesced nor silently dropped. Initial planning is before the 300 s mission clock; the Home reserve is 30 s. Sensing continues until 300 s, including any terminal Home wait.

At stop, any residual observations are committed to the final GP without requesting post-mission navigation planning. In the 5 s case this normally means four residual records after the 295 s periodic update. This terminal partial-batch operation is separate from the periodic GRG deadline test. The worker finishes only its in-flight calculation; queued requests remain saved and are not counted as executed.

## Executed results

| Metric | GP + GRG every 1 s | GP + GRG every 5 s |
| --- | --- | --- |
| Successful measurements | 299 | 299 |
| GP updates (including terminal partial batch) | 299 | 60 |
| Pre-mission initial GRG (s) | 3.811 | 3.753 |
| GRG requests | 299 | 59 |
| GRG completed before stop | 65 | 59 |
| GRG completed including in-flight drain | 66 | 59 |
| Unprocessed requests | 233 | 0 |
| GRG compute mean / P95 (s) | 4.580 / 5.221 | 2.625 / 5.893 |
| GP update mean / P95 (ms) | 40.11 / 54.58 | 35.80 / 45.19 |
| Total GP update compute (s) | 11.99 | 2.15 |
| Total completed GRG compute (s) | 302.25 | 154.85 |
| GRG queue wait P95 (s) | 218.71 | 6.34 |
| GRG end-to-end latency P95 (s) | 224.01 | 11.34 |
| Requests meeting update deadline | 0 | 47 |
| Search guard timeouts | 0 | 0 |
| Peak queued requests | 233 | 2 |
| Sampled process peak RSS (MiB) | 1495 | 1326 |
| Sampled aggregate CPU mean / peak (%) | 502 / 544 | 351 / 543 |
| Consumed plan age P95 (s) | 43.59 | 12.03 |
| Arrival waiting (s) | 164.09 | 0.08 |
| Completed navigation legs | 4 | 11 |
| Distance (m) | 76.9 | 190.6 |
| Returned Home | True | True |
| Missed sampling ticks | 0 | 0 |
| Real-time factor | 0.99993 | 0.99995 |
| Final synthetic RMSE (°C) | 0.07994 | 0.08074 |
| History Only RMSE (°C) | 0.09073 | 0.09073 |
| All GP+GRG requests met interval deadline | False | False |

Five-second updates reduce request pressure, but not the intrinsic cost of a matched GRG search. Late-mission searches have fewer feasible branches as the remaining budget shrinks; the two per-request means summarize different completed request sets and are not an algorithm speedup factor. Computation overlaps navigation and must not be added to mission duration. The results distinguish functional success (sensing, exact GP assimilation, unchanged active goals and Home return) from timing acceptance. A complete search that exceeds its configured update interval is late even without hitting the 180 s guard. This is an implementation performance test of GP/GRG scheduling with one fixed scoring method, not a comparison of MI/EMSE/WIVR or proof about their reconstruction ranking. GP + GRG every 1 s: deadline acceptance = False; unfinished requests = 233. GP + GRG every 5 s: deadline acceptance = False; unfinished requests = 0.

[Figures and details](test_report.ipynb). [4× MCAP pose replay](recordings/run_20261117T023229Z/robot_replay.mp4).

## Files and computations

| File | Purpose |
| --- | --- |
| code/protocol.json | Shared configuration; 1 s / 5 s case overrides saved per run |
| code/src/core.hpp | Exact GP, WIVR, continuous sensing events and depth-2 GRG |
| code/src/nodes.cpp | Configured GP batching, FIFO GRG worker, goal-epoch checks and arrival-only goal switching |
| code/prepare.py | Common history, SRF, map, candidates and evaluation grid |
| code/scene.launch.py, navigation_profile.* | Existing Bunker/Nav2 launch and speed limits |
| code/srf_monitor.py | Existing Python temperature source and resource monitor |
| code/run.py, run.sh, Run-ClosedLoop.ps1 | Paired launch, MCAP recording, stop and worker drain |
| code/evaluate.py, evaluate_case.py | Independent numerical, data and performance checks |
| code/report.py | Single README and notebook |
| results/run_20261117T023229Z | Runtime inputs, logs, queued requests, final GP models, metrics and figures |
| code/snapshots/run_20261117T023229Z | Original runtime/analysis sources and current report source |
| recordings/run_20261117T023229Z | Video replay and update_1s/update_5s rosbag folders |

`GP::append/factor/posterior` implements the exact fixed-parameter RBF posterior. `Search::events` places predicted 1 Hz events on Nav2 polylines. `Search::score` computes joint posterior variance reduction, weighted by area × variance from the request GP; weights are fixed during each search. `Search::recurse` performs depth-2 GRG. `Experiment::update_gp` implements the 1/5 s batching; `process_jobs` computes GRG; `dispatch/poll` enforce arrival-only goal switching and remaining-time/Home checks. All actual samples use simulator pose and timestamp.

Dependencies: existing Docker bunker-jazzy, Jazzy/Nav2/Gazebo overlay, trained RBF hyperparameters, Nav2_test scene generation, SRF_test TemperatureField, Continuous_sensing_test GP reference and Scoring_policy_test objectives. WELL-E logic is unchanged.

```powershell
powershell -ExecutionPolicy Bypass -File .\code\Run-ClosedLoop.ps1
```
