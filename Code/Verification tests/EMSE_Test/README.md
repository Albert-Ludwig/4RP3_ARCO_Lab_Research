# EMSE closed-loop verification

Verify EMSE in the shared ROS 2 closed-loop simulation with a 300 s return-inclusive deadline and continuous sensing on the return.

## Configuration

30 × 20 m arena, Home and spawn at the field origin (0, 0) m, all 10 original candidate waypoints, Bunker Mini in Gazebo with Nav2, and a 300 s mission from departure through actual arrival at Home. The three policies share the origin-Home map, directed Nav2 path graph, Matérn SRF realization, 980 synthetic history samples, fixed real-data-fitted RBF hyperparameters, 1 Hz sampling and GP updates, and 9,150 common evaluation queries. The west and south simulator walls are 1.25 m outside the field, providing clearance for the robot. The temperature model and evaluation grid retain their original coordinates. The same random field also supplies samples within the clearance if needed. Nav2 generates the graph for this arena. GRG uses depth 2 and two time splits.

EMSE scores the expected reduction in GP posterior variance over the common space-time grid. It averages the reduction using fixed cell-area weights, repeated at each evaluation time. Each search uses a fixed GP snapshot. The worker takes the latest pending snapshot when it is ready for another search. GRG reserves the Nav2 return to Home before accepting another sampling waypoint. All legs use the fitted duration L / 0.891822 + 2.709215 s. The predicted GRG score covers the ordinary sampling route. After waypoint selection ends, Nav2 navigates to the fixed Home goal and the 1 Hz sampler keeps collecting temperatures. The final GP includes the return observations, and actual arrival at Home determines whether the 300 s deadline was met.

## Results

| Metric | EMSE |
| --- | --- |
| Home arrival / mission budget (s) | 284.110 / 300 |
| Waypoint phase duration (s) | 276.900 |
| Return measurements in final GP | 8 |
| Sampling stop reason | no_sampling_leg_fits |
| Final return duration (s) | 7.200 |
| Elapsed time through final return (s) | 284.110 |
| Measurements / missed ticks | 284 / 0 |
| GP updates | 284 |
| Initial GRG compute (s) | 1.840 |
| Initial route (node IDs) | [0, 10, 1, 10] |
| GRG requests started / completed within mission | 176 / 176 |
| GRG compute mean / P95 / max (s) | 1.375 / 2.524 / 2.625 |
| Cumulative GRG compute (s) | 241.96 |
| Peak pending slot / unfinished requests | 1 / 0 |
| Search guard timeouts | 0 |
| Arrival waiting (s) | 0.072 |
| Handoffs with new plan / retained tail | 9 / 0 |
| Sampling legs / distance (m) | 10 / 226.58 |
| Returned Home / error (m) | True / 0.163 |
| Mean process CPU (100% = one core) | 442.4% |
| Peak aggregate process RSS (MiB) | 1296.9 |
| Real-time factor | 0.99992 |
| Final synthetic reconstruction RMSE (°C) | 0.07376 |
| History-only reconstruction RMSE (°C) | 0.09073 |

Mission acceptance **PASS**. The reference Python objective reproduced the initial score and 3 saved search scores within the 1e-8 absolute and relative tolerances (maximum absolute error 1.73e-17). The mission checks cover observation retention, arrival-only handoffs, return-inclusive feasibility, actual Home arrival, and retention of the return samples. The reconstruction RMSE summarizes the final GP posterior on the shared synthetic evaluation grid for this single run.

The robot stopped adding sampling waypoints at 276.90 s near waypoint 1. At that point, 23.10 s remained and the shortest next leg plus return was estimated at 23.21 s. The final return started at 276.91 s. The robot reached Home at 284.11 s and recorded 8 return measurements.

## Comparison with the archived protocol

| Metric | Archived sampling-only budget | Return-inclusive mission |
| --- | --- | --- |
| Waypoint phase duration (s) | 283.260 | 276.900 |
| Home arrival (s) | 320.980 | 284.110 |
| Return samples | 0 | 8 |
| Final return duration (s) | 37.710 | 7.200 |
| Valid samples | 283 | 284 |
| Sampling distance (m) | 227.41 | 226.58 |
| Single-search GRG P95 (s) | 3.868 | 2.524 |
| Reconstruction RMSE (°C) | 0.07587 | 0.07376 |

Both runs use Home (0, 0) m and the same origin fixture. The revised run changes the travel estimate, the budget endpoint, and whether return samples feed the final GP.

[Previous origin run summary](results/run_20261123T224551Z/latest_gp/summary.json).

[Notebook with plots and embedded replay](test_report.ipynb) · [Run summary](results/run_20261124T031414Z/latest_gp/summary.json)

## Files and dependencies

| Location | Contents |
| --- | --- |
| `code/src/` | C++ GP, scoring, GRG search, latest-GP background scheduling and ROS/Nav2 interfaces |
| `code/protocol.json` | Mission and planner configuration |
| `code/run.sh`, `code/run.py` | ROS build and run, fixed input loading and MCAP recording |
| `code/scene.launch.py`, `code/srf_monitor.py` | Gazebo/Nav2 scene, SRF measurements and resource sampling |
| `code/evaluate_case.py`, `code/report.py` | Score audits, reconstruction evaluation, report and replay |
| `results/<run>/` | Inputs, hashes, event logs, final GP state and metrics |
| `recordings/<run>/` | Raw rosbag and MP4 replay |

The runner uses the existing `bunker-jazzy` Docker container with the ROS 2 Jazzy/Nav2/Bunker Mini overlay and the SRF verification Python environment. Historical inputs are read from the saved [WIVR run](../WIVR_test/results/run_20261120T145549Z/). The Python evaluator uses the repository's GP, scoring and SRF reference modules. The shared sampling_fixture.py creates the origin Home, spawn and clearance geometry.

To reuse the paired Nav2 paths, invoke `docker exec -e 'CL_SHARED_GRAPH_CASE=/workspace/Code/Verification tests/WIVR_test/results/run_20261123T225159Z/latest_gp' bunker-jazzy bash '/workspace/Code/Verification tests/EMSE_Test/code/run.sh'`. The runner recalculates travel costs from the same path geometry. Run `code/report.py` with the reporting Python environment to refresh the notebook.
