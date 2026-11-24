# EMSE closed-loop verification

The next verification uses the [current mission protocol](../../Final_Code/README.md#current-mission-protocol). The 300 s budget includes the Nav2 return, sensing continues at 1 Hz on the return, and GRG uses the [calibrated travel-time model](../../Final_Code/Trained_timecost/README.md) for budget checks and predicted sampling times. Results for this protocol are pending.

## Archived verification

The run below validated EMSE with a 300 s sampling window and a separately timed final return.

## Configuration

30 × 20 m arena, Home and spawn at the field origin (0, 0) m, all 10 original candidate waypoints, Bunker Mini in Gazebo with Nav2, and up to 300 s for sampling and planning. The three policies share the origin-Home map, directed Nav2 path graph, Matérn SRF realization, 980 synthetic history samples, fixed real-data-fitted RBF hyperparameters, 1 Hz sampling and GP updates, and 9,150 common evaluation queries. The west and south simulator walls are 1.25 m outside the field, providing clearance for the robot. The temperature model and evaluation grid retain their original coordinates. The same random field also supplies samples within the clearance if needed. Nav2 generates the graph for this arena. GRG uses depth 2 and two time splits.

EMSE scores the expected reduction in GP posterior variance over the common space-time grid. It averages the reduction using fixed cell-area weights, repeated at each evaluation time. Each search uses a fixed GP snapshot. The worker takes the latest pending snapshot when it is ready for another search. At each waypoint, the next sampling leg must fit within the remaining budget. The GRG route ends at any feasible waypoint. Home is reserved for the initial pose and the final return. Once no next sampling leg fits, or the budget reaches 300 s, the robot stops sampling and starts a separate return to Home. The return has its own timing and supplies no samples to reconstruction.

## Results

| Metric | EMSE |
| --- | --- |
| Sampling duration / budget (s) | 283.260 / 300 |
| Sampling stop reason | no_sampling_leg_fits |
| Final return duration (s) | 37.710 |
| Elapsed time through final return (s) | 320.980 |
| Measurements / missed ticks | 283 / 0 |
| GP updates | 283 |
| Initial GRG compute (s) | 3.130 |
| Initial route (node IDs) | [0, 10, 1, 10] |
| GRG requests started / completed within mission | 151 / 150 |
| GRG compute mean / P95 / max (s) | 1.661 / 3.868 / 4.145 |
| Cumulative GRG compute (s) | 250.84 |
| Peak pending slot / unfinished requests | 1 / 0 |
| Search guard timeouts | 0 |
| Arrival waiting (s) | 0.786 |
| Handoffs with new plan / retained tail | 10 / 0 |
| Sampling legs / distance (m) | 11 / 227.41 |
| Returned Home / error (m) | True / 0.162 |
| Mean process CPU (100% = one core) | 413.1% |
| Peak aggregate process RSS (MiB) | 1295.0 |
| Real-time factor | 0.99997 |
| Final synthetic reconstruction RMSE (°C) | 0.07587 |
| History-only reconstruction RMSE (°C) | 0.09073 |

Mission acceptance **PASS**. The reference Python objective reproduced the initial score and 3 saved search scores within the 1e-8 absolute and relative tolerances (maximum absolute error 1.73e-17). The mission checks cover observation retention, arrival-only handoffs, per-leg sampling feasibility, the stop condition, and a separate final return. The reconstruction RMSE summarizes the final GP posterior on the shared synthetic evaluation grid for this single run.

The robot stopped sampling at 283.26 s near waypoint 10. At that point, 16.74 s remained and the shortest next leg was estimated at 17.23 s. The final return started at 283.27 s. Sampling arrivals at Home 0 time(s).

## Effect of reserving Home for return

| Metric | Home eligible for sampling | Home reserved for return |
| --- | --- | --- |
| Sampling duration (s) | 288.228 | 283.260 |
| Final return duration (s) | 22.950 | 37.710 |
| Valid samples | 288 | 283 |
| Sampling distance (m) | 230.11 | 227.41 |
| Single-search GRG P95 (s) | 5.237 | 3.868 |
| Reconstruction RMSE (°C) | 0.07365 | 0.07587 |

Both runs use Home (0, 0) m and the same origin fixture.

[Previous origin run summary](results/run_20261123T221211Z/latest_gp/summary.json).

[Notebook with plots and embedded replay](test_report.ipynb) · [Run summary](results/run_20261123T224551Z/latest_gp/summary.json)

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

With the container running, invoke `docker exec bunker-jazzy bash '/workspace/Code/Verification tests/EMSE_Test/code/run.sh'`. Run `code/report.py` with the reporting Python environment to refresh the notebook.
