# MI closed-loop verification

The next verification uses the [current mission protocol](../../Final_Code/README.md#current-mission-protocol). The 300 s budget includes the Nav2 return, sensing continues at 1 Hz on the return, and GRG uses the [calibrated travel-time model](../../Final_Code/Trained_timecost/README.md) for budget checks and predicted sampling times. Results for this protocol are pending.

## Archived verification

The run below validated MI with a 300 s sampling window and a separately timed final return.

## Configuration

30 × 20 m arena, Home and spawn at the field origin (0, 0) m, all 10 original candidate waypoints, Bunker Mini in Gazebo with Nav2, and up to 300 s for sampling and planning. The three policies share the origin-Home map, directed Nav2 path graph, Matérn SRF realization, 980 synthetic history samples, fixed real-data-fitted RBF hyperparameters, 1 Hz sampling and GP updates, and 9,150 common evaluation queries. The west and south simulator walls are 1.25 m outside the field, providing clearance for the robot. The temperature model and evaluation grid retain their original coordinates. The same random field also supplies samples within the clearance if needed. Nav2 generates the graph for this arena. GRG uses depth 2 and two time splits.

MI scores the joint expected information of the 1 Hz sampling events along each candidate route. Each search uses a fixed GP snapshot. The worker takes the latest pending snapshot when it is ready for another search. At each waypoint, the next sampling leg must fit within the remaining budget. The GRG route ends at any feasible waypoint. Home is reserved for the initial pose and the final return. Once no next sampling leg fits, or the budget reaches 300 s, the robot stops sampling and starts a separate return to Home. The return has its own timing and supplies no samples to reconstruction.

## Results

| Metric | MI |
| --- | --- |
| Sampling duration / budget (s) | 282.972 / 300 |
| Sampling stop reason | no_sampling_leg_fits |
| Final return duration (s) | 32.210 |
| Elapsed time through final return (s) | 315.192 |
| Measurements / missed ticks | 282 / 0 |
| GP updates | 282 |
| Initial GRG compute (s) | 0.634 |
| Initial route (node IDs) | [0, 10, 1, 10] |
| GRG requests started / completed within mission | 293 / 293 |
| GRG compute mean / P95 / max (s) | 0.433 / 0.771 / 0.862 |
| Cumulative GRG compute (s) | 126.74 |
| Peak pending slot / unfinished requests | 1 / 0 |
| Search guard timeouts | 0 |
| Arrival waiting (s) | 0.810 |
| Handoffs with new plan / retained tail | 10 / 0 |
| Sampling legs / distance (m) | 11 / 226.97 |
| Returned Home / error (m) | True / 0.161 |
| Mean process CPU (100% = one core) | 326.4% |
| Peak aggregate process RSS (MiB) | 1081.9 |
| Real-time factor | 0.99999 |
| Final synthetic reconstruction RMSE (°C) | 0.07065 |
| History-only reconstruction RMSE (°C) | 0.09073 |

Mission acceptance **PASS**. The reference Python objective reproduced the initial score and 3 saved search scores within the 1e-8 absolute and relative tolerances (maximum absolute error 2.27e-13). The mission checks cover observation retention, arrival-only handoffs, per-leg sampling feasibility, the stop condition, and a separate final return. The reconstruction RMSE summarizes the final GP posterior on the shared synthetic evaluation grid for this single run.

The robot stopped sampling at 282.97 s near waypoint 8. At that point, 17.03 s remained and the shortest next leg was estimated at 17.23 s. The final return started at 282.98 s. Sampling arrivals at Home 0 time(s).

## Effect of reserving Home for return

| Metric | Home eligible for sampling | Home reserved for return |
| --- | --- | --- |
| Sampling duration (s) | 284.310 | 282.972 |
| Final return duration (s) | 12.010 | 32.210 |
| Valid samples | 284 | 282 |
| Sampling distance (m) | 225.93 | 226.97 |
| Single-search GRG P95 (s) | 1.178 | 0.771 |
| Reconstruction RMSE (°C) | 0.07431 | 0.07065 |

Both runs use Home (0, 0) m and the same origin fixture.

[Previous origin run summary](results/run_20261123T220640Z/latest_gp/summary.json).

[Notebook with plots and embedded replay](test_report.ipynb) · [Run summary](results/run_20261123T223950Z/latest_gp/summary.json)

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

With the container running, invoke `docker exec bunker-jazzy bash '/workspace/Code/Verification tests/MI_Test/code/run.sh'`. Run `code/report.py` with the reporting Python environment to refresh the notebook.
