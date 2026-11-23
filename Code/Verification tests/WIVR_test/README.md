# WIVR closed-loop verification

Validate the existing WIVR objective in the shared latest-GP ROS 2 simulation, with the final return outside the sampling budget.

## Configuration

30 × 20 m arena, 10 candidate waypoints plus Home, Bunker Mini in Gazebo with Nav2, and up to 300 s for sampling and planning. Both policies use the saved map, directed path graph, Matérn SRF realization, 980 synthetic history samples, fixed real-data-fitted RBF hyperparameters, 1 Hz sampling and GP updates, and 9,150 common evaluation queries. GRG uses depth 2 and two time splits.

Adaptive WIVR scores expected GP variance reduction from the 1 Hz sampling events, weighted by cell area times the posterior variance at the current sample time. Each search uses a fixed GP snapshot. The worker takes the latest pending snapshot when it is ready for another search. At each waypoint, the next sampling leg must fit within the remaining budget. The GRG route ends at any feasible waypoint. A visit to Home during sampling follows the same rules as any other waypoint. Once no next sampling leg fits, or the budget reaches 300 s, the robot stops sampling and starts a separate return to Home. The return has its own timing and supplies no samples to reconstruction.

## Results

| Metric | WIVR |
| --- | --- |
| Sampling duration / budget (s) | 286.264 / 300 |
| Sampling stop reason | no_sampling_leg_fits |
| Final return duration (s) | 20.030 |
| Elapsed time through final return (s) | 306.304 |
| Measurements / missed ticks | 286 / 0 |
| GP updates | 286 |
| Initial GRG compute (s) | 3.087 |
| Initial route (node IDs) | [0, 9, 2, 9] |
| GRG requests started / completed within mission | 139 / 138 |
| GRG compute mean / P95 / max (s) | 1.776 / 4.610 / 5.109 |
| Cumulative GRG compute (s) | 246.92 |
| Peak pending slot / unfinished requests | 1 / 0 |
| Search guard timeouts | 0 |
| Arrival waiting (s) | 0.954 |
| Handoffs with new plan / retained tail | 10 / 0 |
| Sampling legs / distance (m) | 11 / 226.95 |
| Returned Home / error (m) | True / 0.152 |
| Mean process CPU (100% = one core) | 429.1% |
| Peak aggregate process RSS (MiB) | 1314.6 |
| Real-time factor | 0.99984 |
| Final synthetic reconstruction RMSE (°C) | 0.07033 |
| History-only reconstruction RMSE (°C) | 0.09073 |

Mission acceptance **PASS**. The reference Python objective reproduced the initial score and 3 saved search scores within the 1e-8 absolute and relative tolerances (maximum absolute error 3.82e-17). The mission checks cover observation retention, arrival-only handoffs, per-leg sampling feasibility, the stop condition, and a separate final return. The reconstruction RMSE summarizes the final GP posterior on the shared synthetic evaluation grid for this single run.

The robot stopped sampling at 286.26 s near waypoint 10. At that point, 13.74 s remained and the shortest next leg was estimated at 17.23 s. The final return started at 286.27 s. Home served as an intermediate sampling waypoint 0 time(s).

[Notebook with plots and embedded replay](test_report.ipynb) · [Run summary](results/run_20261123T210449Z/latest_gp/summary.json)

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

The runner uses the existing `bunker-jazzy` Docker container with the ROS 2 Jazzy/Nav2/Bunker Mini overlay and the SRF verification Python environment. Its fixture is read from the saved [WIVR run](../WIVR_test/results/run_20261120T145549Z/). The Python evaluator uses the repository's GP, scoring and SRF reference modules.

With the container running, invoke `docker exec bunker-jazzy bash '/workspace/Code/Verification tests/WIVR_test/code/run.sh'`. Run `code/report.py` with the reporting Python environment to refresh the notebook.
