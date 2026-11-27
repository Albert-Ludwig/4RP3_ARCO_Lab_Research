# WELL-E simulation code guide

The nine formal missions used **one C++ controller built from [WIVR/src](WIVR/src/)**. For each mission, [run.py](Final_Test/code/run.py#L62) writes `mi`, `emse`, or `wivr` into the protocol. The [mission node](WIVR/src/nodes.cpp#L45) reads that value and passes it to the same GP and GRG implementation. The [MI](MI/src/) and [EMSE](EMSE/src/) directories preserve the controllers from the earlier individual policy tests. All three source trees declare the same ROS package and executable names.

[Formal experiment design](Final_Test/README.md) · [Executed comparison notebook](Final_Test/Comparison.ipynb) · [Formal RMSE results](Final_Test/results/formal_all_runs/evaluation.json)

## Run the nine formal missions on the lab workstation

The commands below use the existing `bunker-jazzy` container (ROS 2 Jazzy and Gazebo Harmonic). It mounts the full 4RP3 repository at `/workspace` and the ROS asset directory at `/ros-data`. The recorded comparison is `formal_all_runs`. Use a fresh run name for a repeat so the earlier measurements and recordings remain available.

### 1. Open the container from Windows PowerShell

Start Docker Desktop and wait for its engine to be ready. Run this block in **PowerShell**. Run the remaining blocks in the **same Bash session inside the container**.

```powershell
docker start bunker-jazzy
docker exec -it bunker-jazzy bash
```

### 2. Build and load the formal controller

```bash
set -e
source /opt/ros/jazzy/setup.bash
sim=/workspace/Code/4RP3_Simulation
export ROS_HOME=/ros-data
export ROS_DOMAIN_ID=106
export PYTHON_BIN=/home/ubuntu/.venvs/srf-verification/bin/python
export OPENBLAS_NUM_THREADS=1

build_dir=$(mktemp -d /tmp/welle-formal-build.XXXXXX)
cd "$build_dir"
colcon build --base-paths "$sim/Nav2_ROS/src" "$sim/WIVR/src" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source "$build_dir/install/setup.bash"

cd "$sim/Final_Test"
"$PYTHON_BIN" ../Nav2_ROS/tools/fetch_asset.py
ros2 pkg executables well_e_closed_loop
```

The executable list should contain `well_e_closed_loop closed_loop`. The asset helper verifies the cached Bunker Mini mesh and downloads the pinned copy if missing.

### 3. Select the kernel and start a new comparison

```bash
kernel=../Trained_Kernel/result/multirun_fit_20261127T025820Z/hyperparameters.json
run_name="formal_$(date -u +%Y%m%dT%H%M%SZ)"
printf 'Run name: %s\n' "$run_name"
printf '%s  %s\n' 9996fb3cde538d004ba134f7345315467ce24949a0711982c5c08db854136276 "$kernel" | sha256sum -c -
```

This checksum identifies the kernel fitted to all **6,448 real observations**. Each mission starts with zero observations and a 25 °C prior mean. A uses 30 × 20 m / 300 s, B uses 30 × 20 m / 150 s, and C uses 60 × 40 m / 300 s. The first launch freezes shared inputs under `results/$run_name/`. Keep this run name for all nine missions and the evaluation.

### 4. Run MI, EMSE and Adaptive WIVR with rosbag recording

Paste the whole block. It runs one mission at a time. The runner returns a nonzero exit code for a missed deadline. The loop keeps a recorded mission outcome and proceeds to the next policy. A launch error without a mission record stops the batch and prints the log directory.

```bash
for group in "Baseline Test A" "Test B" "Test C"; do
  for policy in mi emse wivr; do
    out="$group/results/$run_name/$policy"
    printf '\nRunning %s / %s\n' "$group" "$policy"
    if ! bash "$group/code/run.sh" --kernel "$kernel" --name "$run_name" --policy "$policy" --record; then
      if [[ ! -s "$out/mission.json" || ! -s "$out/run_status.json" ]]; then
        printf 'Launch or recording error. Inspect %s\n' "$out" >&2
        exit 1
      fi
      cat "$out/run_status.json"
    fi

    # If the first C mission returned late, reuse its completed Nav2 graph for the next policies.
    graph="results/$run_name/navigation/large/graph.json"
    if [[ "$group" == "Test C" && ! -e "$graph" && -s "$out/graph.json" && -s "$out/mission.json" ]]; then
      cp "$out/graph.json" "$graph"
    fi
  done
done
```

For a single mission, use the same entry point and pass one policy, e.g. `bash "Test B/code/run.sh" --kernel "$kernel" --name "$run_name" --policy emse --record`. If an attempt was interrupted, inspect its `run_status.json` first. To finish the comparison, run the missing scenario/policy pairs under the same name. To repeat an existing failed attempt, choose a new comparison name. The runner preserves existing mission directories.

### 5. Compute RMSE and inspect the results

```bash
"$PYTHON_BIN" code/evaluate.py --name "$run_name"
"$PYTHON_BIN" - "$run_name" <<'PY'
import json
import sys
from pathlib import Path

run = sys.argv[1]
evaluation = json.loads((Path("results") / run / "evaluation.json").read_text())
print("Case  Policy  Status       Home(s)   Full RMSE(C)")
for case, group in [("A", "Baseline Test A"), ("B", "Test B"), ("C", "Test C")]:
    for policy, row in evaluation[case].items():
        status = Path(group) / "results" / run / policy / "run_status.json"
        mission = json.loads(status.read_text()).get("mission_summary", {}) if status.exists() else {}
        arrival = mission.get("home_arrival_t")
        score = row.get("domains", {}).get("full", {}).get("rmse_c")
        home = f"{arrival:.2f}" if arrival is not None and arrival >= 0 else "--"
        rmse = f"{score:.5f}" if score is not None else "--"
        print(f"{case:4}  {policy:6}  {row['status']:10}  {home:>7}   {rmse:>12}")
PY
```

- **Shared inputs and scores** are in `Final_Test/results/<run_name>/`. The evaluator writes `evaluation.json` with full-field RMSE, A/B common-time scores, A/C common-space scores and C's added-area score. Only accepted missions receive a formal score.
- **Mission evidence** is in `Final_Test/<group>/results/<run_name>/<policy>/`. Read `run_status.json` for acceptance, `mission.json` for actual arrival time and valid samples, and `final_gp.json` for the posterior inputs. MCAP recordings are in `Final_Test/<group>/recordings/<run_name>/<policy>/rosbag/`. For example, run `ros2 bag info "Baseline Test A/recordings/$run_name/mi/rosbag"` inside the container.
- **The saved report** is [Comparison.ipynb](Final_Test/Comparison.ipynb). Open it on Windows to view the completed `formal_all_runs` figures. Its reconstruction slices and A/MI overtime diagnostic are supplemental files specific to that run. For a fresh run the commands above produce the formal scores and the terminal summary.

## Follow one mission

| Stage | Source | What it does and who uses it |
|---|---|---|
| Select the scenario | [A entry point](<Final_Test/Baseline Test A/code/run.sh>), [B entry point](<Final_Test/Test B/code/run.sh>), [C entry point](<Final_Test/Test C/code/run.sh>) | Each shell script calls `Final_Test/code/run.py` with its scenario ID. The adjacent `scenario.json` supplies the field size and mission budget. [settings.json](Final_Test/settings.json) supplies the common Home, checkpoint count, grid spacing, GP update interval, prior mean and search timeout. |
| Prepare shared inputs | [prepare.py](Final_Test/code/prepare.py) | On the first launch for a run name, reads the trained kernel, creates an empty GP with mean 25 °C, freezes the SRF and records input hashes. A/B reuse verified checkpoints, map and path graph. C scales the checkpoints and generates its arena. |
| Start and record the mission | [run.py](Final_Test/code/run.py#L58) | Starts the Gazebo/Nav2 scene, synthetic temperature source, resource monitor and C++ controller. With `--record`, starts the MCAP recorder. Saves each mission's status, return and deadline outcome. |
| Plan and drive | [core.hpp](WIVR/src/core.hpp), [nodes.cpp](WIVR/src/nodes.cpp) | The core selects a checkpoint route. The mission node sends the next goal to Nav2, requests temperature at the actual simulated pose, updates the GP and schedules the next search. Target changes occur at waypoint arrivals. |
| Evaluate and report | [evaluate.py](Final_Test/code/evaluate.py), [Comparison.ipynb](Final_Test/Comparison.ipynb) | After the mission, rebuilds the final GP and compares it to noise-free SRF truth on 2 m cell centres every 5 s. The notebook reads the saved results. The A/MI overtime score is stored [separately](Final_Test/results/formal_all_runs/overtime_rmse.json) and labelled in the report. |

The runner and evaluator use the [scenario folder mapping](Final_Test/code/prepare.py#L21) to locate each group's configuration, mission results and recordings.

## GP, route scoring and GRG

| Calculation | Implementation | Use in the formal controller |
|---|---|---|
| Online GP | [GP in core.hpp](WIVR/src/core.hpp#L36) | `posterior()` returns predictive mean and variance. `append()` incorporates valid measurements. `snapshot()` captures the GP state used by one search. Kernel hyperparameters stay fixed during navigation. |
| Predicted route samples | [Search::events](WIVR/src/core.hpp#L176) | Converts the normal sampling legs into predicted 1 Hz spatiotemporal events. GRG reserves time for the final return. Actual return measurements enter the GP. |
| MI | [Search::score, MI branch](WIVR/src/core.hpp#L287) | Scores joint information gain from predicted events using their covariance and measurement noise. |
| EMSE | [Cell weights](WIVR/src/core.hpp#L153) and [variance reduction](WIVR/src/core.hpp#L291) | Scores expected posterior variance reduction over 5 m cell centres at 2 s intervals, using effective cell area as the spatial weight. |
| Adaptive WIVR | [Adaptive weights](WIVR/src/core.hpp#L161) and [weighted reduction](WIVR/src/core.hpp#L291) | Multiplies cell area by GP posterior variance at the snapshot's reference time. Normalized weights stay fixed during that search and are reused across scoring times. |
| GRG route selection | [Search::run and recurse](WIVR/src/core.hpp#L305) | Recursively inserts checkpoints, calls `score()` on candidate routes and checks estimated travel time against the remaining budget. All three policies use this search. |
| Background planning and execution | [offer_latest, update_gp and process_jobs](WIVR/src/nodes.cpp#L109), [dispatch and feasibility checks](WIVR/src/nodes.cpp#L171) | Keeps the newest pending GP state while a search runs. Checks the next leg and onward return before dispatching a goal. [poll()](WIVR/src/nodes.cpp#L268) drives sampling, waypoint handoff and return completion. |

[WIVR/src/CMakeLists.txt](WIVR/src/CMakeLists.txt) builds `well_e_closed_loop/closed_loop` from `nodes.cpp`. [package.xml](WIVR/src/package.xml) declares ROS dependencies, and [matrix_threads.hpp](WIVR/src/matrix_threads.hpp) configures Eigen/OpenMP threads. The per-policy `protocol.json` files document verification settings. For the actual formal settings, inspect the [frozen scenario protocols](Final_Test/results/formal_all_runs/scenarios/).

## Nav2, Gazebo and simulated sensing

| Component | Source and use |
|---|---|
| Navigation launch and configuration | [scene.launch.py](Nav2_ROS/launch/scene.launch.py) loads the scenario map and world, starts localization and Nav2, and connects velocity topics. [nav2_params.yaml](Nav2_ROS/config/nav2_params.yaml) selects NavFn A* and Regulated Pure Pursuit and configures costmaps, tolerances and velocity smoothing. [navigation_profile.json](Nav2_ROS/config/navigation_profile.json) supplies body speed limits. ROS 2 provides the Nav2 plugins. |
| Path graph and waypoint commands | [Experiment::prepare](WIVR/src/nodes.cpp#L77) reads cached edge geometry or requests it through `ComputePathToPose`. [Experiment::dispatch](WIVR/src/nodes.cpp#L171) sends the selected checkpoint or Home through `NavigateToPose` and handles arrival or failure. |
| Bunker Mini and simulator interfaces | [sim.launch.py](Nav2_ROS/src/well_e_bunker_sim/launch/sim.launch.py) uses [model.py](Nav2_ROS/src/well_e_bunker_sim/launch/model.py), [drive settings](Nav2_ROS/src/well_e_bunker_sim/config/drive.json) and the [ROS–Gazebo bridge](Nav2_ROS/src/well_e_bunker_sim/config/bridge.yaml). It starts the [command watchdog](Nav2_ROS/src/well_e_bunker_sim/scripts/cmd_watchdog.py) and [odometry adapter](Nav2_ROS/src/well_e_bunker_sim/scripts/odometry_adapter.py). [truth_localization.py](Nav2_ROS/src/well_e_nav2_test/scripts/truth_localization.py) supplies the simulator-assisted map-to-odom correction. |
| Map, world and robot mesh | [world.sdf](Nav2_ROS/worlds/world.sdf) defines the tracked robot and arena. [maps/](Nav2_ROS/maps/) contains the standalone fixture map. Formal launches use the [prepared scenario geometry](Final_Test/results/formal_all_runs/navigation/). [fetch_asset.py](Nav2_ROS/tools/fetch_asset.py) supplies the manufacturer mesh expected under `ROS_HOME/assets/bunker_mini/`. |
| Synthetic temperature observations | [srf_field.py](Nav2_ROS/tools/srf_field.py) evaluates the shared Matérn field and adds reproducible sensor noise using [field.json](Nav2_ROS/fixtures/field.json). [srf_monitor.py](Nav2_ROS/tools/srf_monitor.py) answers `/cl/query` on `/cl/temperature`; its separate monitor process records resource use. The C++ sampler uses `/simulation/ground_truth/odom` and elapsed mission time. |

## Offline calibration and dependencies

- **GP hyperparameters.** [fit_multirun.py](Trained_Kernel/testing_code/fit_multirun.py) fits the RBF kernel using six real traversals and 6,448 aligned observations. The formal runner receives the [exported hyperparameters](Trained_Kernel/result/multirun_fit_20261127T025820Z/hyperparameters.json) through `--kernel`. [build_report.py](Trained_Kernel/testing_code/build_report.py) produces the training report. [gp_model.py](Trained_Kernel/testing_code/gp_model.py) and [Train-Kernel.ps1](Trained_Kernel/testing_code/Train-Kernel.ps1) belong to the earlier training workflow. See the [multi-run training instructions](Trained_Kernel/README.md) and [aligned input data](<../../Data/Interpolated Data/>).
- **Travel time.** [fit_timecost.py](Trained_timecost/fit_timecost.py) fits travel duration from saved navigation records and writes [model.json](Trained_timecost/model.json). At runtime, [travel_time()](WIVR/src/nodes.cpp#L73) reads the calibrated values from the mission protocol and computes `L / 0.891822 + 2.709215` seconds for each travel leg.
- **Repository dependencies outside this folder.** [prepare.py](Final_Test/code/prepare.py#L18) reads the reference protocol, checkpoints, occupancy map and directed path graph from the [saved WIVR verification run](<../Verification tests/WIVR_test/results/run_20261124T031956Z/latest_gp/>). [evaluate.py](Final_Test/code/evaluate.py#L15) imports the offline `BatchGP` from [components.py](<../Verification tests/Continuous_sensing_test/testing_code/components.py>). Keep both paths available to reproduce the full workflow.
- **Software environment.** The formal build uses `Nav2_ROS/src` and `WIVR/src` in the same ROS 2 Jazzy overlay, Gazebo Harmonic, and the SRF Python environment. See the [runtime dependencies](Nav2_ROS/README.md#build-and-start) and [formal build and run instructions](Final_Test/README.md#running-the-formal-experiments).
