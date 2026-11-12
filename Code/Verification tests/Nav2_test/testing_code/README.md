# Nav2 Test Code

This stage runs a known-map point-to-point Nav2 test using the calibrated Bunker Mini package from `../../BunkerMini_drive_test/testing_code/src/well_e_bunker_sim`.

`src/well_e_nav2_test/config/scenario.json` defines the field, central obstacle, goal, conservative full-body footprint and acceptance criteria. `nav2_test.launch.py` creates matching Gazebo and occupancy-map geometry and launches NavFn, Regulated Pure Pursuit, the behavior-tree navigator and a map server.

`truth_localization.py` supplies an explicit ground-truth `map→odom` correction. This isolates navigation planning/control from localization error. No LiDAR, depth camera, AMCL or live obstacle detector is used; the obstacle is known before planning.

`navigation_probe.py` requests a plan and sends one `NavigateToPose` goal. It records actual simulator body poses, evaluates full-body footprint clearance and checks arrival and stopping. It sends no intermediate waypoints or manual velocity commands.

From `Nav2_test/` on Windows:

```powershell
.\testing_code\Start-Nav2.ps1 -Action test -Gui
.\testing_code\Start-Nav2.ps1 -Action sim
```

Inside the container: `bash testing_code/run_test.sh [--gui]`. ROS domain: 72; Gazebo partition: `well_e_nav2_test`. Each run gets a separate result directory.

`tools/report.py` replays saved results to generate the English README and Notebook. `tools/verify_results.py` checks geometry fixtures and saved records without running ROS.

Each automatic test now starts `ros2 bag record` before sending the navigation goal and closes it before simulation shutdown. The primary ROS recording is `../result/run_.../rosbag/` (MCAP plus metadata). It includes clock, map, robot description, static/dynamic TF, odometry, velocity commands, plan, costmaps, mirrored target pose, and action feedback/status. Joint states or other selected optional topics are recorded when publishers exist; no camera/LiDAR data is fabricated.

`tools/rosbag_capture.py` checks every recorded message, cross-checks physical poses and minimum clearance against the analysis CSV, and performs actual ROS replay on isolated domain 73. Visualization replay excludes velocity commands. CSV/JSON remain analysis outputs; they do not replace the raw bag.

Unmapped obstacles, real sensing/localization, ordered waypoints, unreachable goals, and explicit cancellation tests remain pending. Stopping after a successful goal is checked here.
