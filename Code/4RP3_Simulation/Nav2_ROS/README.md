# WELL-E ROS 2 and Nav2 Runtime

Shared simulation runtime for MI, EMSE and Adaptive WIVR. It packages the 30 × 20 m Gazebo Harmonic verification fixture, with Home at `(0, 0)` and a 0.05 m occupancy map. The map and runtime settings come from the 24 November 2026 return-inclusive runs. This setup uses Gazebo ground truth for the `map → odom` correction and for the mission sampler's pose. Physical WELL-E deployment needs a real localization and base driver stack.

| Location | Role |
| --- | --- |
| [launch/scene.launch.py](launch/scene.launch.py) | Starts Gazebo, the ROS–Gazebo bridge, localization, Nav2 and lifecycle activation. Resolves the map and Bunker mesh paths at launch. |
| [config/](config/) | Nav2 parameters and Bunker velocity limits. The global planner is NavFn A* and the controller is Regulated Pure Pursuit. Both costmaps use static and inflation layers. |
| [maps/](maps/) and [worlds/](worlds/) | Matched occupancy map and Gazebo world with the tracked Bunker Mini and physical obstacles. |
| [src/](src/) | Bunker simulation package with URDF generation, bridge, command watchdog and track odometry, plus the ground-truth localization node. |
| [tools/](tools/) | Pinned manufacturer mesh fetcher and the ROS temperature source, SRF implementation and optional process monitor. |
| [fixtures/](fixtures/) | Shared candidates, initial GP state and SRF parameters used by the verification runs. |

The ROS action clients live in the policy packages because they apply GRG decisions, enforce the mission budget and handle arrival and return. Choose one policy package per build. [MI](../MI/src/nodes.cpp) · [EMSE](../EMSE/src/nodes.cpp) · [WIVR](../WIVR/src/nodes.cpp)

A ROS 2 Jazzy smoke check from this directory activated the Nav2 lifecycle, received a temperature response and successfully navigated from Home to (3, 4) m.

## Build and start

Use Bash in ROS 2 Jazzy with Gazebo Harmonic. The runtime uses `navigation2`, `nav2_bringup`, `ros_gz`, `robot_state_publisher` and colcon. The policy package also requires Eigen 3, nlohmann/json and OpenMP. The temperature source requires Python with `rclpy`, `numpy`, `gstools` and `psutil`. Use the existing SRF virtual environment or a compatible environment with access to the ROS 2 Python packages. Rosbag recording is optional and uses the MCAP storage plugin.

From this directory, run the following in a fresh shell:

```bash
source /opt/ros/jazzy/setup.bash
export NAV2_ROS="$PWD"
export ROS_HOME="${ROS_HOME:-$HOME/.ros}"
export CL_OUTPUT="$(mktemp -d /tmp/well-e-mission.XXXXXX)"
policy=MI  # MI, EMSE or WIVR
python3 "$NAV2_ROS/tools/fetch_asset.py"
cp "$NAV2_ROS/fixtures/"*.json "$CL_OUTPUT/"
cp "$NAV2_ROS/../$policy/protocol.json" "$CL_OUTPUT/"
colcon --log-base "$CL_OUTPUT/log" build \
  --base-paths "$NAV2_ROS/src" "$NAV2_ROS/../$policy/src" \
  --build-base "$CL_OUTPUT/build" --install-base "$CL_OUTPUT/install"
source "$CL_OUTPUT/install/setup.bash"
ros2 launch "$NAV2_ROS/launch/scene.launch.py"
```

Set the same `NAV2_ROS`, `CL_OUTPUT`, `ROS_HOME` and optional `ROS_DOMAIN_ID` in each additional shell. Source ROS Jazzy and the same colcon overlay. Start the temperature source (using its Python environment), then start the selected controller after the Nav2 lifecycle nodes are active.

```bash
# Shell 2, with the SRF Python environment available
python "$NAV2_ROS/tools/srf_monitor.py"

# Shell 3, after Nav2 is active
ros2 run well_e_closed_loop closed_loop --ros-args -p use_sim_time:=true
```

Pass `gui:=true` to the scene launch for a Gazebo window. Use a fresh `CL_OUTPUT` for each run. The controller computes its Nav2 graph when there is no `graph.json` in that directory. With the pre-existing mesh cache, the smoke check can run without a network connection.

To record motion and sampling, start the recorder before the controller:

```bash
ros2 bag record -s mcap -o "$CL_OUTPUT/rosbag" \
  /clock /odom /tf /tf_static /simulation/ground_truth/odom \
  /cl/query /cl/temperature /cmd_vel
```
