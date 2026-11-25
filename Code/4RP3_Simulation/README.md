# Final Codes and Trained Kernels

- [Nav2_ROS](Nav2_ROS/README.md) holds the shared ROS 2, Nav2, Gazebo and Bunker Mini simulation runtime.
- [Final_Test](Final_Test/README.md) holds the final temperature-field reconstruction experiment.
- [Trained_Real_Data_Kernel](Trained_Real_Data_Kernel/README.md) holds the real-data RBF training and model artifacts.
- [Trained_timecost](Trained_timecost/README.md) holds the Nav2 travel-time fit and its training and validation data.

## Current mission protocol

The verified MI, EMSE and WIVR runs share a 300 s deadline from departure to arrival at Home. All use the fitted `d_ij = L_ij / v + delta`, where `L_ij` is the Nav2 path length, `v = 0.891822 m/s` and `delta = 2.709215 s`. GRG scores the normal candidate routes and accepts a next waypoint only when its travel and onward return fit the remaining time. When no waypoint fits, Nav2 navigates to the fixed Home goal. The 1 Hz sampler remains active during return, and all valid return samples condition the final GP. The revised protocol passed all three [Gazebo verification runs](../Verification%20tests/README.md#return-inclusive-ros-closed-loop-verification). The final registered-site experiment is pending.

## Planner sources

| Policy | ROS 2 node | Verified run | Route score |
| --- | --- | --- | --- |
| MI | [nodes.cpp](MI/src/nodes.cpp) | [run_20261124T030829Z](<../Verification tests/MI_Test/results/run_20261124T030829Z/latest_gp/summary.json>) | Joint mutual information of predicted sampling events |
| EMSE | [nodes.cpp](EMSE/src/nodes.cpp) | [run_20261124T031414Z](<../Verification tests/EMSE_Test/results/run_20261124T031414Z/latest_gp/summary.json>) | Expected variance reduction averaged over the evaluation grid using fixed cell-area weights |
| WIVR | [nodes.cpp](WIVR/src/nodes.cpp) | [run_20261124T031956Z](<../Verification tests/WIVR_test/results/run_20261124T031956Z/latest_gp/summary.json>) | Expected variance reduction with adaptive weights based on cell area and current GP posterior variance |

Each policy folder contains the C++ controller package and mission protocol from the corresponding verified return-inclusive run.

| File | Role |
| --- | --- |
| src/core.hpp | GP posterior, predicted 1 Hz sampling events, route scoring and GRG search |
| src/nodes.cpp | Sensor observations and GP updates, background GRG worker, Nav2 goals, and mission timing |
| src/matrix_threads.hpp | Eigen and OpenMP thread settings |
| src/CMakeLists.txt and src/package.xml | Build files for the well_e_closed_loop ROS 2 package and its closed_loop executable |
| protocol.json | Verified settings for the 300 s return-inclusive mission, with the fitted travel-time model and 1 Hz sensing through Home arrival |

Each search uses a fixed GP snapshot. The worker takes the newest pending snapshot when it starts the next search. The node hands off goals at waypoint arrivals and can continue a retained route when its next leg and the onward return fit the remaining time. Home is reserved for the starting pose and final return. The ten original candidates supply the sampling destinations.

## Runtime requirements

These files implement the mission-level controller. It sends waypoint poses to Nav2 through NavigateToPose and requests paths through ComputePathToPose. Nav2 and the robot base driver handle trajectory following and motor commands.

- Build one policy at a time with the selected policy's src directory as the colcon base path. The three copies use the same ROS package and executable names. Build dependencies include C++17, Eigen, nlohmann/json, OpenMP, ament_cmake, rclcpp, rclcpp_action and ROS message packages.
- Before launching, set CL_OUTPUT to a writable run directory containing protocol.json, candidates.json and initial.json. The initial GP file supplies its fixed hyperparameters and conditioning samples. The node can load path geometry from graph.json or request paths from Nav2. In both cases, it computes edge durations using the fitted protocol parameters.
- The [shared runtime](Nav2_ROS/README.md) provides Gazebo, the ROS simulation clock, Nav2, the map and TF, the Bunker Mini model and the temperature source. The controller reads /simulation/ground_truth/odom and exchanges temperatures on /cl/query and /cl/temperature.
- Physical robot deployment needs a compatible localizer, coordinate transform, timestamped temperature interface, and base driver. The saved node currently uses simulator odometry and the origin fixture's coordinate conversion.

Use the [shared build and run instructions](Nav2_ROS/README.md#build-and-start). The saved verification runs are linked above.
