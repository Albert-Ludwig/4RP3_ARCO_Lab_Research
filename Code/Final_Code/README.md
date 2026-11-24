# Final Codes and Trained Kernels

- [Integration_test](Integration_test/README.md) holds the final temperature-field reconstruction experiment.
- [Trained_Real_Data_Kernel](Trained_Real_Data_Kernel/README.md) holds the real-data RBF training and model artifacts.
- [Trained_timecost](Trained_timecost/README.md) holds the Nav2 travel-time fit and its training and validation data.

## Current mission protocol

The next MI, EMSE and WIVR runs share a 300 s deadline from departure to arrival at Home. All use the fitted `d_ij = L_ij / v + delta`, where `L_ij` is the Nav2 path length, `v = 0.891822 m/s` and `delta = 2.709215 s`. GRG scores the normal candidate routes and accepts a next waypoint only when its travel and onward return fit the remaining time. When no waypoint fits, Nav2 navigates to the fixed Home goal. The 1 Hz sampler remains active during return, and all valid return samples condition the final GP. This revision is pending integration and verification.

## Planner sources

| Policy | ROS 2 node | Verified run | Route score |
| --- | --- | --- | --- |
| MI | [nodes.cpp](MI/src/nodes.cpp) | [run_20261123T223950Z](<../Verification tests/MI_Test/README.md>) | Joint mutual information of predicted sampling events |
| EMSE | [nodes.cpp](EMSE/src/nodes.cpp) | [run_20261123T224551Z](<../Verification tests/EMSE_Test/README.md>) | Expected variance reduction averaged over the evaluation grid using fixed cell-area weights |
| WIVR | [nodes.cpp](WIVR/src/nodes.cpp) | [run_20261123T225159Z](<../Verification tests/WIVR_test/README.md>) | Expected variance reduction with adaptive weights based on cell area and current GP posterior variance |

Each policy folder contains the source used for its archived origin-Home run.

| File | Role |
| --- | --- |
| src/core.hpp | GP posterior, predicted 1 Hz sampling events, route scoring and GRG search |
| src/nodes.cpp | Sensor observations and GP updates, background GRG worker, Nav2 goals, and mission timing |
| src/matrix_threads.hpp | Eigen and OpenMP thread settings |
| src/CMakeLists.txt and src/package.xml | Build files for the well_e_closed_loop ROS 2 package and its closed_loop executable |
| protocol.json | Archived settings for the runs listed above, which used a 300 s sampling window and a separately timed return |

Each search uses a fixed GP snapshot. The worker takes the newest pending snapshot when it starts the next search. In the archived runs, the node handed off goals at waypoint arrivals and could continue a retained sampling route when its next leg fit. Home is reserved for the starting pose and final return. The ten original candidates supply the sampling destinations.

## Runtime requirements

These files implement the mission-level controller. It sends waypoint poses to Nav2 through NavigateToPose and requests paths through ComputePathToPose. Nav2 and the robot base driver handle trajectory following and motor commands.

- Build one policy at a time with the selected policy's src directory as the colcon base path. The three copies use the same ROS package and executable names. Build dependencies include C++17, Eigen, nlohmann/json, OpenMP, ament_cmake, rclcpp, rclcpp_action and ROS message packages.
- Before launching, set CL_OUTPUT to a writable run directory containing protocol.json, candidates.json and initial.json. The initial GP file supplies its fixed hyperparameters and conditioning samples. The node can use an existing graph.json or request the paths from Nav2.
- The verification launcher supplies Gazebo, a ROS simulation clock, Nav2 and map/TF configuration, the Bunker Mini simulation model and the temperature source. The node reads /simulation/ground_truth/odom and exchanges temperatures on /cl/query and /cl/temperature. The shared [fixture builder](<../Verification tests/sampling_fixture.py>) prepares the origin spawn and southwest wall clearance for the 30 × 20 m evaluation region.
- Physical robot deployment needs a compatible localizer, coordinate transform, timestamped temperature interface, and base driver. The saved node currently uses simulator odometry and the origin fixture's coordinate conversion.

See the linked verification reports for the full Gazebo run commands and results.
