# Final Codes and Trained Kernels

- [Integration_test](Integration_test/README.md): the final temperature-field reconstruction experiment.
- [Trained_Real_Data_Kernel](Trained_Real_Data_Kernel/README.md): real-data RBF GP training and saved model artifacts.
- [WIVR](WIVR/src/core.hpp): the C++ Adaptive Weighted Integrated Variance Reduction planner and its ROS 2 runtime.

Component checks and functional verification belong in [Verification tests](<../Verification tests/README.md>).

## Adaptive WIVR

[The planning core](WIVR/src/core.hpp) scores candidate routes using the expected reduction in GP posterior variance across a space-time grid. It weights each spatial cell by its area and current posterior variance. Each GRG search holds the GP snapshot and weights fixed and scores the measurements expected along the whole route at 1 Hz.

[The ROS 2 node](WIVR/src/nodes.cpp) incorporates incoming temperature observations into the GP and keeps one pending snapshot, replacing it when fresher data arrives. The background worker finishes the current search before starting from the latest snapshot. Nav2 switches goals at waypoint arrival. The node can continue the remaining route if a matching new plan is unavailable, if the next sampling leg fits within the remaining 300 s. The final return begins after sampling ends and is timed separately. Home supplies the initial pose and final return goal. GRG selects its sampling destinations from the ten other waypoints.

| File | Purpose |
| --- | --- |
| [core.hpp](WIVR/src/core.hpp) | GP posterior, predicted sensing events, Adaptive WIVR scoring and GRG search |
| [nodes.cpp](WIVR/src/nodes.cpp) | Background planning, ROS 2 sensing and Nav2 goal handoff |
| [matrix_threads.hpp](WIVR/src/matrix_threads.hpp) | Eigen and OpenMP thread settings |
| [CMakeLists.txt](WIVR/src/CMakeLists.txt), [package.xml](WIVR/src/package.xml) | ROS 2 package for the closed_loop executable |
| [protocol.json](WIVR/protocol.json) | Sampling budget and separate final return settings from the WIVR verification run |

The planning core uses C++17, Eigen and nlohmann/json. The ROS 2 package also uses OpenMP, ament_cmake, rclcpp, rclcpp_action and the std_msgs, geometry_msgs, nav_msgs and nav2_msgs packages. The node reads initial.json, candidates.json and protocol.json from CL_OUTPUT. At runtime it expects a ROS clock, odometry, a temperature source and Nav2 action servers. The verification setup uses a 30 × 20 m evaluation region, a separate origin Home, and ten candidates. Its southwest simulator walls provide clearance around the origin.

The source matches WIVR verification run `run_20261123T225159Z` with the sampling-only budget. See the [WIVR verification report](<../Verification tests/WIVR_test/README.md>) for the Gazebo setup, results and comparisons.
