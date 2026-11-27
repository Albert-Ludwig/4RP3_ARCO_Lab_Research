# C — Larger region

60 × 40 m with a 300 s budget including return. This scenario expands the occupancy map and Gazebo walls and ground plane. It doubles A's checkpoint coordinates, retains 10 sensing checkpoints plus Home, and prepares a new directed Nav2 graph. It keeps the 0.05 m map resolution, 5 m / 2 s scoring grid, field origin and Home. The GP starts with zero observations and a 25 °C prior mean. The shared export supplies the RBF length scales, signal variance, and measurement noise variance. Its working region contains A's region in the same physical coordinates.

[Scenario configuration](code/scenario.json) · [Run entry point](code/run.sh) · [Shared setup, launch and evaluation](../README.md#running-the-formal-experiments)

Use the entry point in this folder with the shared kernel export. Evaluate the full region, the region shared with A, and the added area.

Mission files go in esults/ and rosbag recordings in ecordings/.
