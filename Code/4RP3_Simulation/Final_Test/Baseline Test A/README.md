# A — Baseline

30 × 20 m with a 300 s mission budget including return. Reuse the closed-loop ROS verification map, world, 10 sensing checkpoints plus Home, directed path graph, navigation configuration and 5 m / 2 s online scoring grid. The GP starts with zero observations and a 25 °C prior mean. The training export supplies the RBF length scales, signal variance, and measurement noise variance. Test B shares this geometry and graph.

[Scenario configuration](code/scenario.json) · [Run entry point](code/run.sh) · [Shared setup, launch and evaluation](../README.md#running-the-formal-experiments)

Use the entry point in this folder with the shared kernel export and evaluate the full domain, the common 0–150 s interval and the region shared with C.

Mission files go in `results/` and rosbag recordings in `recordings/`.
