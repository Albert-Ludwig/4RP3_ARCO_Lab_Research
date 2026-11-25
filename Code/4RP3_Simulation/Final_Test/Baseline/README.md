# A — Baseline

30 × 20 m with a 300 s mission budget including return. Reuse the closed-loop ROS verification map, world, 10 sensing checkpoints plus Home, directed path graph, navigation configuration and 5 m / 2 s online scoring grid. The formal run starts with zero observations, a 25 °C GP prior mean, and a 1 °C² prior variance. It reads the fitted RBF length scales and measurement noise from the training export. Test B shares this geometry and graph.

[Scenario configuration](code/scenario.json) · [Run entry point](code/run.sh) · [Shared setup, launch and evaluation](../README.md#running-the-formal-experiments)

After the new kernel is ready, use the entry point in this folder and evaluate the full domain, the common 0–150 s interval and the region shared with C.

Mission files go in esults/ and rosbag recordings in ecordings/.
