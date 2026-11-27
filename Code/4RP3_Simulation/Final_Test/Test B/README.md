# B — Short mission

30 × 20 m with a 150 s budget including return. This scenario uses A's exact navigation geometry, Home and candidate set. The GP starts with zero observations and a 25 °C prior mean. The shared export supplies the RBF length scales, signal variance, and measurement noise variance. The mission budget and predicted-event horizon are 150 s. Keep the 5 m scoring cells and use the 76 times from 0 to 150 s at 2 s spacing. Evaluate the common 0–150 s domain against A.

[Scenario configuration](code/scenario.json) · [Run entry point](code/run.sh) · [Shared setup, launch and evaluation](../README.md#running-the-formal-experiments)

Use the entry point in this folder with the shared kernel export. Compare the final posterior with A on the common 0–150 s grid.

Mission files go in esults/ and rosbag recordings in ecordings/.
