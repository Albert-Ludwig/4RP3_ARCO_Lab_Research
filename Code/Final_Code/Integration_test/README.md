# Final Reconstruction Experiment

Reserved for the final ROS 2 / Gazebo / Nav2 experiment comparing MI, EMSE and Adaptive WIVR reconstruction under the shared continuous-sensing and SRF protocol.

## Mission protocol

- The mission has **300 s from departure to arrival at Home**, including the return. Each GRG waypoint must leave enough time for the Nav2 leg back to Home.
- Estimate travel with `d_ij = L_ij / v + delta`, using Nav2 path length and the [calibrated parameters](../Trained_timecost/README.md), `v = 0.891822 m/s` and `delta = 2.709215 s`.
- GRG scores the normal sampling route. When no next waypoint fits, Nav2 sends the robot back to Home. Continue 1 Hz sampling on the return and include those observations when evaluating the final GP posterior.

Status: Not started. Registered field map/data and verification readiness are pending. Preliminary functional tests belong in [Verification tests](<../../Verification tests/README.md>).
