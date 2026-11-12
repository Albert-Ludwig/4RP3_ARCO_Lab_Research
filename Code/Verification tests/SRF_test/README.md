# SRF Verification

Verifies independent, on-demand GSTools Matern temperature queries: repeatability, arbitrary-coordinate sampling, separate measurement noise and bounded batch computation.

Current: **97 x 64 m, 120 s, full obstacle-free rectangle; 0.1 m spatial / 0.5 s temporal reference spacing.** All 38 checks passed. Single-query P95: **0.271 ms**; complete 0.1 m spatial slice: **7.795 s**; sampled process peak: **122.62 MiB**. Three slices were executed; all 241 complete slices would cost about **31.3 min** by linear extrapolation (not executed).

The 97 x 64 m / 900 s baseline and 30 x 30 m / 120 s results are preserved. All variants use identical field parameters/seeds and full rectangular SRF domains. This source-only test creates no Gazebo scene and does not measure ROS/GP/GRG cost.

[Results and three-run comparison](test_report.ipynb) | [Latest saved output](result/run_20261111T013002148927Z/summary.json) | [Protocol](testing_code/protocol.json)

Files: `testing_code/srf_field.py` provides the source; `verify.py` runs checks; `build_report.py` produces the single notebook; `result/` stores measurements and provenance. Historical temperature mean/variance come from `Data/Real_Data_Integration/result/gp_samples.csv`; field calibration remains provisional.

Run: `powershell -ExecutionPolicy Bypass -File .\testing_code\Run-SRF.ps1` (existing Docker SRF and host reporting environments required).
