# Continuous Sensing Verification

Python checks for along-path events, actual-input 1 Hz measurements, buffering, segment-end batch RBF GP updates, duplicate/invalid records and interruption handling. Uses the verified 30 x 30 m / 120 s SRF and saved Nav2 path geometry; no new ROS run or policy ranking.

Status: **48 checks passed.** 120 synthetic measurements, four batches; synthetic history sized to 980 and a hypothetical 3920 records. Frozen real-data RBF parameters are reused without training or real-map registration.

[Results and figures](test_report.ipynb) | [Protocol](testing_code/protocol.json) | [Components](testing_code/components.py) | [Dependencies](../SRF_test/testing_code/requirements.txt)

Run: `powershell -ExecutionPolicy Bypass -File .\testing_code\Run-ContinuousSensing.ps1` (existing Docker SRF and host reporting environments required).
