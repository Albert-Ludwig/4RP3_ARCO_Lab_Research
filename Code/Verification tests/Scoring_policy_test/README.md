# Scoring Policy Verification

Verifies MI, area-averaged EMSE and Adaptive WIVR: 10 m grid areas, 0.5 s scoring times, weight refresh/freeze, exact along-path events, recursive GRG dispatch and cache isolation. Uses synthetic GP conditioning from stage 2; no reconstruction ranking or ROS execution.

Status: **71 checks and 6 independent returned-route audits passed.** All 30/50-Checkpoint searches completed. The 50-point EMSE/WIVR component-plus-motion estimate exceeds the nominal 120 s budget; full mission acceptance remains pending.

[Results and figures](test_report.ipynb) | [Protocol](testing_code/protocol.json) | [Scoring](testing_code/scoring.py) | [GRG adapter](testing_code/routing.py)

Run: `powershell -ExecutionPolicy Bypass -File .\testing_code\Run-ScoringPolicy.ps1` (existing Docker SRF and host reporting environments required).
