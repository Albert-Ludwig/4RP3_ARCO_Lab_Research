# 11.2 - GP/GRG Policy Comparison

## Purpose and comparison

Compare how the GRG route-scoring objective affects temperature-field reconstruction and future prediction while holding the GP, navigation, acquisition rules and budget fixed. This is a Python replay of Intel Lab measurements, separate from ROS 2/Nav2 simulation.

Sampling policies: **MI, EMSE, MI:EMSE 5:5, MI:EMSE 4:6, MI:EMSE 6:4, MI:EMSE 7:3**. A history-only GP is the no-new-measurement control. MI maximizes conditional information gain; EMSE maximizes average latent-variance reduction over a fixed target grid. Fusion uses `w_MI * MI/s_MI + w_EMSE * EMSE_reduction/s_EMSE`, with shared initial single-event normalizers fixed before sampling.

| Item | Setting |
|---|---|
| Data/model | Intel Lab; 2,000 observations from days 1-3; saved GP with fixed kernel/noise parameters |
| Missions | Day-four starts at 72, 80 and 88 h; same initial checkpoint per policy/window |
| Navigation | 29 sensor candidates; synthetic 30 x 30 m map; grid/Dijkstra navigation; GRG depth 2 |
| Budget | 900 s; 0.35 m/s; 5 s dwell; travel and acquisition waiting included; computation charged zero, wall time saved separately |
| Reconstruction | Latest held-out record per reachable sensor in the final five minutes; retrospective, at actual record timestamps |
| Forecast | Freeze final GP at the common deadline; test strictly later records in (0,15], (15,30], (30,60] min; no future updates |
| Metrics | Mean of three window RMSEs; MAE, sensor-balanced RMSE, coverage and per-window NLPD also saved |

## Dependencies

Paths below are relative to the repository root. Keep these sibling experiments and data files in place.

| File/directory | Required role |
|---|---|
| `Code/11.2_policy_compare/testing_code/` | Runners, objective scoring, split rules, verification and the single report builder |
| `Code/10.18_Robot_close_loop_system/testing_code/baseline/` | GP wrapper, GRG recursion, mission controller, grid navigation and output helpers |
| `Code/10.18_Robot_close_loop_system/testing_code/intel_lab/` | Intel loading, candidate construction and recorded measurement acquisition |
| `Code/10.16_GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/` | Shared GP/scaling/checkpoint loader and evaluation metrics |
| `Data/Intel_lab_data/processed/three_day_2000/` | `train_2000.csv`, `test_day4.csv`, `split.json` |
| `Code/10.18_Robot_close_loop_system/10.18_result/baseline/intel_lab/forecast/` | `initial_gp.pt`, `hyperparameters.json` |
| `Code/10.18_Robot_close_loop_system/10.18_result/baseline/intel_lab/replay/` | Historical MI control for regression verification |
| [requirements.txt](testing_code/requirements.txt), [requirements-lock.txt](testing_code/requirements-lock.txt) | Pinned numerical/reporting packages; saved runs use Python 3.12 and CPU PyTorch |

Use the existing Docker container `bunker-jazzy`, project mount `/workspace`, and Python `/home/ubuntu/.venvs/policy-compare/bin/python`. ROS Python is separate. Git history retains the superseded reports and verifies their original hashes. The local `GP-Planning` reference clone is not a runtime dependency.

## Run

From this folder, with Docker Desktop running, reproduce the five-policy comparison and then the 7:3 extension:

```powershell
powershell -ExecutionPolicy Bypass -File .\testing_code\Start-StrictCompare.ps1
powershell -ExecutionPolicy Bypass -File .\testing_code\Start-70-30Compare.ps1
```

Rebuild only the consolidated report from saved results:

```powershell
docker --context desktop-linux exec -u ubuntu bunker-jazzy /home/ubuntu/.venvs/policy-compare/bin/python -B /workspace/Code/11.2_policy_compare/testing_code/build_report.py
```

All report commands write only `README.md` and `11.2_test_report.ipynb`. The Notebook analyzes saved artifacts without running missions. The original three-policy runner remains available for regression reproduction.

## Results and records

- [Executed Notebook: results, figures and conclusions](11.2_test_report.ipynb)
- [Aggregate metrics](11.2_result/strict_plus_70_30_20261103T025338Z/aggregate.csv), [per-window metrics](11.2_result/strict_plus_70_30_20261103T025338Z/evaluation.csv)
- [Frozen settings](11.2_result/strict_plus_70_30_20261103T025338Z/config_frozen.json), [checks](11.2_result/strict_plus_70_30_20261103T025338Z/verification.json), [provenance](11.2_result/strict_plus_70_30_20261103T025338Z/manifest.json)
- `11.2_result/latest.json`, `latest_strict.json`, `latest_70_30.json` identify the initial, five-policy and six-policy records. This report uses `strict_plus_70_30_20261103T025338Z`.
- Raw runs retain measurements, paths, score components, GP checkpoints, predictions, logs and hashes. A failed initial setup attempt is retained for diagnosis. Current figures and report verification are in `11.2_result/report_figures/` and `11.2_result/report_verification.json`.

**Current finding:** EMSE has the lowest mean reconstruction RMSE; 6:4 has the lowest mean RMSE in all three forecast bins. These are exploratory results from three correlated windows of one dataset; 7:3 was added after inspecting earlier results. Separate-day/field validation is needed. Reconstruction covers recorded sensor locations, not dense measured map truth. Historical manifests remain unchanged; superseded report/source hashes are verified against Git history.
