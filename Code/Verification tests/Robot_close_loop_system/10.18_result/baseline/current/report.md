# Current baseline: consolidated implementation

## Scope

The latest implementation replaces the earlier code in [testing_code/baseline/](../../../testing_code/baseline/). The extra `revisit_enabled` code directory has been removed. There is one runtime implementation, not an old and a new version.

Retained changes:

- Revisit spatial locations at different times. Keep every observation in the joint covariance and GP dataset.
- Permit explicit Home measurements. A final return-only Home vertex does not force a measurement.
- Record travel, sampling and online-computation time. Online computation is currently 0 s.
- Permanently exclude only confirmed static-map collision or no-path cases. Temporary failures skip the next planning round and become eligible in the following round.
- Reserve the measurement-failure interface. No sensor-failure simulator or detector is added.

The GP, RBF kernel, sampling-point MI objective, candidate set and mission budget are unchanged. This consolidation does not implement ROS/Nav2 or target-field variance reduction.

## Verification

All 51 current unit and integration tests passed. All 6 historical benchmark artifact checks passed. Those six checks preserve saved results; they do not rerun the benchmark or validate its old no-revisit policy against current code. Test snapshots confirmed no source or historical result changes during verification.

See [checks.json](verification/checks.json), [tests.log](verification/tests.log), and [historical artifact checks](verification/historical_artifacts.json).

## Synthetic main run

| Item | Result |
|---|---:|
| Seed | 18 |
| Historical observations | 100 |
| Candidate points, including Home | 48 |
| Online observations | 15 |
| Home observations | 2 |
| Planning calls | 16 |
| Travel time | 823.858 s |
| Sampling time | 75 s |
| Online computation charged | 0 s |
| Total mission time | 898.858 / 900 s |
| Returned Home | Yes |
| Target time | 12.25 h |
| Target-field RMSE, before online sampling | 0.053705 °C |
| Target-field RMSE, after online sampling | 0.018171 °C |
| Final mean latent variance | 0.000770 °C² |

All 14 main-run acceptance checks passed. Results match the preceding failure-policy smoke run apart from wall-clock diagnostics. This verifies consolidation, not a new planner comparison. These are synthetic results.

Data and six figures: [main/](main/). Mission record: [mission_summary.json](main/mission_summary.json). Acceptance checks: [verification.json](main/verification.json).

## Files and commands

Run from the 10.18 experiment root:

```bash
python testing_code/baseline/experiment.py
python testing_code/baseline/verify.py
python testing_code/baseline/verify_artifacts.py
```

Current runs and checks are saved here. Earlier policy-update data are retained in [baseline/updates/](../updates/). Original baseline and benchmark data and Notebook outputs remain historical records, not results of the updated policy. No second copy of the old runtime code is retained.
