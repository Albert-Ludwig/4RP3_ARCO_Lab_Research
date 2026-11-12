# Spatial revisits enabled

Historical policy-update record. The runtime code has since been consolidated into `testing_code/baseline/`. Source links point to current files. Use the [current report](../current/report.md) for the active implementation and output paths.

This is the historical revisit-only revision. Current code also supports Home sampling; see [the current report](home_sampling/report.md). The data and test logs below describe the earlier revision, not the current entry point.

## Authorized change

Sampling the same spatial position at different times is permitted. The current planner does not exclude previously sampled online nodes or older route-prefix nodes. No minimum revisit interval, spatial uniqueness requirement, or post-search deduplication is added.

The current code is [testing_code/baseline](../../../testing_code/baseline). Historical baseline and benchmark result data are retained. The old runtime code has been replaced. The earlier no-revisit verification is archival; its spatial-uniqueness failure is not a failure criterion for this version.

## Implementation boundary

- `planner.py` reuses the original scheduling and joint-MI functions. Its recursive search removes the visited-set exclusion and permits earlier prefix nodes as intermediate waypoints.
- Immediate start/end vertices are still omitted as degenerate recursive split choices. Revisits through another waypoint are allowed. This change does not add a wait-in-place action or Home measurements.
- The original GP, navigation, sensor, evaluation, and mission controller are reused. The controller retains visited locations for records, but its candidate logging no longer excludes them when the planner permits revisits.
- `run.py` loads a separate instance of the existing experiment and injects the new planner. It does not rebind the historical experiment module.
- The configuration differs from the historical main configuration only in `allow_revisit: true`.
- Blocked/unreachable-node handling, budget checks, dwell, collision checking, return-home reserve, GP parameters, reward and other experiment assumptions are unchanged.

## Verification

All **8 new tests** passed. Historical baseline tests passed **22/22**; benchmark checks passed **6/6**. [Verification logs and evidence](verification/checks.json) include the frozen-file checks.

The tests verify previously sampled-node eligibility, `A → B → A` within a route, distinct timestamps, full joint covariance without deduplication, posterior updates, budget/blocked/unreachable constraints, and safe closed-loop return. They also check that the historical experiment module is not rebound.

The small closed-loop fixture used the existing GP checkpoint and original map. It collected **9 observations at 2 spatial positions**, conditioned on every observation, and returned at **562.77 s** within a **600 s** budget. Revisited positions had distinct timestamps. This is a functionality test, not a field-reconstruction performance claim.

## Synthetic main-run smoke test

The standard 48-node, seed-18 configuration completed **12 measurements** and **13 planning calls**. It returned at **885.30 s** within **900 s**. All **14** saved mission checks passed. This run did not revisit a measured position; permitting revisits does not force them.

Final target-field RMSE was **0.023728 °C** at **12.25 h**. Main-run data and six figures are in [main/](main). No depth benchmark was rerun and no planner-performance improvement is claimed.

## Commands

From the 10.18 experiment root:

```bash
python testing_code/baseline/experiment.py
python testing_code/baseline/verify.py
```

These saved outputs belong to the earlier revision; new outputs go to `10.18_result/baseline/current/`. The overall historical Notebook was not rewritten with new-policy results.
