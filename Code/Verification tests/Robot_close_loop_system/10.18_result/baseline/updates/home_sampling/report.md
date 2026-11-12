# Home sampling enabled

Historical policy-update record. The runtime code has since been consolidated into `testing_code/baseline/`. Source links point to current files. Use the [current report](../../current/report.md) for the active implementation and output paths.

This report records the Home-sampling revision. The current entry also records an explicit mission-time model; see [its report](../mission_time_model/report.md). The data and logs below remain unchanged.

## Change

Home is an eligible measurement location. The planner may select it at mission start, as an intermediate waypoint, or after returning. Home measurements use actual arrival position and time, incur the normal 5 s dwell, enter the GP posterior, and participate in the full joint MI. No measurement is forced merely because the mission starts or ends at Home.

Spatial revisits remain enabled. No minimum revisit interval or uniqueness filter is added. The GP, kernel, noise, MI formula, navigation map, candidate coordinates, mission budget and failure policy are unchanged.

## Route semantics

- `[0, 0, 0]`: start at Home, sample at Home, finish at Home. One measurement and one dwell.
- `[A, 0, 0]`: travel from A, sample at Home, then finish. One Home measurement.
- `[A, 0]`: return only. No measurement.
- `[0, A, 0, B, 0]`: sample A, Home, B; the final Home vertex is return-only.

Recursive subproblems explicitly mark whether the endpoint is sampled. Timing includes dwell at sampled Home endpoints. Cache keys contain the ordered route and endpoint role; a Home measurement cannot reuse the reward of a return-only endpoint.

## Implementation

Code: [testing_code/baseline/](../../../../testing_code/baseline).

- `planner.py`: Home-inclusive eligibility, role-aware scheduling, joint scoring and recursive deadlines.
- `controller.py`: the original controller execution logic, with Home included in candidate logs. Navigation, measurement validation, GP append and return checks are retained.
- `run.py`: injects these components into a separate instance of the existing experiment. Historical module bindings are not changed.
- `verify.py`, `test_revisit.py`: current checks and historical regression preservation.

The prior baseline, benchmark, recursive-verification results and overall Notebook remain unchanged. Earlier revisit-only smoke results remain in the parent folder. This revision does not rerun the depth benchmark.

## Verification

All **15 current tests** passed. Historical baseline tests passed **22/22** and benchmark checks passed **6/6**. [Checks and evidence](verification/checks.json) record the source hashes, logs and unchanged historical files.

Checks cover initial Home eligibility, exact dwell-budget boundaries, Home sampling after travel, intermediate versus terminal Home roles, depth-1/2/3 arrival and MI consistency, cache-disabled equivalence, online revisits, posterior updates and collision-free return.

A Home-only 10 s closed-loop fixture collected two observations at `12 h` and `12 h + 5 s`, appended both to the GP, and finished at 10 s. No extra measurement occurred at final termination. A separate return-only fixture confirms that final arrival at Home does not automatically trigger sensing.

## Synthetic main-run smoke test

The 48-node seed-18 run completed **15 measurements** and **16 planning calls**. It finished at **898.857967 s**, within the **900 s** budget. All **14** saved mission acceptance checks passed.

Two measurements were taken at Home:

| Arrival elapsed time | GP timestamp | Dwell |
|---:|---:|---:|
| 888.857967 s | 12.246904991 h | 5 s |
| 893.857967 s | 12.248293880 h | 5 s |

The final target-field RMSE was **0.018171 °C** at **12.25 h**. This is one synthetic smoke run, not a multi-seed performance comparison. Data and six figures are in [main/](main).

## Commands

From the 10.18 experiment root:

```bash
python testing_code/baseline/experiment.py
python testing_code/baseline/verify.py
```

Default outputs are isolated under `10.18_result/baseline/updates/home_sampling/`.
