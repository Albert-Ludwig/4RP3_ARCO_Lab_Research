# Explicit mission-time model

Historical policy-update record. The runtime code has since been consolidated into `testing_code/baseline/`. Source links point to current files. Use the [current report](../../current/report.md) for the active implementation and output paths.

This report records the mission-time revision. The current implementation also classifies permanent and temporary candidate failures; see [the failure-policy report](../failure_policy/report.md). The results below are retained.

## Scope

Use the requested decomposition:

$$
T_{\mathrm{mission}}=T_{\mathrm{travel}}+T_{\mathrm{sampling}}+T_{\mathrm{online\ computation}}.
$$

For the current Python simulation, `T_online_computation = 0 s`. Only the data structure is reserved for future integration. Spatial revisits and Home sampling remain enabled. No GP, MI, route-search, sensing, navigation or failure policy changes are included.

## Data structure and accounting

[timing.py](../../../../testing_code/baseline/timing.py) defines `MissionTime` with `travel_s`, `sampling_s`, `online_computation_s`, and the derived `total_s`. The current controller initializes online computation to zero and never increments it.

The final summary saves:

```json
{
  "travel_time_s": 823.8579669298437,
  "sampling_time_s": 75.0,
  "online_computation_time_s": 0.0,
  "mission_time_s": 898.8579669298437,
  "time_breakdown": {
    "travel_s": 823.8579669298437,
    "sampling_s": 75.0,
    "online_computation_s": 0.0,
    "total_s": 898.8579669298437
  }
}
```

- Travel includes actual motion and the final return.
- Sampling records actual dwell, including rejected measurements. It is not inferred solely from successful measurement count.
- Each planning log row reserves `online_computation_time_s: 0`, records consumed sampling time, and stores cumulative breakdown snapshots.
- `plan_s`, `grg_runtime_s` and other measured wall runtimes remain diagnostic. They are not added to this mission budget.
- Initial GP fitting and map preparation are outside the online mission.

## Verification

All **18 current tests** passed. Historical baseline tests passed **22/22**, and benchmark checks passed **6/6**. [Verification evidence](verification/checks.json) includes source hashes and historical-file preservation.

The added checks verify serialization and summation, zero online charges, agreement with navigation elapsed time, and consumed dwell after invalid data. A test changes reported planning wall runtime to 100,000 s per call: trajectories, measurements and mission timestamps remain unchanged because the current online charge is zero. This is a test double, not an actual delay.

The standard synthetic main run completed **15 measurements**, including two at Home, and returned at **898.857967 s**. Its three components satisfy:

`823.857967 s + 75 s + 0 s = 898.857967 s`.

All **14** existing mission acceptance checks passed. Final RMSE remained **0.018171 °C** at **12.25 h**. No depth benchmark was rerun. Data and six figures are in [main/](main).

## Future boundary

The reserved online-computation field can represent a future charge. The current controller does not implement nonzero timing, a clock provider, or a ROS/Nav2 adapter. Future integration must synchronize computation charges, actual elapsed time, sampling timestamps, replanning and return reserves. Merely populating the summary field is not a complete timing integration; avoid counting time already included in navigation feedback twice.

## Commands

```bash
python testing_code/baseline/experiment.py
python testing_code/baseline/verify.py
```

Outputs are under `10.18_result/baseline/updates/mission_time_model/`. Previous Home-sampling and revisit-only results, the original baseline, depth benchmark and historical Notebook are retained.
