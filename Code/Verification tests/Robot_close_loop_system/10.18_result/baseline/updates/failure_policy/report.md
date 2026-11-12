# Candidate failure policy

Historical policy-update record. The runtime code has since been consolidated into `testing_code/baseline/`. Source links point to current files. Use the [current report](../../current/report.md) for the active implementation and output paths.

## Rules

| Evidence or event | Record | Release |
|---|---|---|
| Confirmed collision with the static navigable map, including footprint clearance | Permanent | No automatic release within this mission. |
| Confirmed absence of a path in the static navigation map | Permanent | No automatic release within this mission. |
| Pedestrian blockage, local navigation failure, timeout | Temporary | Skip the next planning call only. |
| Generic `unreachable`, unknown failure or unavailable travel estimate without static-map proof | Temporary | Skip the next planning call only. |
| Measurement failure reported through the reserved entry | Temporary | Skip the next planning call only. |

Failure in round k sets `retry_at_plan = k + 2`. The node is blocked in k+1 and removed from the temporary record before k+2. Eligibility is restored; the planner is not forced to select the node. Repeated temporary failures never automatically become permanent. These rounds are actual GRG planning calls, not seconds or successful measurements.

## Implementation

Code: [testing_code/baseline/](../../../../testing_code/baseline).

- `failures.py`: `CandidateFailures.permanent` maps node IDs to map-evidence reasons. `temporary` maps node IDs to a reason and `retry_at_plan`.
- `navigation_backend.py`: `static_failure_reason(goal)` returns `static_collision`, `static_no_path`, or `None`. The Python backend checks its static footprint-aware map and path model. A future Nav2 adapter must use static evidence, not a pedestrian/local-planner event.
- `controller.py`: validates available static evidence before the first plan, passes the current exclusion union to GRG, classifies navigation failures, and logs reasons and expiry rounds. Generic `unreachable` is not itself permanent evidence.
- The existing invalid-measurement path calls `measurement_failed(node, planning_round)`. Sensor generation and detection are unchanged. No real failure generator or immediate measurement-retry mechanism is added.
- If a temporary cooldown produces no sampling route, another planning call is allowed while a safe return remains feasible. No waiting or online-computation charge is invented.

Permanent evidence supersedes a temporary record. Navigation blockage still prevents invalid data from entering the GP. GP, MI, route search, Home sampling, spatial revisits and the three-term mission-time model are otherwise unchanged. Online computation remains 0 s.

## Planning-round example

| Round | Candidate 1 | Controller outcome |
|---|---|---|
| k | Eligible; navigation reports a temporary failure. | Register `retry_at_plan=k+2`; do not append a measurement. |
| k+1 | Excluded. | Plan with other candidates, or defer if only cooldowns prevent sampling. |
| k+2 | Temporary entry removed; eligible again. | Reconsider normally using the current posterior and budget. |

## Verification

All **29 current tests** passed: 18 revisit/Home/timing checks and 11 failure-policy checks. Historical baseline tests passed **22/22**; benchmark checks passed **6/6**. [Verification evidence](verification/checks.json) records logs, source hashes and unchanged historical files.

Tests cover both permanent reasons, exact one-call expiry, repeated temporary failures, permanent precedence, sensor-failure registration, map-invalid candidates before planning, and static no-path evidence. Six controlled navigation statuses (`pedestrian_blocked`, `local_failure`, `timeout`, `unreachable`, `failed`, `blocked`) each produced eligibility `[eligible, excluded, eligible]` across three calls. The first failure produced no GP update; a subsequent successful visit updated the GP. These are test doubles, not implemented pedestrian or Nav2 simulations.

The invalid-measurement interface was also checked with a test-only failed measurement status. It produced the same cooldown and no permanent ban. Its consumed dwell remains in the sampling-time ledger.

## Safety boundary

The pre-existing return-after-partial-motion behavior is retained. If a failed navigation stops at a non-candidate pose, the point receives the correct failure classification, but the controller attempts safe return rather than planning from an incorrect candidate ID. Replanning from arbitrary intermediate poses requires a separate pose/travel-cost interface change and was not introduced here.

No repeated-failure threshold, permanent escalation, dynamic-map detector, nonzero online-time charge or automatic sensor recovery was added. The existing iteration cap remains unchanged.

## Smoke run and files

The ordinary seed-18 synthetic run completed **15 measurements**, including two at Home, and returned at **898.857967 s** within **900 s**. All **14** saved mission acceptance checks passed. No failures occurred in this smoke run, so its performance is not evidence of recovery behavior; controlled tests provide that evidence. Data and six figures are in [main/](main).

```bash
python testing_code/baseline/experiment.py
python testing_code/baseline/verify.py
```

Outputs are under `10.18_result/baseline/updates/failure_policy/`. Earlier revision results and the original baseline/benchmark are retained.
