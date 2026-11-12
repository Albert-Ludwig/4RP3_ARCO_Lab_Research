# Updated GRG correctness verification

## Policies and implementation

Allow all intentional repeated spatial events, including A-B-A and A-A. The start vertex denotes robot state; intermediate vertices denote measurements; the final Home vertex denotes return only. Every intentional event pays dwell and keeps its own time and joint-covariance entry. Pass the completed left prefix directly to the right recursive branch, so a shared junction is represented once. A new explicit insertion at the same node is a separate event.

Remove residual filters that excluded the current non-Home location or a matching recursive sampling endpoint. Retain the existing recursive search, budget splits, full-route MI, kernel and GP update semantics. Chunking score batches bounds temporary covariance memory without changing their values.

Temporary failure during cycle k excludes the node in k+1 and restores eligibility in k+2. Only top-level cycles count. Static collision/no-path evidence permits permanent exclusion. Invalid measurements do not update the GP; the existing one-cycle measurement-failure hook remains. If no feasible alternative exists during exclusion, stop and return safely. Remove the earlier empty-cycle deferral. Recovery tests now execute a genuine alternative measurement between failure and re-eligibility.

The existing partial-motion fallback returns from the actual pose and charges consumed travel. Replanning from arbitrary non-candidate poses remains unimplemented. The original 200-iteration guard remains a mission safety bound, not a new navigation retry limit. Home sampling stays enabled because the user explicitly authorized it before this task.

## Results

Eight updated categories: exhaustive bounded repeated-event routes; recursive deadlines/prefixes/junctions; consecutive and nonconsecutive revisits; cache consistency; time-aware posterior MI; dwell/budget/return boundaries; candidate eligibility/failures; obstacle paths and prior depth-3 reproduction.

- Updated recursive checks: 8 run, 0 failures, 0 errors.
- Current baseline regressions: see `baseline_regression.log` (52 tests).
- Historical benchmark artifact/watchdog regressions: see `benchmark_regression.log` (6 tests).
- Recorded-data checks: 5 tests; passed = True.
- Maximum production versus independent joint-MI difference: 1.42e-11 nats.
- Recursive subproblems checked: 269.
- Cache-enabled versus disabled states checked: 18.
- Bounded exhaustive configurations: 12.

The GP independent reference uses its fitted kernel and a separate conditioning solve and Cholesky log determinant. Route references independently sum travel, event dwell and final return. Tests compare event times, finish times and MI. The exhaustive event bound is three with an 18 s budget and 5 s dwell; this bounds enumeration, not production revisits. Optimality gaps are diagnostics only.

Existing navigation, posterior, invalid-measurement and return tests are retained. Update old assumptions requiring [0,A,0] when consecutive A samples are now valid; verify feasibility and event costs instead. Update cooldown recovery fixtures to execute a real alternative action. Add a strict no-alternative termination check. The obsolete spatial-uniqueness test is replaced by event-consistency checks, while numerical and obstacle checks remain.

## Prior depth-3 case

Analyze [0,3,1,2,3,0] directly, independently of whether today's planner selects it. It contains four intended measurements, including two at node 3, each with its own time and 5 s dwell. Joint covariance includes all four. Independent timing, MI, obstacle paths and the 900 s budget pass. The old spatial-uniqueness failure is obsolete under the authorized policy. A duplicated recursive junction would still be a defect; prefix and deadline tracing checks this distinction. See `obstacle_map_reproduction.json`.

## Limits

These are bounded simulation tests, not proof of global optimality or verification for arbitrary depth. No depth-5 benchmark is executed. ROS/Nav2, dynamic obstacle recovery from arbitrary poses, real sensor timing adapters, and nonzero online computation charges remain future work. Recorded replay adds explicitly authorized acquisition waits; the synthetic timing convention is unchanged.
