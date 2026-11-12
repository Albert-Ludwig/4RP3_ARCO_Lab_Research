# Time-aware GRG recursive correctness verification

## 1. Verdict and scope

**Not fully verified. One implementation defect is confirmed.** With `allow_revisit=False`, depth 3 can return a route that samples a non-home node twice. This is a constraint violation, not an optimality gap.

The new suite has **7 passes and 1 failure**. Existing baseline tests pass **22/22**; benchmark tests pass **6/6**. All **416** existing baseline and benchmark files covered by the preservation snapshot are byte-identical before and after verification. Generated Python caches are excluded. No production algorithm, GP, objective, configuration, saved mission, or existing Notebook was modified. No large benchmark was rerun.

## 2. Existing verification and gaps

All tests below were rerun and passed. These are finite checks, not a proof of recursive correctness. Test names refer to [test_components.py](../../testing_code/baseline/test_components.py), [test_system.py](../../testing_code/baseline/test_system.py), and [test_benchmark.py](../../testing_code/grg_depth_benchmark/test_benchmark.py). Exact execution output is in [baseline_regression.log](baseline_regression.log) and [benchmark_regression.log](benchmark_regression.log).

### Baseline tests

| Test function | Source functions | What it verifies; remaining gap |
|---|---|---|
| `TestAEnvironment.test_map_and_footprint` | `navigation.GridMap.__init__`, `is_free` | Obstacle validity, inflation and footprint clearance. Not recursive planning. |
| `TestAEnvironment.test_candidates_reachable_repeatable` | `GridMap.candidates`, `route` | Repeatable, unique and reachable candidates. Not unique visits within a planned route. |
| `TestAEnvironment.test_edges_and_no_corner_cutting` | `GridMap.segment_free`, graph construction | Every map edge is safe; diagonals cannot cut corners. Not recursive budget allocation. |
| `TestAEnvironment.test_motion_cost_and_return` | `PythonNavigationSimulator.estimate_travel_cost`, `navigate_to` | Travel time equals path length/speed; navigation returns home. Not recursive route assembly. |
| `TestAEnvironment.test_invalid_requests` | `GridMap.node`, `route`, `navigate_to` | Invalid destinations do not move the robot. Not infinite entries inside planner travel matrices. |
| `TestBPosterior.test_joint_matches_conditioning_and_scaled_units` | `gp.GPModel.joint`, `covariance_many`, `predict` | Initial posterior agrees with analytical conditioning and GPyTorch; correct units and noise. Does not independently check the full covariance after online appends. |
| `TestBPosterior.test_append_fixed_parameters_and_variance` | `GPModel.append`, `refresh`, `parameters` | Multiple updates retain observations and parameters; variance decreases. Not planner cache behavior. |
| `TestBPosterior.test_temperature_changes_mean_not_covariance` | `GPModel.append`, `joint` | Fixed-parameter covariance does not depend on observed temperature values. Not recursive search. |
| `TestBPosterior.test_future_history_and_invalid_measurement_rejected` | `GPModel.train`, `append`, `arrays` | Rejects future history and invalid observations. Not search scheduling. |
| `TestCGRG.test_information_gain_correlations` | `grg.information_gain` | Log-determinant MI includes correlations. Not independent Cholesky scoring of recursive routes. |
| `TestCGRG.test_route_covariance_is_posterior_and_time_dependent` | `GRGPlanner.schedule`, `score_routes` | A fixed route uses posterior covariance, arrival times, preceding dwell and route order. Not every recursive level. |
| `TestCGRG.test_recursive_search_budget_and_updated_posterior` | `GRGPlanner.plan`, nested `search` | One depth-2 route respects budget; low budget terminates; new GP state changes scoring. Does not inspect child deadlines or depth-3 duplication. |
| `TestCGRG.test_visited_nodes_and_configurable_revisit` | `GRGPlanner.plan` exclusion set | Already visited nodes are excluded unless revisits are enabled. Does not forbid reusing a future ancestor endpoint inside one route. |
| `TestDClosedLoop.test_full_loop_and_repeated_run` | `mission.MissionController.run` | Repeatable execution, appends, replanning versions and return home. Selected depth-2 fixture only. |
| `TestDClosedLoop.test_actual_navigation_clock_used` | `MissionController.run`, navigation feedback | Measurements use actual delayed arrival time. Not recursive estimated scheduling at all depths. |
| `TestEEvaluation.test_fixed_query_variance_and_latent_coverage` | `evaluate.Evaluation.record` | Fixed-target variance, RMSE and coverage calculations. Not the planning objective or recursive search. |
| `TestFInterfaces.test_failed_navigation_does_not_measure` | `MissionController.run` | Failed navigation without motion does not create data. Not recursive route construction. |
| `TestFInterfaces.test_invalid_sensor_and_wrong_timestamp` | `MissionController.run`, `Measurement.valid` | Invalid data are not appended. Not recursion. |
| `TestFInterfaces.test_partial_navigation_failure_returns_from_actual_position` | `MissionController.run` | Return after partial failure starts from actual pose. Not recursive budget splits. |
| `TestFInterfaces.test_empty_plan_and_insufficient_budget` | `MissionController.run` | Empty plan or insufficient budget terminates safely at home. Not an away-from-home return boundary. |
| `TestFInterfaces.test_empty_navigation_path` | `PythonNavigationSimulator.navigate_to` | Empty path is unreachable and consumes no motion time. Not planner-level unreachable nodes. |
| `TestFInterfaces.test_core_modules_have_no_truth_or_ros_import` | GP, GRG and mission import boundaries | No direct truth or ROS imports in core modules. Not numerical correctness. |

### Benchmark tests

| Test function in `DepthBenchmarkTests` | Source / saved evidence | What it verifies; remaining gap |
|---|---|---|
| `test_original_baseline_files_are_preserved` | Git baseline tree and relocation record | Original artifacts and baseline Notebook content are preserved, allowing recorded path changes. Not algorithm correctness. |
| `test_planning_timeout_stops_worker` | `run_benchmark.supervised` | Per-call watchdog kills a delayed worker. Not search correctness. |
| `test_mission_wall_timeout_stops_worker` | `run_benchmark.supervised` | Worker wall-time limit is enforced. Not simulated time allocation. |
| `test_depth_two_reproduces_baseline` | Saved `baseline_reproduction.json` | Recorded baseline reproduction passed. This test reads evidence; it does not rerun the mission. |
| `test_only_depth_changes_and_initial_checkpoint_is_shared` | Saved configurations and checkpoints | Depth is the only changed model setting; initial checkpoints match per seed. Not recursion. |
| `test_completed_missions_and_waypoint_bound` | Saved mission verification and summaries | Completed runs meet recorded checks, waypoint bounds and zero-repeat summaries. Only those saved runs are covered. |

### Source inspection

- [grg.py](../../testing_code/baseline/grg.py), lines 47–55: `schedule` advances travel time, samples at arrival, adds dwell at non-home destinations, then includes the return leg. The initial route vertex is not sampled again.
- Lines 57–67: `score_routes` evaluates the full ordered prefix/route as one joint MI. It does not sum independent waypoint rewards.
- Lines 69–117: `plan` / nested `search` use absolute completion deadlines. `lo` includes arrival and dwell at the intermediate node; `hi` reserves travel and any dwell at the subproblem endpoint. The right branch starts from the actual left-route completion, not by waiting until its deadline.
- Lines 74–81: the reward cache is local to one synchronous `plan` call and keyed by the entire ordered route tuple. Elapsed time, GP object, dwell and travel matrix are fixed during that call. There is no recursive-subproblem memoization. A new call creates a fresh cache.
- [gp.py](../../testing_code/baseline/gp.py), `append` and `refresh`: observations are appended, the training covariance factor is refreshed, and fitted parameters remain fixed. `covariance_many` conditions on the complete current training set.
- [navigation.py](../../testing_code/baseline/navigation.py), `GridMap.route`, `travel_matrix`: travel costs come from collision-checked graph paths. GRG itself receives a cost matrix; it does not perform collision detection. The separate map-path cache assumes a static graph.

## 3. Added checks and results

Code: [test_recursive.py](../../testing_code/grg_correctness/test_recursive.py). Reference calculations: [reference.py](../../testing_code/grg_correctness/reference.py). Results: [new_tests.json](new_tests.json), [new_tests.log](new_tests.log), [evidence.json](evidence.json).

| New test function in `RecursiveChecks` | Result | Evidence |
|---|---|---|
| `test_exhaustive_small_routes_and_independent_reward` | PASS | Enumerates unique feasible routes for 3–6 sampling nodes. Checks depths 1–3, independent timing and returned-route MI. Invalid uniqueness is recorded separately, not accepted as valid. |
| `test_recursive_deadlines_prefix_and_full_joint_scores` | PASS | Traces 61 recursive returns at depth 3, starting away from home. Checks child split limits, inherited prefixes, endpoint order and deadlines. Independently checks every scored route in that case. |
| `test_no_duplicate_sampling_nodes_when_revisit_disabled` | **FAIL** | Depth 3 returns `[0, 3, 1, 2, 3, 0]` despite `allow_revisit=False`. |
| `test_cache_disabled_matches_original_across_planning_states` | PASS | Twelve cached/uncached pairs across depths, prefixes, elapsed times, budgets, visited and blocked nodes; another pair after a GP update. Route, arrivals, finish, reward and GP version agree. Forced misses increase actual scoring work. |
| `test_same_route_rescored_for_new_time_and_posterior` | PASS | The same route is rescored after elapsed-time changes and two online appends. Independent conditioning uses all 102 observations. |
| `test_exact_budget_dwell_and_return_boundary` | PASS | Exact travel+dwell+return budget is accepted; a slightly smaller budget is rejected. Checks direct return and impossible return away from home. |
| `test_unreachable_blocked_and_visited_nodes_are_excluded` | PASS | Infinite travel costs, blocked nodes and historical visited nodes are excluded at depths 1–3. |
| `test_obstacle_paths_and_return_have_independent_costs` | PASS | Expands planned edges into actual map paths, checks every segment, and recomputes length/speed+dwell+return independently. |

The failing test is not marked as an expected success. `run_checks.py` exits with status 1 while the defect remains.

### Exhaustive reference

The independent reference enumerates permutations without revisits, schedules travel and dwell, and rejects routes that cannot return before the deadline. MI uses the existing fitted GPyTorch kernel, an independent Gaussian-conditioning solve, and Cholesky log-diagonals. It does not call the production MI or batched covariance function. All posterior observations and physical variance units are included.

| Sampling nodes, excluding home | Feasible unique routes | Depth-1 optimality gap (nats) | Depth-2 gap (nats) | Depth-3 result |
|---|---:|---:|---:|---|
| 3 | 16 | 3.141861 | approximately 0 | Duplicate node; gap not applicable. |
| 4 | 65 | 5.193301 | 0.909144 | Duplicate node; gap not applicable. |
| 5 | 326 | 7.002009 | 2.717852 | Duplicate node; gap not applicable. |
| 6 | 1,955 | 8.552834 | 4.268677 | Duplicate node; gap not applicable. |

The gap is the best feasible unique-route MI minus the independently evaluated GRG MI. The reference is not restricted to GRG's lookahead size. Positive gaps are diagnostics, not implementation bugs. An invalid repeated-node route is outside this feasible set, so its gap is left null. Across all 12 returned routes, MI agrees with independent evaluation within `3e-12` nats. Accurate scoring does not make an invalid route valid.

## 4. Confirmed defect: ancestor endpoint revisited

**Affected function:** `GRGPlanner.plan` → nested `search`, [grg.py](../../testing_code/baseline/grg.py), especially lines 86, 91 and 108–111.

**Cause.** Choices exclude the committed prefix and the current endpoint. They do not exclude endpoints reserved by outer recursive calls. A deeper left branch can sample an outer endpoint before it is due. Later route construction appends that endpoint again. This becomes possible at depth 3; depth 2 does not have the same non-home ancestor-endpoint exposure. The final route has no uniqueness validation.

**Minimal reproduction using the unchanged GP checkpoint:**

- Home/node 0: `(2, 2)` m. Sampling nodes 1–3: `(7, 9)`, `(20, 24)`, `(28, 8)` m.
- Load `10.18_result/baseline/main/initial_gp.pt`; no training.
- Use the original obstacle map and navigation travel matrix at `0.35 m/s`.
- `start_h=12`, `sample_s=5`, `depth=3`, `time_splits=2`, `remaining=900`, `allow_revisit=False`.
- Returned route: **`[0, 3, 1, 2, 3, 0]`**.
- Arrival times: `81.386518`, `148.753453`, `211.995671`, `273.851371` s.
- Finish including return: `360.237889 s`. Reward: `5.726779 nats`.
- Node 3 is sampled twice at different times. Budget and paths are valid; the no-revisit constraint is not.

Run the standalone example from the experiment root:

```bash
python testing_code/grg_correctness/reproduce_duplicate.py
```

Exact output: [obstacle_map_reproduction.json](obstacle_map_reproduction.json). A second fixture using Euclidean metric travel costs reproduces the same repeated-node sequence in the failing unit test.

**Impact on existing results.** The read-only audit found **0 repeated-node routes among 171 saved benchmark planning calls**, covering all ten completed depth-2/depth-3 missions. Existing numerical results are preserved as observations from those runs. This does not prove the bug had no effect on rejected intermediate branches or search decisions. The depth-3 benchmark is not a general correctness certification. Depth-5 full missions remain unattempted; nothing here establishes their correctness or performance. The receding-horizon controller executes one waypoint, so an invalid planned route does not automatically mean the robot actually resamples that node.

**Proposed correction, not implemented.** Track reserved future ancestor endpoints through recursive calls. Exclude them while constructing a left subroute, and preserve the relevant reservations for the right subroute. Before accepting a route with revisits disabled, validate uniqueness of non-home sampling nodes. Do not deduplicate a completed route afterward: that changes timing, feasibility and joint MI. Any correction needs explicit approval and regression verification.

## 5. Limits and next-stage decision

- Budget splitting, ordered scheduling, joint posterior MI, cache consistency, and path feasibility passed the stated finite cases. They are not universally proved.
- Cache checks assume synchronous planning, an immutable GP during a call, and static travel costs. Concurrent GP updates, dynamic obstacles and navigation-cache invalidation are unverified.
- Deterministic tests cover depths 1–3. No large depth-5 execution was performed.
- Static shortest-path cost matrices, valid indices, positive noise and nonnegative timing are input assumptions. Arbitrary malformed matrices are not exhaustively validated.
- No global optimality claim, planner comparison, alternative reward, ROS timing validation or new field-reconstruction benchmark is included.

**Decision:** retain the unchanged depth-2 baseline as a limited reference. Do not sign off the general recursive implementation or deeper-depth integration as fully correct. Resolve the confirmed route-construction defect before treating depth 3+ as a verified no-revisit planner. Preserve old results, and label their finite-case scope.

## 6. Files and rerun

```bash
python testing_code/grg_correctness/run_checks.py
```

- `new_tests.json`, `new_tests.log`: all eight added tests, including the real failure.
- `evidence.json`: exhaustive routes, independent scores, recursive trace, cache pairs and defect fixture.
- `obstacle_map_reproduction.json`: actual-map minimal reproduction.
- `baseline_regression.log`, `benchmark_regression.log`, `regressions.json`: 22+6 existing checks rerun after adding isolated tests.
- `saved_route_audit.json`: all 171 saved planning routes and duplicate checks.
- `preservation.json`: before/after hashes of 416 existing files.

No existing baseline or benchmark source was edited. Only isolated verification code, new verification results, and the overall README index were added or updated.
