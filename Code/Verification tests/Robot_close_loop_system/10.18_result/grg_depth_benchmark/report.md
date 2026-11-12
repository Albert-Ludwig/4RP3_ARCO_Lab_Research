# GRG recursive depth benchmark

## Objective and controls

Test whether deeper GRG lookahead improves sampling and reconstruction enough to justify its planning latency.
Only `grg.depth` changes. All runs for a seed load byte-identical copies of one initial GP checkpoint. The field, historical data, candidates, navigation, MI reward, target grid, and 900 s simulated budget are fixed.

## Depth meaning

Depth counts recursive left/right route-growth levels. At depth one, direct split duplicates are batched. The upper bound on sampling visits is N(d)=2 N(d-1)+1, N(0)=0: 3, 7, and 31 for depths 2, 3, and 5. This is an upper bound, not a promise of distinct or feasible stops. Budget and exclusions may reduce it. Depth 5 is accepted by the existing code; its computational feasibility is measured below.

## Procedure

1. Prepare one initial state for each seed. Existing checkpoints are loaded for seeds 18–20; seeds 21–22 are each trained once.
2. Reproduce the seed-18 depth-2 baseline before comparisons.
3. Screen one initial planning call at depths 2, 3, and 5 with a 90 s planning limit.
4. Run independent missions for every screened-feasible depth across seeds 18–22. A worker is stopped after 90 s in a planning call or 900 s of total worker wall time.
5. Compute realized MI from all executed samples conditioned on the original historical GP. Here realized refers to executed positions and timestamps; fixed-parameter covariance MI does not depend on their measured temperature values. Do not sum revised route rewards.
6. Aggregate completed runs and paired differences. Keep timeouts and unattempted missions separate.

## Baseline reproduction

Seed 18, depth 2 reproduced all 14 checks, including identical routes, measurements, trajectories, and metrics. Wall times are excluded.

## Initial planning screening

| Depth | Status | Planning time (s) | Sampling visits | Planned MI (nats) |
|---|---|---:|---:|---:|
| 2 | SUCCESS | 0.081 | 3 | 9.0979 |
| 3 | SUCCESS | 20.847 | 7 | 18.9803 |
| 5 | TIMEOUT | ≥ 90.07 (censored) | — | — |

## Completed-mission comparison

Values are mean ± sample standard deviation across successful seeds. No statistical significance is claimed.

| Depth | Completed | Final RMSE (°C) | Final mean variance (°C²) | Realized MI (nats) | Mean planning time (s) |
|---|---:|---:|---:|---:|---:|
| 2 | 5/5 | 0.0259878 ± 0.00861 | 0.00132257 ± 0.000535 | 28.5052 ± 1.71 | 0.0516795 ± 0.0019 |
| 3 | 5/5 | 0.0218551 ± 0.00276 | 0.000797148 ± 0.000264 | 35.2453 ± 2.75 | 7.77459 ± 0.375 |
| 5 | 0/5 | — | — | — | — |

## Paired results and answers

- Depth 3 versus depth 2 (5 pairs): mean final RMSE difference -0.004133°C; mean final variance difference -0.000525°C²; mean realized MI difference +6.7401 nats. Mean planning latency ratio: 150.4×; total planning ratio: 207.5×. Executed sample sequences changed in 5/5 pairs. Final RMSE improved in 4/5 pairs and worsened in 1/5 pairs.
- Depth 5: no complete paired missions. Reconstruction, realized MI, route changes, and mission runtime cannot be compared from the screening result alone.
- Depth(s) [5] exceeded or failed initial screening. Full missions were not attempted at those depths. They were not silently reduced or counted as successful runs.
- Additional lookahead does not guarantee better realized information or RMSE under receding-horizon execution.
- Depth 2 completes the validated mission with low latency. Retain it as the provisional ROS/Nav2 default unless the paired gains justify a larger planning delay. This is an engineering recommendation for this configuration, not proof of optimality.

## Limits and interpretation

The reward remains Sampling-Point MI, not target-field variance reduction. Five seeds are a sensitivity check, not a significance study. Compute time is excluded from simulated mission time, as in the baseline. Simulated return within 900 s does not establish a 900 s ROS wall-clock mission. Latency must be reconsidered for ROS execution. Timeout duration is right-censored: it is a lower bound, not a measured completion time. Unattempted full missions are not mission timeouts and have no final metrics. Peak memory for completed planning calls is the worker's lifetime peak RSS, not incremental GRG allocation. Planning statistics include the initial and terminal planning calls. Replanning calls equal total planning calls minus the initial call. Runs execute sequentially on one host; wall times are hardware- and load-dependent. The mean of per-seed percentage improvements differs from the percentage change between aggregate means; both use their stated denominators. The initial depth-2 export had a JSON boolean serialization issue in the benchmark wrapper; it was fixed before comparison. Its diagnostic files are excluded from all tables.

## Run and files

```bash
python testing_code/grg_depth_benchmark/run_benchmark.py
python testing_code/grg_depth_benchmark/report.py
```

- `shared/`: one immutable initial state per seed.
- `screening/`: initial-call plans, events, runtimes, and timeout records.
- `runs/`: each mission's configuration, checkpoints, trajectory, measurements, plans, metrics, and verification.
- `runs.csv`, `aggregate.csv`, `paired_differences.csv`, `paired_summary.csv`: comparisons.
- `planning_calls.csv`: completed calls and censored timeouts; planned rewards are not summed.
- `figures/`: seven comparison plots.
- `verification/`: baseline reproduction, original regression tests, benchmark tests, and source checks.

## Research-question checklist

| Question | Evidence / interpretation |
|---|---|
| 1. Higher realized information? | Depth 3 − 2: +6.7401 nats on average over 5 complete pairs. |
| 2. Better final RMSE? | Depth 3 − 2: -0.004133°C. Negative differences favour depth 3. |
| 3. Lower final variance? | Depth 3 − 2: -0.000525°C². Negative differences favour depth 3. |
| 4. Different routes? | Exact per-seed sample-sequence comparisons are in paired_differences.csv; see the paired results above. |
| 5. Depth-3 computation cost? | Mean paired latency ratio 150.4×; mean paired total-planning ratio 207.5×. |
| 6. Depth-5 cost? | Use its censored initial-call duration in the screening table. There is no measured completion or full-mission cost when screening times out. |
| 7. Meaningful extra benefit? | Mean paired final-RMSE improvement is 7.8% at 207.5× total planning cost. The value of that tradeoff requires an agreed latency/accuracy criterion; no significance is claimed. |
| 8. Is depth 2 sufficient? | It satisfies the tested closed-loop correctness and mission goals. This does not establish global route optimality. |
| 9. Best tradeoff? | No unique optimum is defined without a latency/accuracy preference. Depth 2 is the low-latency baseline; depth 3 is the higher-cost option. Depth 5 is unavailable after its failed screen. |
| 10. ROS/Nav2 default? | Provisionally retain depth 2. The current experiment excludes computation from simulated time; ROS timing requires a separate validation. |
