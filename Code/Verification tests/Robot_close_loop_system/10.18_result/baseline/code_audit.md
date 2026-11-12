# Code audit

## Reuse

| Existing component | Decision |
|---|---|
| 10.16 `gp_model.py`: `SpaceTimeGP`, `fit_gp`, scaling, prediction, checkpoints | Import unchanged. Train on 100 historical observations only. |
| 10.16 `synthetic_field.py`: `temperature_true` | Import unchanged in the sensor and evaluation modules only. |
| 10.16 `evaluation.py`: `distribution_metrics` | Import unchanged. Evaluate latent truth with latent variance. |
| Reference `src/planners.py`: `grg_plan` | Adapt the recursive waypoint and budget-split search in a separate module. |
| Reference Notebook 05: `receding_horizon_grg_tour` | Reuse the execute-one-waypoint control pattern. Add explicit navigation results and dwell time. |
| Reference Notebook 07: temporal greedy planner | Reuse arrival-time bookkeeping, not its greedy EMSE search. |
| Reference Notebook 07: `continuous_route` | Retain path-length/speed semantics. Use a grid graph for the required footprint and corner checks. |

## Required extensions

- `FittedGP.predict` returns marginal moments. Add joint covariance queries and fixed-parameter observation appends.
- Keep GPyTorch as the fitted model. Use an exact batched RBF/Cholesky covariance evaluator to score many short routes. Check it against GPyTorch and direct conditioning.
- The old GRG signature is `grg_plan(s, t, tau_start, tau_end, tt, K_all, noise_variance, depth, n_tau_splits=5, _X_frozen=None, _mi_cache=None, objective_fn=None)`.
- Its objective accepts unordered node sets and a fixed covariance. Its recursive calls omit `objective_fn`. A callback alone cannot supply ordered arrival times correctly.
- Preserve recursive left/right route construction. Score ordered routes using posterior covariance at their arrival times. Start the right branch at the actual left-branch completion time. Budget splits are deadlines, not waiting commands.
- The old reward is full-candidate latent entropy reduction given committed measurements. The new reward is `I(y_S; f_S | D) = 0.5 log det(I + Sigma_SS|D / noise)`. For a fixed joint Gaussian field containing S, these information expressions agree. Here S changes with route timestamps; the fixed-set submodularity guarantee is not assumed.
- Navigation requires an inflated map, reachable candidates, segment collision checks, and simulated path traversal. The old notebooks do not expose replaceable navigation or sensor interfaces.
- Add explicit failure handling, mission logs, fixed-target evaluation, and automated acceptance tests.

## Units and scope

Positions use metres. GP times use hours. Navigation uses elapsed seconds. Convert with `t_h = start_h + elapsed_s / 3600`. Reference Notebook 07 uses seconds inside its temporal GP; that convention is not copied.

No 10.16 checkpoint is reused. No future history, real Dashboard data, ROS 2, Nav2, or planner comparison is included. Existing experiments remain unchanged.

## Implemented choices

- The reference revision inspected was `f3dd7a30568824a99b4248abebd15702a2694d7b`.
- `grg.py` keeps recursive left/right growth. It passes the full ordered prefix to every recursive reward calculation. Home and the current location are not counted as new measurements.
- A split time is a completion deadline. The robot does not wait until that deadline. Direct depth-one paths are identical across split choices, so duplicate evaluations are collapsed and batched exactly.
- Candidate generation uses seeded farthest-point coverage on reachable grid cells. It does not use GP uncertainty. The 48-node set includes home.
- Grid paths use Dijkstra. Every edge is checked against inflated polygons. Diagonal moves also require both adjacent axial cells to be free.
- The footprint radius is 0.360555 m; the margin increases it to 0.460555 m. Polygon buffers are enlarged slightly to compensate for inscribed circular arc segments.
- Samples use arrival coordinates and arrival time. Five seconds of dwell precede the next departure. The model uses hours; trajectory logs use elapsed seconds.
- GPyTorch's Gaussian likelihood retains its default lower bound of `1e-4` in scaled variance units. The noise parameter is learned from historical data, then frozen. Zero added synthetic noise does not imply zero model noise.
- No validation or test field is used for model selection. The historical-only fit uses a configured 300 Adam steps. Evaluation truth never affects route selection.
- The dense evaluation grid is offset from the navigation grid. Only points connected to the home component are retained. Initial and final predictions use exactly the same target.
- Planning and GP wall time are logged separately. They do not advance the simulated clock. Path traversal and sampling do advance it.
- New source and tests are under `testing_code/`. Only this experiment's README and Notebook were edited. No reference or 10.16 source was modified.
