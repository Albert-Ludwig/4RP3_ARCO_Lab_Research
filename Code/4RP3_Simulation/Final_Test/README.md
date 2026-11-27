# 10. Experiment Plan — Continuous Edge Sensing and Real-Data-Informed Synthetic Fields

### 10.1 Scope, research question, and evidence status

**Status: the all-data RBF kernel has been exported. The formal comparison with this kernel is paused pending review.** The final comparison consists of three ROS 2 / Gazebo scenarios using one synthetic temperature-field realization. Each scenario runs MI, EMSE, and Adaptive WIVR once. All mission budgets run from departure to actual Home arrival, and sensing continues at 1 Hz on the return.

Adaptive WIVR and return-inclusive continuous sensing were exercised in the earlier 30 × 20 m / 300 s verification. The earlier comparison using the 5,087-observation kernel was stopped during Scenario B and its output was cleared. The three scenarios have yet to run with the all-data kernel. Future temperature forecasting is outside the current research scope.

**Research question** Which scoring objective gives the lowest synthetic reconstruction RMSE within each scenario, and how does performance change when the mission budget is shortened or the operating area is expanded?

| Scenario | Sensing region | Mission budget including return | Comparison |
| --- | --- | --- | --- |
| A — Reference | 30 × 20 m | 300 s | Reference for the three policies. |
| B — Short mission | 30 × 20 m | 150 s | Keep A's map, candidates, Home and GP prior. Change the mission budget and scoring time window. |
| C — Larger region | 60 × 40 m | 300 s | Keep A's mission budget and expand the map, waypoint layout and evaluation domain. |

Run **nine ROS missions** in total. Each mission starts with zero temperature observations and the same fixed GP prior. A and B use an identical navigation setup. C uses the same coordinate origin, Home, map resolution and navigation parameters. Use 10 checkpoints plus a separate Home across all scenarios. A and B reuse the verified map, checkpoints and directed path graph. C scales only the checkpoint positions with the physical region and builds a new directed graph. Share the exact candidate set and feasible Nav2 path graph among the three policies within each scenario. With a fixed candidate count and four times the area, C measures expansion with a fixed waypoint count and zero initial observations. Report the resulting coverage change alongside RMSE.

| Evidence category | Existing location / scope | How it may be used |
| --- | --- | --- |
| Historical Python experiments | Original 10.18 / 11.2 experiments, including the [archived GP–GRG replay](<../../Verification tests/Robot_close_loop_system/README.md>) | Preserve code, saved models, logs, and established results; reuse verified components where applicable. |
| Completed node-based Nav2 prototype | [saved Nav2 prototype](<../../Verification tests/GRG grading policy compare verification/result/synthetic_ros_full_01/>) and its saved reports | Evidence for the previous synthetic-record, checkpoint-sampling workflow. It does not validate continuous edge sensing or a GSTools SRF. |
| Return-inclusive verification | [Saved 30 × 20 m / 300 s runs](<../../Verification tests/README.md#return-inclusive-ros-closed-loop-verification>) | The three policies reached Home on time and retained return samples on the verification fixture. |
| Planned three-scenario comparison | Scenarios A–C above | Results and readiness checks for these settings are pending. |
| Physical WELL-E experiment | Future deployment and validation | Pending; neither synthetic navigation results nor ideal-localization tests establish hardware performance. |

Do not overwrite the original 10.18 or 11.2 experiments, the node-based Nav2 prototype, or any completed result. Use a separate implementation/configuration and result namespace for the revised protocol; keep the final experiment in this directory. Retain one experiment README for purpose, objects and dependencies, and one `Comparison.ipynb` for saved results, figures and conclusions.

### 10.2 Real datasets, Initial GP, and common reference frames

- The three simulation scenarios use the declared 30 × 20 m and 60 × 40 m working regions. A separate registered-site evaluation remains pending. Its estimated **97 × 64 m** enclosing extent and traversable mask still need confirmation.
- Use the fitted RBF length scales, signal variance, and measurement noise variance from the real Lawnmower data export. Start every mission with **zero observations** and a **fixed prior mean of 25 °C**. The fitted signal variance is the pointwise prior variance at cold start. The formal setup creates empty observation arrays and starts SRF queries at mission time zero. The GP first conditions on a valid measurement collected after mission start.
- Fit the formal RBF kernel using all **6,448 aligned temperature observations** from the six traversals on **3 November and 17–21 November 2026**, including the 1,361 observations on 21 November. The previously configured synthetic field is shared across policies. Freeze the GP prior and simulation configuration before comparing policies.
- All six real traversals are used for the formal kernel fit. Independent real-data validation is a separate study and requires unseen data or a separately fitted held-out model. Formal policy scores use the synthetic ground truth.
- Preserve measured coordinates, dates, timestamps, source identities, units and synchronization assumptions. Register runs to a common metric map frame; a GNSS trajectory is not an occupancy map.
- Define the ROS mission origin and its explicit mapping to the real-data and SRF time references. Do not compress dates or stretch recorded intervals to create apparent temporal coverage.
- Any independent real-data reconstruction evaluation remains **Pending**. It requires a predefined interpolation-valid holdout with suitable spatial and temporal support.

### 10.3 Checkpoints, feasible paths, and continuous route events

1. Use a 2D map and work-area mask for each declared simulation region, with A and B sharing the same files. Align the smaller region within C in the same metric coordinate frame. The real-site registration belongs to a separate evaluation.
2. A and B reuse the closed-loop verification fixture's saved checkpoint coordinates and directed Nav2 graph. The 10 checkpoints use $x=\{3,9,15,21,27\}$ m and $y=\{4,16\}$ m in field coordinates.
3. C doubles those checkpoint coordinates to preserve the layout relative to the 60 × 40 m rectangle. Home remains at (0, 0) and separate from the 10 sensing checkpoints. Build and validate the new directed Nav2 graph for C before scoring routes. Share it across policies.
4. GRG continues to output an ordered Checkpoint Route. A Checkpoint is a route-optimization and segment-boundary reference, not a mandatory stop-and-measure site.
5. Retain Nav2 path geometry and estimate motion timing/progress along each edge. Convert the entire candidate route into expected `(x,y,t)` sampling events using a common **initial 1 s sampling interval** and global sampling phase.
6. Score the joint event collection along the regular sampling route and reserve time for the return leg. The executed return produces valid GP observations. Avoid duplicate events at edge junctions and account for correlation between 1 Hz observations.
7. Recompute affected event predictions when navigation geometry, timing, current pose or the task schedule changes. A pairwise travel-time matrix alone cannot determine where measurements will be taken along a detour.
8. Keep the fixed checkpoint layout, **5 m / 2 s online EMSE and WIVR scoring grid**, **2 m / 5 s offline RMSE grid**, and **0.1 m / 0.5 s SRF visualization references** separate. Scoring representatives are cell centres.

Sampling during navigation, including return motion, is the proposed default. Rules for samples produced during planning pauses, waiting, recovery and target switching must be frozen and tested. Segment-based Nav2 goal handling may still slow or stop the robot; continuous sensing does not by itself establish uninterrupted motion.

### 10.4 Independent GSTools SRF temperature source

Construct an on-demand simulated temperature field

$$
T_{\mathrm{sim}}(x,y,t;\eta),
$$

where $\eta$ identifies **one fixed realization shared by all nine missions**. Define the master SRF over C's 60 × 40 m operating region and through mission time 300 s. Include the navigable clearance in the spatial query bounds. The simulated sensor queries the field from mission time zero onward. With a shared spatial origin and departure at $t=0$, A and B query the 30 × 20 m subregion of C. B uses the interval $[0,150]$ s. A and C use $[0,300]$ s. Use physical coordinates throughout, so matching $(x,y,t)$ queries return the same noise-free temperature in every scenario.

An initial candidate is a **Matérn covariance model** for a spatially/temporally scaled SRF, with an empirical mean or deterministic trend informed by historical Lawnmower data. The exact spatiotemporal construction, spatial and temporal scales, smoothness, variance, mean/trend and units remain **Pending**. Time in seconds must not be silently treated as another spatial coordinate in metres. Independent 2D redraws at successive timestamps are not a validated temporal model.

Calibrate plausible temperature range, spatial variation scales and temporal behavior from historical data; characterize the resulting distribution rather than assuming every random realization remains inside the historical minimum and maximum. The generator remains separate from the planning GP: the proposed Matérn SRF is the simulation source, while the planner retains the RBF GP. Do not use the GP Posterior Mean as Ground Truth or claim the generated field is the actual measured field.

Freeze one field seed and the complete generator configuration before evaluation, including the model, scales, mean, variance, trend and number of modes. Keep the field in physical units as the queried region changes. Use a separate fixed sensor-noise seed and stable measurement IDs, and reset each mission to the same initial state. Across scenarios, the field and noise process stay fixed while routes, actual query positions and times can differ. The planner receives only the noisy acquired measurements. Keep unmeasured truth offline and select the field independently of policy results.

**On-demand and memory requirements**

- Evaluate the field only at requested sensor or offline evaluation inputs. Accept actual continuous coordinates and timestamps; do not require them to lie on a dense lattice.
- Do not precompute or retain a complete 3D space–time temperature array. Retain the reproducible generator state, required measurements, configurations, GP snapshots and necessary statistics.
- Treat **0.1 m spatial resolution** and **0.5 s temporal resolution** as initial visualization, discrete-evaluation or output references, not limits on SRF query coordinates or the 1 s sensor schedule.
- Generate large evaluation inputs in fixed indexed batches and accumulate error statistics without materializing the whole temperature volume. Batch size, peak memory, runtime and the generator's storage behavior require measurement.
- Test repeated queries, reversed/shuffled query orders, different batch partitions, and a fresh generator reconstructed from the same saved configuration. Each must reproduce the same noise-free field at identical inputs within a documented numerical tolerance. Do not reinitialize a different realization at each request.
- If sensor noise is enabled, use a separate reproducible noise mechanism with its own seed and stable measurement identities. Evaluation queries and batching must not consume or alter the sensor-noise stream. Define whether repeated physical measurements receive independent noise; keep noise-free field truth separate from noisy observations.

### 10.5 Continuous-sensing ROS 2 / Gazebo mission

Establish an integrated **MI baseline first**, using the Gazebo Bunker Mini and Nav2, before substituting EMSE and Adaptive WIVR.

1. Initialize an empty GP with a 25 °C prior mean and the shared fitted length scales, signal variance, and measurement noise variance. Load the shared SRF and restore the scenario map, candidate Checkpoints, Home and mission timing setup.
2. Obtain feasible Nav2 paths and estimate each leg with the calibrated $d_{ij}$ in [§5.1](../../../Notes/WELL-E项目核心逻辑.md#51-路线边代价与预算). Predict the normal GRG sampling events and reserve enough budget to navigate back to Home.
3. Execute the next Checkpoint-to-Checkpoint segment through Nav2. Let the simulated body move through the physics/controller stack; do not substitute direct pose changes for execution.
4. While navigating, initially sample every 1 s using the **actual simulated pose and timestamp** to query the fixed SRF. Record the noise-free source value and noisy observation separately when applicable, with source/measurement identities and quality flags.
5. Buffer valid observations and apply the verified **1 Hz GP posterior update** as readings arrive. Keep GRG in the background and switch goals at waypoint arrival. Adaptive WIVR refreshes weights on the 5 m grid from the latest available GP snapshot and freezes the weights for each search. MI and EMSE skip adaptive weighting.
6. Handle navigation failures independently from measurement validity: valid samples already acquired on a partially executed edge remain real observations of the simulated field. Specify and test their batch-update handling; do not fabricate samples for missing positions or times.
7. When no further sampling waypoint fits with its onward return, send Home as the Nav2 goal. Continue sampling at 1 Hz along the executed return path. Include those observations in the final posterior and record actual Home arrival time, return duration, return sample count, final pose and termination reason.

The robot senses continuously along each navigation segment, including return, without a fixed per-checkpoint dwell. Validation of the three prepared simulation scenarios and the physical sensor response remains pending.

### 10.6 Mission budget and stopping rule

Use the scenario budget $T_{\mathrm{mission}}$, measured from departure to actual Home arrival. Set it to **300 s** for A and C and **150 s** for B.

$$
\tau_{\mathrm{motion}}
+\tau_{\mathrm{planning}}
+\tau_{\mathrm{other}}
\leq T_{\mathrm{mission}}.
$$

Motion includes navigation and return, with temperature sensing performed concurrently. Planning includes online GP updates, Adaptive WIVR weight calculation where applicable, and GRG work that actually occupies mission time. Other includes necessary non-overlapping waiting, switching or recovery overhead. Define a non-overlapping time ledger and use actual elapsed ROS simulation time as the authoritative budget; do not add a separate sensing duration to motion, or count overlapping computation twice. Offline hyperparameter fitting and final evaluation are outside the mission clock.

Before dispatching another sampling goal, check actual elapsed time plus the fitted cost of the next leg and the fitted cost from that waypoint to Home. Use the same $d_{ij}=L_{ij}/v+\delta$ for all three policies, with $v=0.891822\,\mathrm{m/s}$ and $\delta=2.709215\,\mathrm{s}$. Recheck after delays or path changes. When no next leg preserves return feasibility, Nav2 navigates to Home. The return samples contribute to the final GP; the information objectives continue to score the normal candidate routes.

Record any overrun, failed return or watchdog separately. The fitted travel times are estimates, and actual ROS elapsed time determines whether the mission met the deadline.

**150 s configuration.** The [shared GRG core](../WIVR/src/core.hpp#L137) now uses the scenario's `navigation_budget_s` for the mission cap, the EMSE/WIVR temporal scoring grid and the predicted route-event limit for all three objectives. At the configured 2 s scoring step, B uses **0, 2, …, 150 s**, and A/C use **0, 2, …, 300 s**. MI scores the predicted route events. Runtime checks of the 150 s scoring endpoint, event bounds and return remain pending.

**GRG search timeout.** All formal scenarios and policies use a 60 s wall-clock limit per initial or background search (`search_guard_s` in [settings.json](settings.json)). Timed-out results are ineligible for route adoption. The return-inclusive mission budgets remain 300 simulation seconds in A/C and 150 in B.

### 10.7 Policies and fair comparison

| Method | Objective / role |
| --- | --- |
| GRG + MI | Joint conditional information gain from all predicted along-route events, accounting for correlation and redundancy. |
| GRG + EMSE | Average posterior-variance reduction over the common fixed spatiotemporal scoring domain, with effective-area correction. |
| GRG + Adaptive WIVR | Posterior variance reduction with dynamically updated uncertainty-based spatial weights over the same domain. |

Freeze and share the Initial GP, RBF hyperparameters, candidate set, start/Home, navigation map, Gazebo model, Nav2/localization configuration, SRF realization/seed/trend, noise protocol, continuous sensing interval/phase, batch-update and replanning rules, mission time mapping, the fitted travel-time parameters, the scenario budget including return, 1 Hz return sensing, GRG settings and reconstruction evaluation protocol. Only the scoring objective and its necessary weight computation differ; account for their actual online costs.

**Defined Adaptive WIVR configuration**

- Use the verified **5 × 5 m** cells with area-weighted centres in each rectangular working region. A and B each have **24** spatial cells. C has **96**. At 2 s time steps, the total scoring queries are **3,624**, **1,824** and **14,496** for A, B and C, respectively.
- Use a fixed scoring set $Q=Q_s\times\mathcal T_Q$, with the same spatial representatives, mask, evaluation times and effective-area treatment for EMSE and WIVR. Predeclare the common equally spaced temporal scoring grid; no separate temporal importance function is added.
- At planning iteration $k$, query the current 3D GP at $(\mathbf p_i,t_k)$ and compute $w_i^{(k)}=a_i\sigma_{\mathcal D_k}^{2}(\mathbf p_i,t_k)$. Reuse this spatial weight at every scoring time for that cell. No regional variance-averaging algorithm is introduced.
- At cold start, the stationary RBF prior has constant variance, so Adaptive WIVR uses the same area weights as EMSE. After the first observation, compute the weights from the updated GP. Freeze weights across all routes, hypothetical observations, gain calculations and recursive subproblems in one GRG search. Model-based conditioning on a proposed route must not refresh weights within that search.
- Use the normalized WIVR objective from [§4.4](../../../Notes/WELL-E项目核心逻辑.md#44-wivr带空间重要性的方差下降量) without a second weight normalization. If all current variances are near zero, fall back to effective-area weights $a_i$; predeclare and validate the numerical threshold.
- For the Uniform Spatial Weighting check, set per-unit-area importance constant, so weights are proportional to $a_i$. This must match area-averaged EMSE numerically, including boundary cells; with equal-area cells it reduces to the ordinary arithmetic-mean identity.
- Weights use only the online GP posterior. Freeze the adaptive rule before evaluation, not the weight values for the whole mission. The earlier pool-distance rule and upper-region 3:1 fixture remain historical examples.

Use the same continuous-event construction at every GRG recursion level. Cache keys/invalidation must distinguish objective, GP version, event coordinates/times, grid/mask/area configuration, frozen weight version and affected path/timing state. Do not allow an Adaptive WIVR recursive call or shared cache to fall back to, or mix with, default MI. Checkpoints alone are insufficient cache identities.

Pair the three policies within A, B and C on the single predeclared field realization. Reset the simulator, initial pose and time, GP data, sampler phase, noise state, buffer and initial adaptive-weight state before every mission. Report policy differences within scenarios. Interpret cross-scenario differences on this shared field. Replication across independent fields remains future work.

### 10.8 Synthetic reconstruction and separate real-data validation

Define a common fixed spatiotemporal evaluation grid over the predeclared work-area domain and mission interval, independent of policy trajectories. Freeze its map mask, time origin/range, spatial/temporal resolution and batching protocol before evaluation. Do not restrict it to the convex hull of measurements acquired by a particular policy.

Use noise-free SRF values as Synthetic Ground Truth:

$$
\operatorname{RMSE}_{\mathrm{syn}}
=\sqrt{\frac{1}{N_{\mathrm{eval}}}
\sum_{j=1}^{N_{\mathrm{eval}}}
\left[\mu_{\mathrm{final}}(q_j)-T_{\mathrm{sim}}(q_j;\eta)\right]^2}.
$$

Compute metrics in batches on a **2 m spatial and 5 s temporal** offline grid with a common grid origin. Freeze the masks and query sets before running policies. Use each policy's final GP conditioned on all valid measurements, including return samples.

| RMSE report | Evaluation domain | Interpretation |
| --- | --- | --- |
| Within each scenario | A/C use their full work region × 0–300 s. B uses its full work region × 0–150 s. | Compare MI, EMSE, and Adaptive WIVR on the scenario's own task. Use the full declared window even when Home arrival is earlier, and flag queries beyond the last valid measurement. |
| A vs. B on common time support | Shared 30 × 20 m region × 0–150 s. | Compare final posteriors on identical targets. A can condition on data collected after 150 s, so this is a retrospective comparison. To compare performance at the same elapsed time, use each run's GP state at 150 s instead. |
| A vs. C on common spatial support | Shared 30 × 20 m valid region × 0–300 s. | Compare performance in the original operating region on identical queries. Report C's full-region RMSE and the RMSE in its added area separately to expose the effect of expanding coverage. |

For each scenario, compare the final RMSE of MI, EMSE and Adaptive WIVR on identical queries. Use the common-support scores for direct comparisons across scenarios. Keep SRF truth and offline RMSE out of the online planner.

Additional outputs: MAE, posterior variance, error versus mission time and successful measurement count, spatial coverage versus temporal revisits, actual Nav2 trajectories, estimated/actual travel time, GP-update/GRG latency, peak memory, actual sampling interval, mission-time ledger, failure/recovery events and return success. Reconstruct curves from saved GP snapshots and logs; evaluation truth never feeds online choices.

Maintain separate result categories for **Synthetic Reconstruction RMSE**, **independent real-data reconstruction/interpolation validation**, and **future physical-robot experiments**. The all-data fit has no independent real-data score. A separate validation study requires suitable spatial and temporal support.

### 10.9 Staged validation, implementation trace, and reports

1. **Review/reuse:** identify existing GP, GRG, Nav2, logging and numerical components. Preserve historical records and document which old assumptions are superseded.
2. **Component checks:** test Checkpoint filtering, path-to-event timing, global sampling phase/junction deduplication, correlated joint scoring, batch updates, recursive objective propagation, cache invalidation and non-overlapping budget accounting.
3. **SRF checks:** fit/check historical-data-informed parameters; test fixed-realization consistency, query order and batching, noise separation, on-demand latency and memory. Freeze the accepted configuration before comparisons.
4. **MI integration:** verify that the calibrated cost governs the return-aware budget, that the sampler stays active during Nav2 return, and that the final GP retains the return observations. Record actual Home arrival against the scenario deadline.
5. **Policy comparison:** change only the scoring objective and its necessary Adaptive WIVR weight calculation; run all three policies under the same conditions within each scenario.
6. **Reporting:** produce tables and figures from saved outputs; report independent real-data reconstruction/interpolation only when the held-out dataset and spatial/temporal support protocol are available.

**Adaptive WIVR-specific validation plan — all checks below are Pending, not passed results:**

1. Verify the 5 m scoring grid, its shared origin, representative points, cell counts and scenario time endpoints.
2. Check current GP posterior variance at every scoring representative, including units and numerical nonnegativity.
3. Check that higher variance produces higher **per-unit-area** importance; distinguish this from total cell weight when cell areas differ.
4. Check weight refresh after each actual Batch GP Update and the near-zero-variance area-only fallback.
5. Check that weights remain identical for all routes and recursive subproblems within one GRG search.
6. Check numerical equality of Uniform Spatial Weighting and EMSE using identical areas, query domain and scoring times.
7. Check that WIVR scores predicted **continuous along-path sampling events**, not Checkpoints alone.
8. Check effective areas and representative-point validity for incomplete boundary cells.
9. Measure the 5 m grid's weight and scoring cost on each scenario.
10. Check recursive objective propagation and cache isolation so WIVR never mixes with default MI.

Document each major calculation with its actual repository-relative file/function, input/output, equation, units, caller and downstream interface. Specifically trace Initial GP → current variance/adaptive weights (WIVR only) → path/timing → continuous events → joint score → GRG route → Nav2 execution → SRF observation → batch GP update → weight refresh/replanning → return. Use Pending implementation entries instead of inventing functions or claiming an interface already works.

Save full configuration/version/seed records, registered coordinate/time conventions, selected Checkpoints, predicted paths/events and actual trajectories, every valid/rejected observation, batch membership, GP snapshots, online computation and memory measurements, navigation outcomes and exact termination reasons. Also save the scoring-grid mask/representatives/effective areas, per-iteration variances and weight vectors, frozen weight versions, fallback events and weight-calculation latency. Record enough generator state to reproduce offline truth without storing the full 3D field.

Keep one README for the experiment purpose, test objects and dependencies, and one `Comparison.ipynb` for the actual results and concise conclusions. The notebook should include the three-policy RMSE summary, RMSE versus mission time/sample count, variance, actual trajectories, sampling coverage/revisits, estimated versus actual travel time, GP-update/GRG cost and return/budget outcomes. New experiment tables remain **TBD** until the revised runs are completed; do not relabel previous node-based results as continuous-SRF results.

### 10.10 Pending decisions and completion criteria

| Item | Status / required evidence |
| --- | --- |
| Real sensor response and proposed 1 Hz measurement | **Pending:** response/delay, synchronization, valid sample rate and noise characterization. |
| Checkpoints and GRG cost | **Configured:** 10 sensing checkpoints plus Home. Reuse the verified A/B path graph. C uses the scaled layout, with graph reachability and search cost pending runtime validation. |
| SRF calibration from real data | **Pending:** Matérn/spatiotemporal form, units, range/variance, spatial/temporal scales, empirical trend and parameter-fit checks using historical data only. |
| On-demand SRF consistency and performance | **Pending:** same realization under repeat/order/batch/restart tests, reproducible noise, query latency and peak memory. |
| Adaptive WIVR | **Pending for A–C:** verify the 5 m scoring grid and weights on each scenario. The earlier small-arena pilot exercised the adaptive weighting rule. |
| Scoring cost across regions | **Pending:** time the common 5 m scoring configuration as the number of scoring cells increases in C. |
| Three-scenario ROS 2 protocol | **Pending:** validate the expanded map and shared path fixtures, exercise the configured 150 s scoring window for B, and verify budget-aware dispatch, on-time Home arrival, continued 1 Hz return sensing, and assimilation of return observations in A–C. |
| Physical-robot validation | **Pending:** real sensor/navigation interfaces, localization, environment sensing and actual WELL-E trials. |
| Real-data fit and registration | **Fit complete:** six traversals and 6,448 observations. **Pending:** registered field extent and navigability, plus a separate independent real-data validation protocol. |
| References, paper notation and LaTeX | **Pending:** bibliography completion, formal manuscript symbol harmonization and publication formatting; this note revision is not a completed paper-format/reference update. |

The earlier verification runs provide implementation evidence for continuous sensing, Adaptive WIVR and return-to-Home. The three-scenario comparison requires frozen shared inputs, readiness checks for the new horizons and geometry, and saved results for all nine missions.

## Running the formal experiments

Each group follows the verification layout. The `code/` directory contains its scenario and entry point, `results/` receives mission logs and GP artifacts, and `recordings/` receives MCAP rosbag files. Use one final `Comparison.ipynb` after results are available.

| Group | Entry point | Only scenario change |
| --- | --- | --- |
| [Baseline](Baseline/README.md) | [`Baseline/code/run.sh`](Baseline/code/run.sh) | Reuse the verified closed-loop geometry, graph and execution settings. Use the new trained GP kernel. |
| [Test B](<Test B/README.md>) | [`Test B/code/run.sh`](<Test B/code/run.sh>) | Same geometry, 150 s budget and scoring window. |
| [Test C](<Test C/README.md>) | [`Test C/code/run.sh`](<Test C/code/run.sh>) | 60 × 40 m, same 300 s budget. |

Use the same ROS 2 Jazzy, Gazebo Harmonic and SRF Python environment as the [WIVR verification](<../../Verification tests/WIVR_test/README.md>). Build the [shared Nav2 packages](../Nav2_ROS/src) and the [configurable closed-loop controller](../WIVR/src) in the ROS overlay for these experiments. Use the same [multi-run RBF export](../Trained_Kernel/result/multirun_fit_20261127T025820Z/hyperparameters.json) for all groups. It currently supplies length scales of 6.24703 m east, 44.10294 m north, and 37.93368 s, a signal variance of 0.47980997 °C², and a measurement noise variance of 0.00084209 °C². Run the commands below from `Final_Test`.

```bash
kernel="../Trained_Kernel/result/multirun_fit_20261127T025820Z/hyperparameters.json"
run_name="formal_all_runs"
bash "Baseline/code/run.sh" --kernel "$kernel" --name "$run_name" --record
bash "Test B/code/run.sh" --kernel "$kernel" --name "$run_name" --record
bash "Test C/code/run.sh" --kernel "$kernel" --name "$run_name" --record
python code/evaluate.py --name "$run_name"
```

Each entry point runs MI, EMSE and Adaptive WIVR sequentially. Use `--policy` to select one policy. A shared `--name` identifies one comparison and must match across the three groups and the evaluator. The runner prepares and pins the shared inputs automatically on the first launch. Every formal mission starts with zero observations and the 25 °C prior mean in [settings.json](settings.json). The export supplies the fitted RBF length scales, signal variance, and measurement noise variance. The [same SRF](../Nav2_ROS/fixtures/field.json), GP update and replanning rules apply throughout. Shared provenance and the offline RMSE report are saved in `results/<name>/` at the Final_Test level. Use the all-data kernel and a new comparison name for the next run.
