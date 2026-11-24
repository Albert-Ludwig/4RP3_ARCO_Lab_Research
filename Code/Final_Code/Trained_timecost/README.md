# GRG Travel-Time Calibration

GRG estimates a leg's duration as $d_{ij}=L_{ij}/v+\delta$, with the Nav2 path length $L_{ij}$ in metres, effective speed $v$ in m/s and per-leg delay $\delta$ in seconds. It uses the duration to check the sampling budget and estimate the timing of along-route samples.

## Selected estimator

Use the joint least-squares fit for the next ROS comparison. Use the same estimate for the mission-budget check, including the Nav2 return to Home, and the timing of predicted samples on the normal sampling route. The effective speed is **0.891822 m/s**, and the per-leg delay is **2.709215 s**.

$$
\min_{v>0,\;\delta\ge 0}\sum_k
\left[T_k-\left(\frac{L_k}{v}+\delta\right)\right]^2 .
$$

Positive and negative errors of the same size have equal weight. Squaring the error gives larger misses more weight, which matches the goal of keeping both overestimates and underestimates small.

In a leave-one-run-out check on the 11 training runs, the selected fit had an RMSE of **1.910 s**. Equal weighting by run gave 1.912 s, and the best of three Huber thresholds gave 1.913 s. On the 33 held-out legs, equal weighting improved RMSE by just 0.002 s and increased the largest absolute error slightly. The original fit is therefore retained. Its held-out RMSE was **2.178 s**, and its largest absolute error was **5.623 s**.

The 300 s limit runs from departure to actual arrival at Home. Use the same fitted model for sampling legs and the required return. During the Nav2 return, keep sensing at 1 Hz and include the valid return observations in the final GP. The policy continues to score the normal sampling route. This integration is pending.

## Comparison with fixed measured speed

The fixed-speed fit uses $v=\sum_k S_k/\sum_k T_k=0.792$ m/s, where $S_k$ is the Gazebo ground-truth trajectory distance and $T_k$ is the elapsed time from sending the sampling goal to successful arrival. The 113 training legs cover 2,340.24 m in 2,955.58 s. This elapsed-time average includes acceleration, turning and any pauses during a leg.

With $v$ fixed, the script minimizes $\sum_k |T_k-(L_k/v+\delta)|$ over $\delta\ge 0$. The unconstrained median residual is −0.478 s, so the constrained optimum is $\delta=0$ s.

## Held-out comparison

| Estimator | $v$ (m/s) | $\delta$ (s) | MAE (s) | RMSE (s) | Underestimated legs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original settings | 0.650 | 8.000 | 14.372 | 14.998 | 0/33 |
| Joint least-squares fit | 0.892 | 2.709 | 1.547 | 2.178 | 24/33 |
| Joint fit with training median residual added | 0.892 | 3.205 | 1.444 | 2.312 | 14/33 |
| Joint fit with training 90th-percentile residual added | 0.892 | 4.637 | 2.341 | 3.137 | 2/33 |
| Fixed measured speed with MAE-fitted delay | 0.792 | 0.000 | 1.523 | 2.580 | 9/33 |

Residual adjustments for the joint fit use leave-one-run-out predictions from the training runs. The fixed-speed fit underestimated fewer legs than the joint fit with the median adjustment, with a larger worst underestimate (4.264 s versus 1.909 s).

## Data and reproduction

Bunker Mini Gazebo / ROS 2 runs are from [MI_Test](../../Verification%20tests/MI_Test/results), [EMSE_Test](../../Verification%20tests/EMSE_Test/results), and [WIVR_test](../../Verification%20tests/WIVR_test/results). The fixed split uses 113 legs from 11 training runs and 33 legs from three held-out runs, one per policy.

The script reads planned paths from `graph.json`, goal and arrival timestamps from `events.jsonl`, and measured distances from `trajectory.npy`. Return legs, unsuccessful goals, and waiting between legs are excluded. [legs.csv](legs.csv) records individual distances, durations and predictions. [model.json](model.json) stores full-precision estimates, source hashes and validation metrics.

Run `python fit_timecost.py` with NumPy to reproduce the fitted parameters and held-out results in the table.
