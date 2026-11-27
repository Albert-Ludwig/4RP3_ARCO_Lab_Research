# Trained Real-Data Kernel

## Multi-run RBF fit

[Executed training report](test_report.ipynb)

The formal export uses **all 6,448 aligned temperature measurements** across six traversals on 3 November and 17-21 November 2026. The 21 November traversal contributes 1,361 measurements. This full-data fit has no held-out traversal or independent real-data accuracy score.

| Quantity | Multi-run fit |
| --- | ---: |
| East length scale (m) | 6.24703 |
| North length scale (m) | 44.10294 |
| Time length scale (s) | 37.93368 |
| Fitted signal variance (deg C squared) | 0.47980997 |
| Fitted measurement noise variance (deg C squared) | 0.00084209 |

The exact Gaussian likelihood treats each traversal as an independent block, with a separate fitted constant mean. The ARD RBF length scales, signal variance, and measurement noise are shared across runs. Inputs use the shared local ENU frame and elapsed seconds from the first valid GPS fix in each traversal. L-BFGS-B optimizes log parameters in physical units from three starting points. The best converged restart is selected by training likelihood. The alignment retains the logged zero-extra-offset assumption.

Use [hyperparameters.json](result/multirun_fit_20261127T025820Z/hyperparameters.json) as the formal runner's kernel export. The formal cold-start configuration imports the length scales, signal variance, and measurement noise variance. It starts with zero observations and a fixed prior mean of 25 °C. See the [run summary](result/multirun_fit_20261127T025820Z/training_summary.json) and [optimizer history](result/multirun_fit_20261127T025820Z/training_history.csv) for provenance and convergence.

To reproduce, run [fit_multirun.py](testing_code/fit_multirun.py) in a Python environment with NumPy and SciPy. Pass the [3 November CSV](<../../../Data/Interpolated Data/result/gp_samples.csv>) as `--legacy`, the [November campaign result folder](<../../../Data/Interpolated Data/result/2026-11-17-johnson-baseball/>) as `--campaign`, and a new directory as `--output`.

## Archived fits

The [earlier five-run fit](result/multirun_fit_20261127T015936Z/hyperparameters.json) uses 5,087 measurements and excludes 21 November. Its source and outputs are retained with that run.

### Single-run fit

Fits and stores the RBF spatiotemporal GP trained on all 980 aligned WELL-E measurements, with scalers and coordinate/time metadata. This full-data fit has no independent held-out score.

[Saved kernel](result/full_fit_20261105T031701003839Z/spatiotemporal_gp.pt) | [Training report](test_report.ipynb)
