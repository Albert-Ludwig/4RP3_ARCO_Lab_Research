# WELL-E GP Sampling Simulation

Simulation framework for Gaussian Process-based environmental field modelling and
informative sampling for the WELL-E vernal pool monitoring project.

## Repository layout

```
src/                       reusable library
  fields.py                synthetic 2D fields + grid
  samplers.py              random / lawnmower baselines
  metrics.py               path length
  solvers.py               greedy MI, sequential active sampling, fixed posterior variance
  gp_model.py              exact GP (GPyTorch) + fit/evaluate driver  (only heavy module)
notebooks/                 exploratory, audited notebooks (01-03)
experiments/
  configs/*.yaml           experiment configurations
  run_experiment.py        runs all strategies, writes timestamped results/<run>/
scripts/
  make_paper_figures.py    regenerate figures from CSVs  (no torch needed)
  make_paper_tables.py     regenerate tables from CSVs   (no torch needed)
results/<run>/             timestamped raw outputs (config.yaml + CSVs + figures/ + tables/)
docs/                      explanations and speaker notes
```

## Reproducible pipeline

The pipeline is split into a **compute** step (needs torch/gpytorch, produces raw
CSVs) and a **presentation** step (needs only numpy/pandas/matplotlib/pyyaml, reads
CSVs). This means results can be regenerated from raw data alone — no need to install
the GP stack or re-run the simulation.

### 1. Run the experiment (heavy)

```bash
pip install -r requirements.txt
python experiments/run_experiment.py            # quick budgets [10,20,30,50,64]
python experiments/run_experiment.py --full     # full budgets
python experiments/run_experiment.py --config experiments/configs/greedy_active_default.yaml
```

This writes `results/<tag>_<timestamp>/` containing `config.yaml` (resolved config for
reproducibility) and raw CSVs: `metrics.csv`, `final_maps.csv`, `selected_points.csv`,
`greedy_gains.csv`, `random_seeds.csv`, `outputscale_comparison.csv`,
`border_fraction.csv`, `field_eval.csv`. A `results/LATEST.txt` pointer is updated.

### 2. Make figures and tables (light, no GP dependencies)

```bash
pip install -r requirements-plotting.txt         # numpy/pandas/matplotlib/pyyaml only
python scripts/make_paper_figures.py             # uses results/LATEST.txt
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py --run results/greedy_active_20260622_133322   # or a specific run
```

Figures are written to `results/<run>/figures/*.png` and tables to
`results/<run>/tables/*.{md,tex}`.

## Strategies compared

Random, fine lawnmower, coarse lawnmower, Krause-style greedy mutual-information
placement, and sequential entropy active sampling (entropy ≡ max-variance for
single-point selection, kept as one method). A **two-sensor cold-start** experiment
seeds the GP with two random sensors and then samples sequentially. See
`docs/GP_Greedy_Active_Notebook_Full_Explanation.md`.
