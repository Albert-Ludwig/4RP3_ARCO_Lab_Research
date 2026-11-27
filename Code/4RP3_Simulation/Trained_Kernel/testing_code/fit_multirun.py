"""Fit shared ARD RBF hyperparameters to independent real WELL-E traversals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import check_grad, minimize


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def block(rows):
    x = np.array([[float(row[k]) for k in ("x_east_m", "y_north_m", "t_s")] for row in rows])
    y = np.array([float(row["temperature_c"]) for row in rows])
    if not (len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all()
            and (np.diff(x[:, 2]) > 0).all()):
        raise ValueError("Each traversal requires finite observations with increasing times")
    stamps = [int(row["timestamp_utc_ns"]) for row in rows]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("UTC timestamps must strictly increase within a traversal")
    squared = [np.asfortranarray((values[:, None] - values[None, :])**2) for values in x.T]
    return squared, np.column_stack((y, np.ones_like(y))), np.eye(len(y), order="F")


def marginal(log_params, blocks):
    ell_x, ell_y, ell_t, signal, noise = np.exp(log_params)
    inv_lengths = 1 / np.square([ell_x, ell_y, ell_t])
    loss = 0.0
    gradient = np.zeros(5)
    means = []
    for squared, targets, identity in blocks:
        kernel = -0.5 * sum(inv * d for inv, d in zip(inv_lengths, squared))
        np.exp(kernel, out=kernel)
        kernel *= signal
        cov = kernel.copy(order="F")
        cov.ravel(order="F")[::len(cov) + 1] += noise
        factor = cho_factor(cov, lower=True, overwrite_a=True, check_finite=False)
        inv_y, inv_one = cho_solve(factor, targets, check_finite=False).T
        mean = inv_y.sum() / inv_one.sum()
        means.append(float(mean))
        residual = targets[:, 0] - mean
        alpha = inv_y - mean * inv_one
        loss += 0.5 * (residual @ alpha + len(alpha) * np.log(2 * np.pi))
        loss += np.log(factor[0].diagonal()).sum()
        weight = cho_solve(factor, identity, check_finite=False) - np.outer(alpha, alpha)
        weighted_kernel = weight * kernel
        gradient[:3] += [0.5 * inv * np.einsum("ij,ij->", weighted_kernel, d)
                         for inv, d in zip(inv_lengths, squared)]
        gradient[3] += 0.5 * weighted_kernel.sum()
        gradient[4] += 0.5 * noise * np.trace(weight)
    n = sum(len(targets) for _, targets, _ in blocks)
    return loss / n, gradient / n, means


CAMPAIGN_DATES = ("2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-21")


def fit(legacy, campaign, output, holdout=None):
    dated_rows = {"2026-11-03": read_rows(legacy)}
    train_dates = tuple(date for date in CAMPAIGN_DATES if date != holdout)
    dated_rows.update({date: read_rows(campaign / date / "gp_samples.csv") for date in train_dates})
    blocks = [block(rows) for rows in dated_rows.values()]
    starts = [(4, 20, 25, 0.2, 0.001),
              (15, 15, 100, 1, 0.005),
              (40, 40, 350, 2.5, 0.02)]
    ranges = [(0.25, 300), (0.25, 300), (1, 3600), (1e-4, 100), (1e-6, 4)]
    bounds = np.log(ranges)
    tiny = [block(list(rows[::max(1, len(rows)//12)])) for rows in dated_rows.values()]
    point = np.log(starts[1])
    gradient_error = check_grad(lambda p: marginal(p, tiny)[0],
                                lambda p: marginal(p, tiny)[1], point)
    if gradient_error > 1e-4:
        raise AssertionError(f"Analytic gradient differs by {gradient_error}")
    attempts, history, fits = [], [], []
    for i, start in enumerate(starts, 1):
        started = time.perf_counter()
        evaluations = 0

        def objective(p):
            nonlocal evaluations
            value, grad, _ = marginal(p, blocks)
            evaluations += 1
            history.append(dict(restart=i, evaluation=evaluations,
                                negative_mll_per_observation=float(value), **dict(zip(
                                    ("ell_x_m", "ell_y_m", "ell_t_s", "signal_c2", "noise_c2"),
                                    np.exp(p).tolist()))))
            return value, grad

        result = minimize(objective, np.log(start), jac=True, method="L-BFGS-B",
                          bounds=bounds, options={"maxiter": 90, "ftol": 1e-10, "gtol": 1e-6})
        attempts.append(dict(restart=i, converged=bool(result.success), iterations=result.nit,
                             evaluations=result.nfev, negative_mll_per_observation=float(result.fun),
                             seconds=time.perf_counter() - started, termination=str(result.message)))
        fits.append(result)
        print(f"Restart {i}: loss={result.fun:.8f}, steps={result.nit}, converged={result.success}, "
              f"{attempts[-1]['seconds']:.1f} s", flush=True)

    successful = [i for i, fit in enumerate(fits) if fit.success and np.isfinite(fit.fun)]
    if not successful:
        raise RuntimeError("No RBF restart converged")
    selected = min(successful, key=lambda i: fits[i].fun)
    best = fits[selected]
    _, _, means = marginal(best.x, blocks)
    parameters = np.exp(best.x)
    if not (np.isfinite(parameters).all() and (parameters > 0).all()):
        raise ValueError("Nonfinite or nonpositive fitted parameters")
    n = sum(map(len, dated_rows.values()))
    run_means = dict(zip(dated_rows, means))
    hp = dict(kernel="3D ARD RBF with Gaussian measurement noise",
              lengthscales_m_m_s=parameters[:3].tolist(),
              output_variance_c2=float(parameters[3]), noise_variance_c2=float(parameters[4]),
              noise_std_c=float(np.sqrt(parameters[4])), mean_c=float(np.mean(means)),
              mean_c_role="Average of fitted traversal means for export; formal simulation fixes its own prior mean",
              training_observations=n, training_dates=list(dated_rows),
              reserved_validation_date=holdout, independent_test_performed=False,
              negative_training_mll_per_observation=float(best.fun),
              selected_restart=selected + 1)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "hyperparameters.json", hp)
    write_csv(output / "training_history.csv", history)
    write_csv(output / "initialization_comparison.csv", attempts)
    write_json(output / "training_summary.json", {
        "observations_by_date": {k: len(v) for k, v in dated_rows.items()},
        "fitted_run_means_c": run_means,
        "coordinate_frame": "Shared 3 November local ENU",
        "time_input": "t_s since each run's first valid GPS fix; no cross-run covariance",
        "fit": "Exact Gaussian marginal likelihood with a GLS constant mean per run and shared covariance/noise",
        "optimizer": "L-BFGS-B in log physical units, selecting the best converged restart by training likelihood",
        "starts": starts, "bounds": ranges, "gradient_check_error": float(gradient_error),
        "bound_parameters": [
            name for name, value, (lower, upper) in zip(
                ("ell_x", "ell_y", "ell_t", "signal_variance", "noise_variance"), best.x, bounds)
            if min(abs(value-lower), abs(upper-value)) < 1e-5
        ],
        "held_out_data": f"{holdout} excluded from fitting and restart selection" if holdout else None,
        "held_out_score": None,
    })
    check_inputs = [("2026-11-03", legacy)] + [(day, campaign / day / "gp_samples.csv") for day in train_dates]
    write_json(output / "manifest.json", {
        "input_sha256_by_date": {day: hashlib.sha256(p.read_bytes()).hexdigest() for day, p in check_inputs},
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
    })
    print(json.dumps(hp, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", choices=CAMPAIGN_DATES,
                        help="Exclude one whole campaign traversal. By default, fit all six runs.")
    args = parser.parse_args()
    fit(args.legacy, args.campaign, args.output, args.holdout)
