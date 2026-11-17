"""Sampling solvers for greedy / active GP sampling.

Pure-NumPy strategy logic shared by the notebook and the experiment runner:
- closed-form fixed-kernel posterior variance,
- Krause (2008) greedy mutual-information placement,
- sequential variance/entropy active sampling.

These functions take a fixed RBF kernel as explicit arguments (no globals), so
they are reusable and testable. Selection uses only the kernel geometry; the GP
fit used for evaluation lives in ``gp_model.py``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedKernel:
    """Fixed stationary RBF kernel used for selection and the fixed-mode GP."""

    lengthscale: float
    outputscale: float
    noise_variance: float


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


def rbf_covariance(a: np.ndarray, b: np.ndarray, lengthscale: float, outputscale: float) -> np.ndarray:
    d2 = pairwise_sq_dists(a, b)
    return outputscale * np.exp(-0.5 * d2 / (lengthscale ** 2))


def fixed_posterior_variance(train_points: np.ndarray, query_points: np.ndarray, kernel: FixedKernel) -> np.ndarray:
    """Closed-form GP posterior variance at ``query_points`` given ``train_points``."""
    ls, os_, nv = kernel.lengthscale, kernel.outputscale, kernel.noise_variance
    if len(train_points) == 0:
        k_qq = rbf_covariance(query_points, query_points, ls, os_)
        return np.maximum(np.diag(k_qq), 1e-12)

    k_xx = rbf_covariance(train_points, train_points, ls, os_)
    k_xx = k_xx + nv * np.eye(len(train_points))
    k_qx = rbf_covariance(query_points, train_points, ls, os_)
    k_qq_diag = np.diag(rbf_covariance(query_points, query_points, ls, os_))

    solve_term = np.linalg.solve(k_xx, k_qx.T)
    var = k_qq_diag - np.sum(k_qx * solve_term.T, axis=1)
    return np.maximum(var, 1e-12)


def _conditional_variances(selected: list, available: np.ndarray, K_all: np.ndarray, noise_variance: float):
    """Per-candidate conditional variances used by the Krause greedy MI criterion.

    For every index in ``available``, returns:
        var_given_selected: sigma^2(y | selected)
        var_given_rest:     sigma^2(y | (available \\ {y}))   [i.e. V \\ (selected u {y})]
    via the precision trick var(y | rest) = 1 / (Sigma^{-1})_yy, where Sigma is the
    posterior covariance of ``available`` given ``selected``. Shared by the greedy
    argmax sampler and ``path_measurement_objective`` (forced-order evaluation) so
    both use identical linear algebra.
    """
    jitter = 1e-8
    if len(selected) == 0:
        sigma_av = K_all[np.ix_(available, available)].astype(float).copy()
    else:
        A = np.array(selected, dtype=int)
        k_AA = K_all[np.ix_(A, A)] + noise_variance * np.eye(len(A))
        k_vA = K_all[np.ix_(available, A)]
        sigma_av = K_all[np.ix_(available, available)] - k_vA @ np.linalg.solve(k_AA, k_vA.T)

    var_given_selected = np.maximum(np.diag(sigma_av), 1e-12)
    precision = np.linalg.inv(sigma_av + jitter * np.eye(len(available)))
    var_given_rest = np.maximum(1.0 / np.diag(precision), 1e-12)
    return var_given_selected, var_given_rest


def greedy_information_sampler(candidate_points: np.ndarray, n_samples: int, kernel: FixedKernel, initial_indices=None):
    """Krause et al. (2008) greedy mutual-information placement.

    At each step adds the candidate maximizing the greedy MI gain
        delta(y) = 0.5 * log( sigma^2(y | A) / sigma^2(y | V \\ (A u {y})) )
    The denominator is computed with the precision trick:
        var(y | all other available) = 1 / (Sigma^{-1})_ii
    where Sigma is the posterior covariance of the available points given A.

    Returns (selected_indices, selected_points, marginal_gains, cumulative_objective).
    """
    K_all = rbf_covariance(candidate_points, candidate_points, kernel.lengthscale, kernel.outputscale)
    nv = kernel.noise_variance
    selected = [] if initial_indices is None else list(initial_indices)
    n_cand = len(candidate_points)

    gains, objectives = [], []
    current_obj = 0.0

    while len(selected) < n_samples:
        selected_set = set(selected)
        available = np.array([i for i in range(n_cand) if i not in selected_set], dtype=int)

        var_given_A, var_given_rest = _conditional_variances(selected, available, K_all, nv)

        gain_all = 0.5 * np.log(var_given_A / var_given_rest)
        best_pos = int(np.argmax(gain_all))
        best_i = int(available[best_pos])

        selected.append(best_i)
        current_obj += float(gain_all[best_pos])
        gains.append(float(gain_all[best_pos]))
        objectives.append(float(current_obj))

    return selected, candidate_points[selected], np.array(gains), np.array(objectives)


def path_measurement_objective(ordered_indices, kernel: FixedKernel, candidate_points: np.ndarray) -> float:
    """Krause mutual-information objective F(A) for an arbitrary (non-greedy) subset A.

    ``ordered_indices`` is any ordering of the target subset A (indices into
    ``candidate_points``); duplicates are dropped, first occurrence kept.
    NOTE: this sequential greedy-gain sum is order-dependent (shuffling the
    same set produces a different scalar). It is NOT the true mutual information
    I(y_A ; y_{V\\A}); use ``src.planners.gp_mutual_information`` for the
    order-independent GRG objective (Binney, Krause & Sukhatme, IJRR 2013).
    """
    seen = set()
    dedup = []
    for idx in ordered_indices:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            dedup.append(idx)

    if len(dedup) == 0:
        return 0.0

    K_all = rbf_covariance(candidate_points, candidate_points, kernel.lengthscale, kernel.outputscale)
    nv = kernel.noise_variance
    n_cand = len(candidate_points)

    selected: list = []
    current_obj = 0.0
    for next_idx in dedup:
        selected_set = set(selected)
        available = np.array([i for i in range(n_cand) if i not in selected_set], dtype=int)
        var_given_A, var_given_rest = _conditional_variances(selected, available, K_all, nv)

        pos = int(np.searchsorted(available, next_idx))
        gain = 0.5 * float(np.log(var_given_A[pos] / var_given_rest[pos]))

        selected.append(next_idx)
        current_obj += gain

    return current_obj


def min_distance_mask(points: np.ndarray, selected_points, min_dist: float) -> np.ndarray:
    if len(selected_points) == 0:
        return np.ones(len(points), dtype=bool)
    s = np.array(selected_points)
    d2 = pairwise_sq_dists(points, s)
    dmin = np.sqrt(np.min(d2, axis=1))
    return dmin > min_dist


def _point_key(p):
    return (round(float(p[0]), 8), round(float(p[1]), 8))


def sequential_active_sampler(
    candidate_points: np.ndarray,
    n_samples: int,
    kernel: FixedKernel,
    r_min: float,
    measure_fn,
    acquisition: str = "entropy",
    initial_points=None,
):
    """Sequential variance/entropy active sampling on the candidate grid.

    ``measure_fn(points_array) -> noisy_values`` returns noisy measurements for the
    given points (used only to accumulate training targets; selection uses the
    fixed-kernel posterior variance, which is data-independent). Entropy and
    variance acquisition select identical points by construction.
    """
    if initial_points is None:
        initial_points = candidate_points[[0, len(candidate_points) // 2]]

    selected = [np.array(p, dtype=float) for p in initial_points]
    train_y = list(np.atleast_1d(measure_fn(np.array(selected))))
    gains_trace = []

    while len(selected) < n_samples:
        X_train = np.array(selected, dtype=float)
        var_all = fixed_posterior_variance(X_train, candidate_points, kernel)

        if acquisition == "variance":
            score = var_all.copy()
        elif acquisition == "entropy":
            score = 0.5 * np.log(2.0 * np.pi * np.e * var_all)
        else:
            raise ValueError("acquisition must be 'variance' or 'entropy'")

        chosen_set = set(_point_key(p) for p in selected)
        not_chosen = np.array([_point_key(p) not in chosen_set for p in candidate_points], dtype=bool)
        far_enough = min_distance_mask(candidate_points, selected, r_min)
        valid = not_chosen & far_enough
        if not np.any(valid):
            valid = not_chosen

        masked_score = np.where(valid, score, -np.inf)
        next_idx = int(np.argmax(masked_score))

        next_point = candidate_points[next_idx]
        noisy_val = float(np.atleast_1d(measure_fn(next_point.reshape(1, 2)))[0])

        selected.append(next_point)
        train_y.append(noisy_val)
        gains_trace.append(float(score[next_idx]))

    return np.array(selected), np.array(train_y), np.array(gains_trace)


def border_fraction(points: np.ndarray, tol: float = 1e-9) -> float:
    """Fraction of points lying on the [0,1]^2 domain edge."""
    x, y = points[:, 0], points[:, 1]
    on_border = (
        (np.abs(x - 0.0) < tol) | (np.abs(x - 1.0) < tol) |
        (np.abs(y - 0.0) < tol) | (np.abs(y - 1.0) < tol)
    )
    return float(np.mean(on_border))
