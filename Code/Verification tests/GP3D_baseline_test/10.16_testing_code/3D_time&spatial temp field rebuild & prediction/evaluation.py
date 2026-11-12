"""Accuracy and uncertainty metrics for GP predictions."""

import numpy as np
from scipy.stats import norm


def accuracy(truth, mean):
    error = np.asarray(mean) - np.asarray(truth)
    return {"rmse_c": float(np.sqrt(np.mean(error**2))),
            "mae_c": float(np.mean(np.abs(error)))}


def distribution_metrics(truth, mean, variance):
    variance = np.asarray(variance).clip(1e-12)
    std = np.sqrt(variance)
    z = (np.asarray(truth) - mean) / std
    return {**accuracy(truth, mean),
            "coverage_95": float(np.mean(np.abs(z) <= norm.ppf(0.975))),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * variance) + 0.5 * z**2)),
            "interval_width_95_c": float(np.mean(2 * norm.ppf(0.975) * std)),
            "z_mean": float(np.mean(z)), "z_std": float(np.std(z))}



def calibration_curve(truth, mean, variance):
    nominal = np.array([0.1, 0.25, 0.5, 0.68, 0.8, 0.9, 0.95, 0.99])
    z = np.abs((truth - mean) / np.sqrt(variance.clip(1e-12)))
    empirical = np.array([np.mean(z <= norm.ppf((1 + p) / 2)) for p in nominal])
    return nominal, empirical
