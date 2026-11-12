"""Phase 1: explicit synthetic truth and reproducible sparse observations.

Units: x/y metres, t hours, temperatures degrees Celsius. These are invented
experimental conditions, NOT Dashboard/WELL-E observations or a physical model.
The GP training module never imports this module.
"""

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


def temperature_true(xyt):
    """Smooth background + daily component + moving warm/cool local patterns.

    T = 18 + .035*x - .025*y + 1.2*sin(2*pi*(t-6)/24)
        + 2.2*exp(-.5*((x-(8+.45*t))/5.5)^2-.5*((y-11)/7)^2)
        - (1+.25*cos(2*pi*t/24))*exp(-.5*((x-23)/6)^2
                                             -.5*((y-23)/5)^2).
    The sinusoid is synthetic ground truth, not evidence for a periodic kernel.
    """
    a = np.asarray(xyt, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError("Expected (..., 3) inputs: x_m, y_m, t_h")
    x, y, t = np.moveaxis(a, -1, 0)
    background = 18.0 + 0.035 * x - 0.025 * y
    daily = 1.2 * np.sin(2 * np.pi * (t - 6) / 24)
    warm = 2.2 * np.exp(-0.5 * (((x - (8 + 0.45 * t)) / 5.5)**2
                               + ((y - 11) / 7.0)**2))
    cool = (1 + 0.25 * np.cos(2 * np.pi * t / 24)) * np.exp(
        -0.5 * (((x - 23) / 6.0)**2 + ((y - 23) / 5.0)**2))
    return background + daily + warm - cool


@dataclass
class Measurements:
    x: np.ndarray
    truth: np.ndarray
    noise_z: np.ndarray
    observed: np.ndarray

    def with_noise(self, noise_std):
        return Measurements(self.x.copy(), self.truth.copy(), self.noise_z.copy(),
                            self.truth + noise_std * self.noise_z)

    def prefix(self, count):
        return Measurements(*(a[:count].copy() for a in
                              (self.x, self.truth, self.noise_z, self.observed)))


def sample_measurements(count, seed, bounds, noise_std):
    if count < 1 or noise_std < 0:
        raise ValueError("Require positive count and nonnegative noise")
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(count, 3))
    truth = temperature_true(x)
    noise_z = rng.standard_normal(count)
    return Measurements(x, truth, noise_z, truth + noise_std * noise_z)


def save_measurements(path, measurements, split):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "split", "x_m", "y_m", "t_h",
                         "truth_c", "observed_c", "noise_standard_normal"])
        for i, (x, truth, obs, z) in enumerate(zip(
                measurements.x, measurements.truth, measurements.observed,
                measurements.noise_z)):
            writer.writerow([i, split, *x, truth, obs, z])


def snapshot_grid(size, hour, domain):
    x = np.linspace(*domain[0], size)
    y = np.linspace(*domain[1], size)
    xx, yy = np.meshgrid(x, y)
    xyt = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, hour)])
    return xx, yy, xyt
