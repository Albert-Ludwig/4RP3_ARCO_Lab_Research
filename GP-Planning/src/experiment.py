from dataclasses import dataclass

import numpy as np

from .fields import create_grid, get_field, sample_noisy_measurements
from .metrics import path_length
from .samplers import lawnmower_sampler, random_sampler


@dataclass
class Week1Result:
    field: np.ndarray
    random_points: np.ndarray
    lawnmower_points: np.ndarray
    random_clean: np.ndarray
    random_noisy: np.ndarray
    lawnmower_clean: np.ndarray
    lawnmower_noisy: np.ndarray
    random_path_length: float
    lawnmower_path_length: float


def run_week1_baselines(field_name="temperature", grid_size=50, n_samples=120, noise_std=0.05, seed=7):
    rng = np.random.default_rng(seed)

    x, y, x1, x2, x_test = create_grid(grid_size=grid_size)
    field = get_field(field_name, x1, x2)

    random_points = random_sampler(x_test, n_samples=n_samples, rng=rng)
    lawnmower_points = lawnmower_sampler(x, y, n_samples=n_samples)

    random_clean, random_noisy = sample_noisy_measurements(field, random_points, noise_std=noise_std, rng=rng)
    lawnmower_clean, lawnmower_noisy = sample_noisy_measurements(field, lawnmower_points, noise_std=noise_std, rng=rng)

    return Week1Result(
        field=field,
        random_points=random_points,
        lawnmower_points=lawnmower_points,
        random_clean=random_clean,
        random_noisy=random_noisy,
        lawnmower_clean=lawnmower_clean,
        lawnmower_noisy=lawnmower_noisy,
        random_path_length=path_length(random_points),
        lawnmower_path_length=path_length(lawnmower_points),
    )
