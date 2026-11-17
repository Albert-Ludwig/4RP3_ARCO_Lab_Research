import numpy as np


def random_sampler(domain_points: np.ndarray, n_samples: int, rng=None):
    """Pick random unique points from a set of domain points."""
    if rng is None:
        rng = np.random.default_rng()
    if n_samples > len(domain_points):
        raise ValueError("n_samples cannot exceed number of domain points")
    idx = rng.choice(len(domain_points), size=n_samples, replace=False)
    return domain_points[idx]


def lawnmower_path_from_grid(x: np.ndarray, y: np.ndarray):
    """Return back-and-forth coverage path over a structured grid."""
    points = []
    for row_index, y_val in enumerate(y):
        x_row = x if row_index % 2 == 0 else x[::-1]
        for x_val in x_row:
            points.append((x_val, y_val))
    return np.array(points, dtype=float)


def lawnmower_sampler(x: np.ndarray, y: np.ndarray, n_samples: int):
    """Take first n_samples points from lawnmower path."""
    path = lawnmower_path_from_grid(x, y)
    if n_samples > len(path):
        raise ValueError("n_samples cannot exceed number of grid points")
    return path[:n_samples]
