import numpy as np


def path_length(points: np.ndarray) -> float:
    """Compute cumulative Euclidean path length for ordered points."""
    if len(points) < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))
