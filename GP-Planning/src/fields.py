import numpy as np


def create_grid(grid_size: int = 50):
    """Create a square 2D grid in [0, 1] x [0, 1]."""
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    x1, x2 = np.meshgrid(x, y)
    x_test = np.column_stack([x1.ravel(), x2.ravel()])
    return x, y, x1, x2, x_test


def _gaussian_bump(x1, x2, amplitude, cx, cy, sigma):
    r2 = (x1 - cx) ** 2 + (x2 - cy) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma**2))


def field_temperature_like(x1, x2):
    """Smooth field made from a few Gaussian bumps."""
    field = np.zeros_like(x1, dtype=float)
    bumps = [
        (1.0, 0.2, 0.3, 0.18),
        (0.8, 0.7, 0.75, 0.20),
        (-0.6, 0.45, 0.55, 0.16),
        (0.4, 0.85, 0.2, 0.12),
    ]
    for amplitude, cx, cy, sigma in bumps:
        field += _gaussian_bump(x1, x2, amplitude, cx, cy, sigma)
    return field


def field_moisture_like(x1, x2, pool_center=(0.5, 0.5), pool_sigma=0.16):
    """Smooth moisture-like field with a wetter pool region."""
    background = 0.25 + 0.15 * x2
    wet_pool = _gaussian_bump(x1, x2, amplitude=0.9, cx=pool_center[0], cy=pool_center[1], sigma=pool_sigma)
    return background + wet_pool


def field_transition_like(x1, x2, pool_center=(0.5, 0.5), r0=0.25, k=35.0):
    """Sigmoid radial transition field for wet/dry boundary tests."""
    r = np.sqrt((x1 - pool_center[0]) ** 2 + (x2 - pool_center[1]) ** 2)
    return 1.0 / (1.0 + np.exp(-k * (r0 - r)))


def get_field(field_name: str, x1, x2):
    key = field_name.lower()
    if key in {"temperature", "smooth", "field_a"}:
        return field_temperature_like(x1, x2)
    if key in {"moisture", "field_b"}:
        return field_moisture_like(x1, x2)
    if key in {"transition", "field_c"}:
        return field_transition_like(x1, x2)
    raise ValueError(f"Unknown field_name '{field_name}'.")


def sample_noisy_measurements(field, sample_points, noise_std=0.05, rng=None):
    """Sample field values at selected points and add Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()

    grid_size_y, grid_size_x = field.shape
    ix = np.clip(np.round(sample_points[:, 0] * (grid_size_x - 1)).astype(int), 0, grid_size_x - 1)
    iy = np.clip(np.round(sample_points[:, 1] * (grid_size_y - 1)).astype(int), 0, grid_size_y - 1)

    clean = field[iy, ix]
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(clean))
    noisy = clean + noise
    return clean, noisy
