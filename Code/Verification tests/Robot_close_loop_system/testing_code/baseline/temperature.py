"""Synthetic measurement backend; the planner never receives this object."""

import numpy as np

import reuse
from synthetic_field import temperature_true
from interfaces import Measurement


class SyntheticTemperatureSource:
    def __init__(self, history_x, history_noise, online_noise, seed):
        self.history_x = np.array(history_x, copy=True)
        self.history_noise = history_noise
        self.online_noise = online_noise
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1)

    def load_history(self):
        rng = np.random.default_rng(self.seed)
        values = temperature_true(self.history_x) + rng.normal(
            0, self.history_noise, len(self.history_x))
        return [Measurement(*x, float(y), "synthetic_history")
                for x, y in zip(self.history_x, values)]

    def measure(self, position, timestamp_h):
        x = [*position, timestamp_h]
        y = float(temperature_true(np.array([x]))[0] + self.rng.normal(0, self.online_noise))
        return Measurement(*x, y, "synthetic_robot")
