"""Backend contracts. Positions: m; GP time: h; mission time: s."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Measurement:
    x_m: float
    y_m: float
    t_h: float
    temperature_c: float
    source: str
    status: str = "ok"

    def valid(self):
        return self.status == "ok" and bool(np.isfinite(
            [self.x_m, self.y_m, self.t_h, self.temperature_c]).all())


@dataclass(frozen=True)
class RobotState:
    x_m: float
    y_m: float
    yaw_rad: float
    elapsed_s: float

    @property
    def xy(self):
        return np.array([self.x_m, self.y_m])


@dataclass
class NavigationResult:
    status: str
    state: RobotState
    distance_m: float
    travel_s: float
    path: np.ndarray


class TemperatureSource(Protocol):
    def load_history(self) -> list[Measurement]: ...
    def measure(self, position, timestamp_h) -> Measurement: ...


class Navigation(Protocol):
    def get_robot_state(self) -> RobotState: ...
    def estimate_travel_cost(self, start, goal) -> float: ...
    def navigate_to(self, goal) -> NavigationResult: ...
    def wait(self, duration_s) -> None: ...
