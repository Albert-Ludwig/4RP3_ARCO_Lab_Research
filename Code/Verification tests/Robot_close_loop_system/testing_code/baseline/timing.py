"""Mission-time ledger in seconds, separate from diagnostic wall-clock runtimes."""

from dataclasses import asdict, dataclass


@dataclass
class MissionTime:
    travel_s: float = 0.0
    sampling_s: float = 0.0
    online_computation_s: float = 0.0

    @property
    def total_s(self):
        return self.travel_s + self.sampling_s + self.online_computation_s

    def as_dict(self):
        return {**asdict(self), "total_s": self.total_s}
