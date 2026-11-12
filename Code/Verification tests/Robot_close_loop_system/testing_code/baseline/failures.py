"""Candidate exclusions: map-confirmed permanent bans and one-plan cooldowns."""

from dataclasses import asdict, dataclass, field

PERMANENT_REASONS = {"static_collision", "static_no_path"}


@dataclass
class TemporaryFailure:
    reason: str
    retry_at_plan: int


@dataclass
class CandidateFailures:
    permanent: dict = field(default_factory=dict)
    temporary: dict = field(default_factory=dict)

    def mark_permanent(self, node, reason):
        if reason not in PERMANENT_REASONS:
            raise ValueError("A permanent ban requires explicit static-map evidence")
        self.permanent[node] = reason
        self.temporary.pop(node, None)

    def mark_temporary(self, node, planning_round, reason):
        if node not in self.permanent:
            self.temporary[node] = TemporaryFailure(reason, planning_round + 2)

    def measurement_failed(self, node, planning_round, reason="measurement_failed"):
        """Reserved sensor-failure entry; no failure detector or retry is added."""
        self.mark_temporary(node, planning_round, reason)

    def blocked_for(self, planning_round):
        for node, failure in list(self.temporary.items()):
            if planning_round >= failure.retry_at_plan:
                del self.temporary[node]
        return set(self.permanent) | set(self.temporary)

    def snapshot(self):
        return {"permanent": dict(sorted(self.permanent.items())),
                "temporary": {node: asdict(failure) for node, failure in sorted(self.temporary.items())}}
