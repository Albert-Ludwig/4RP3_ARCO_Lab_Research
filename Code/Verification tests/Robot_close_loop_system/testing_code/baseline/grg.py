"""Time-aware GRG with spatial revisits and planned Home measurements."""

from dataclasses import dataclass
import time

import numpy as np


def information_gain(cov, noise):
    if noise <= 0:
        raise ValueError("Observation variance must be positive")
    if cov.shape[-1] == 0:
        return np.zeros(cov.shape[:-2])
    sign, value = np.linalg.slogdet(np.eye(cov.shape[-1]) + cov / noise)
    if np.any(sign <= 0):
        raise FloatingPointError("Invalid information covariance")
    return value * 0.5


@dataclass
class Plan:
    route: list
    arrivals_s: list
    finish_s: float
    gain: float
    runtime_s: float
    evaluations: int
    gp_version: int


class GRGPlanner:
    def __init__(self, points, travel_s, start_h, sample_s, depth=2, time_splits=2,
                 home=0, allow_revisit=True, sampling_time=None):
        if not allow_revisit:
            raise ValueError("Spatial revisits are enabled in the current baseline")
        self.points = np.asarray(points)
        self.travel = np.asarray(travel_s)
        self.start_h, self.sample_s = start_h, sample_s
        self.depth, self.time_splits = depth, time_splits
        self.home, self.allow_revisit = home, True
        self.sampling_time = sampling_time
        if depth < 1 or time_splits < 1 or sample_s <= 0:
            raise ValueError("Invalid search settings")

    def schedule(self, route, elapsed, sample_last=False):
        """Only the final Home vertex is return-only; earlier Home visits sample.

        Recursive subproblems use sample_last=True when their endpoint is a
        measurement, including a Home measurement. The initial vertex is state,
        not an automatic observation.
        """
        arrivals, query = [], []
        for index, (a, b) in enumerate(zip(route[:-1], route[1:]), start=1):
            elapsed += self.travel[a, b]
            if index < len(route) - 1 or b != self.home or sample_last:
                if self.sampling_time is not None:
                    elapsed = self.sampling_time(b, float(elapsed))
                arrivals.append(float(elapsed))
                query.append([*self.points[b], self.start_h + elapsed / 3600])
                elapsed += self.sample_s
        return np.array(query).reshape(-1, 3), arrivals, float(elapsed)

    def score_routes(self, routes, gp, elapsed, sample_last=False):
        scores = np.zeros(len(routes))
        groups = {}
        for i, route in enumerate(routes):
            query = self.schedule(route, elapsed, sample_last)[0]
            if len(query):
                groups.setdefault(len(query), []).append((i, query))
        for group in groups.values():
            # Bound temporary cross-covariance memory for the 2,000-row GP.
            for start in range(0, len(group), 64):
                index, queries = zip(*group[start:start + 64])
                scores[list(index)] = information_gain(gp.covariance_many(np.array(queries)), gp.noise)
        return scores

    def plan(self, gp, current, elapsed, remaining, visited=(), blocked=()):
        """Visited nodes remain eligible. Every occurrence is scored at its arrival time.

        Home may be selected as a measurement destination, including at the
        current Home position. The final Home vertex only closes the route.
        There is no explicit wait-in-place action.
        """
        started = time.perf_counter()
        deadline = elapsed + remaining
        excluded = set(blocked)
        available = [i for i in range(len(self.points)) if i not in excluded]
        cache = {}

        def scores(routes, sample_end=False):
            missing = list(dict.fromkeys(tuple(p) for p in routes if (tuple(p), sample_end) not in cache))
            if missing:
                values = self.score_routes(missing, gp, elapsed, sample_last=sample_end)
                cache.update(((p, sample_end), value) for p, value in zip(missing, values))
            return [cache[tuple(p), sample_end] for p in routes]

        def search(prefix, end, end_s, depth, sample_end=False):
            start = prefix[-1]
            now = self.schedule(prefix, elapsed, sample_last=True)[2]
            base = prefix if len(prefix) == 1 and start == end and not sample_end else prefix + [end]
            if self.schedule(base, elapsed, sample_last=sample_end)[2] > end_s + 1e-8:
                return None
            if depth == 0:
                return base
            # Each insertion is an intentional measurement, even when its
            # location equals the current or terminal sampling location.
            # The left prefix already contains the shared recursive junction;
            # passing it directly to the right branch merges that event once.
            choices = available
            if depth == 1:
                routes = [base]
                for v in choices:
                    route = prefix + [v, end]
                    if self.schedule(route, elapsed, sample_last=sample_end)[2] <= end_s + 1e-8:
                        routes.append(route)
                return routes[int(np.argmax(scores(routes, sample_end)))]
            best, best_score = base, scores([base], sample_end)[0]
            for v in choices:
                lo = self.schedule(prefix + [v], elapsed, sample_last=True)[2]
                hi = end_s - self.travel[v, end] - (self.sample_s if sample_end else 0)
                if lo > hi + 1e-8:
                    continue
                for split in np.unique(np.linspace(lo, hi, self.time_splits + 2)[1:-1]):
                    left = search(prefix, v, split, depth - 1, sample_end=True)
                    if left is None:
                        continue
                    right = search(left, end, end_s, depth - 1, sample_end=sample_end)
                    if right is None:
                        continue
                    value = scores([right], sample_end)[0]
                    if value > best_score + 1e-12:
                        best, best_score = right, value
            return best

        route = search([current], self.home, deadline, self.depth)
        if route is None:
            return Plan([], [], np.inf, 0.0, time.perf_counter() - started, len(cache), gp.version)
        _, arrivals, finish = self.schedule(route, elapsed)
        return Plan(route, arrivals, finish, float(scores([route])[0]),
                    time.perf_counter() - started, len(cache), gp.version)
