"""Keep the existing GRG recursion; override only its route-scoring dispatch."""

import time

import bootstrap
import numpy as np
from replay import IntelPlanner


class ObjectivePlanner(IntelPlanner):
    def __init__(self, *args, policy, catalog, weights, event_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy, self.catalog, self.weights = policy, catalog, weights
        self.event_log, self.components_log = event_log, []
        self.plan_wall_s = []

    def score_routes(self, routes, gp, elapsed, sample_last=False):
        # Every arm computes both components. MI retains the original arithmetic;
        # only the scalar returned to the unchanged recursive search differs.
        mi_scores = super().score_routes(routes, gp, elapsed, sample_last)
        emse_scores = np.zeros(len(routes))
        groups = {}
        for i, route in enumerate(routes):
            query = self.schedule(route, elapsed, sample_last)[0]
            if len(query):
                groups.setdefault(len(query), []).append((i, query))
        for group in groups.values():
            index, queries = zip(*group)
            _, emse = self.catalog.components(gp, queries)
            emse_scores[list(index)] = emse
        return self.catalog.score(self.policy, mi_scores, emse_scores, self.weights)

    def plan(self, gp, current, elapsed, remaining, visited=(), blocked=()):
        call = len(self.plan_wall_s)
        if self.event_log:
            self.event_log("plan_start", call=call, gp_version=gp.version, elapsed_s=elapsed)
        started = time.perf_counter()
        self.catalog.condition(gp)
        plan = super().plan(gp, current, elapsed, remaining, visited, blocked)
        query = self.schedule(plan.route, elapsed)[0]
        if len(query):
            mi, emse = self.catalog.components(gp, [query])
            mi, emse = float(mi[0]), float(emse[0])
        else:
            mi = emse = 0.
        recomputed = float(self.catalog.score(self.policy, mi, emse, self.weights))
        if not np.isclose(plan.gain, recomputed, rtol=1e-7, atol=1e-8):
            raise AssertionError("Selected route score disagrees with independent catalog recomputation")
        wall_s = time.perf_counter() - started
        self.plan_wall_s.append(wall_s)
        self.components_log.append(dict(call=call, gp_version=gp.version, elapsed_s=elapsed,
            policy=self.policy, route=plan.route, route_mi_nats=mi, route_emse_reduction_c2=emse,
            normalized_mi=mi / self.catalog.mi_scale, normalized_emse=emse / self.catalog.emse_scale,
            objective_score=float(plan.gain), plan_wall_s=wall_s,
            current_mean_target_variance_c2=float(self.catalog.current_target_variance.mean())))
        if self.event_log:
            self.event_log("plan_end", call=call, elapsed_s=elapsed, wall_s=wall_s)
        return plan
