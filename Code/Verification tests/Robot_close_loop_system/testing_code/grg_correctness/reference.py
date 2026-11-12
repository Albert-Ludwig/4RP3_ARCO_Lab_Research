"""Independent schedule, Gaussian conditioning, and finite-route references."""

import ast
from itertools import product
import inspect
from pathlib import Path
import sys
import textwrap

import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "testing_code" / "baseline"
sys.path.insert(0, str(BASE))

from grg import GRGPlanner


def schedule(points, travel, route, elapsed, start_h=12, dwell=5, home=0, sample_last=False):
    """Arrival is before sensing dwell. The start vertex is not sampled."""
    clock, query, arrivals = float(elapsed), [], []
    for i in range(1, len(route)):
        clock += float(travel[route[i - 1], route[i]])
        if i < len(route) - 1 or route[i] != home or sample_last:
            arrivals.append(clock)
            query.append([*points[route[i]], start_h + clock / 3600])
            clock += dwell
    return np.asarray(query).reshape(-1, 3), arrivals, clock


class ReferenceMI:
    """Use the fitted GPyTorch kernel and a separate conditioning solve.

    This does not call GPModel.covariance_many, GPModel.joint, or the planner's
    information_gain. The model, likelihood and normalization remain unchanged.
    """
    def __init__(self, gp):
        self.gp = gp
        self.model = gp.fitted.model
        self.x = self.model.train_inputs[0]
        self.scaler = gp.fitted.scaler
        with torch.no_grad():
            k = self.model.covar_module(self.x).to_dense().numpy()
        self.factor = cho_factor(k + gp.noise_scaled * np.eye(len(self.x)), lower=True)

    def covariance(self, queries):
        queries = np.asarray(queries)
        q = torch.as_tensor(self.scaler.transform_x(queries), dtype=torch.float64)
        with torch.no_grad():
            prior = self.model.covar_module(q).to_dense().numpy()
            cross = self.model.covar_module(q, self.x).to_dense().numpy()
        shape = cross.shape
        solved = cho_solve(self.factor, cross.reshape(-1, shape[-1]).T).T.reshape(shape)
        cov = prior - cross @ np.swapaxes(solved, -1, -2)
        return (cov + np.swapaxes(cov, -1, -2)) * (.5 * self.scaler.y_std**2)

    def scores(self, queries):
        cov = self.covariance(queries)
        factor = np.linalg.cholesky(np.eye(cov.shape[-1]) + cov / self.gp.noise)
        return np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(axis=-1)

    def routes(self, points, travel, routes, elapsed=0, dwell=5, sample_last=False):
        values = np.zeros(len(routes))
        groups = {}
        for i, route in enumerate(routes):
            query = schedule(points, travel, route, elapsed, dwell=dwell, sample_last=sample_last)[0]
            if len(query):
                groups.setdefault(len(query), []).append((i, query))
        for group in groups.values():
            index, queries = zip(*group)
            values[list(index)] = self.scores(queries)
        return values


def enumerate_routes(points, travel, current, elapsed, remaining, visited=(), blocked=(), dwell=5, max_events=3):
    """Bounded exhaustive event sequences, including consecutive and Home revisits."""
    available = [v for v in range(len(points)) if v not in blocked]
    routes = []
    for size in range(max_events + 1):
        for middle in product(available, repeat=size):
            route = [current, *middle, 0] if current else [0, *middle, 0]
            if route == [0, 0]:
                route = [0]
            finish = schedule(points, travel, route, elapsed, dwell=dwell)[2]
            if np.isfinite(finish) and finish <= elapsed + remaining + 1e-8:
                routes.append(route)
    return routes


class NoReuseDict(dict):
    def __contains__(self, key):
        return False


def uncached_plan():
    """Test-only copy: force reward cache misses without changing search code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(GRGPlanner.plan)))
    replaced = []

    class ReplaceCache(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "cache" and isinstance(node.value, ast.Dict)):
                node.value = ast.Call(func=ast.Name(id="NoReuseDict", ctx=ast.Load()), args=[], keywords=[])
                replaced.append(True)
            return self.generic_visit(node)

    tree = ast.fix_missing_locations(ReplaceCache().visit(tree))
    assert len(replaced) == 1
    scope = {**GRGPlanner.plan.__globals__, "NoReuseDict": NoReuseDict}
    exec(compile(tree, "<cache-disabled-verification-only>", "exec"), scope)
    return scope["plan"]
