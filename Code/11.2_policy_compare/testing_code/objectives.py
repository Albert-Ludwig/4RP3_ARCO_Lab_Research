"""Exact, fixed-hyperparameter posterior route objectives; no future labels.

The event catalog contains only coordinates and acquisition timestamps. Condition
the initial GP covariance on acquired online events, then use Schur complements.
"""

import time

import bootstrap
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from grg import information_gain


def event_key(row, start_h):
    return (round(float(row[0]) * 1e6), round(float(row[1]) * 1e6),
            round((float(row[2]) - start_h) * 3600 * 1e6))


def kernel(gp, a, b):
    delta = (a[:, None, :] - b[None, :, :]) / gp.length
    return gp.scale * np.exp(-0.5 * np.sum(delta * delta, axis=-1))


def project(gp, query):
    scaled = gp.fitted.scaler.transform_x(query)
    result = np.empty((len(query), len(gp.x)))
    for begin in range(0, len(query), 128):
        cross = kernel(gp, scaled[begin:begin + 128], gp.x)
        result[begin:begin + 128] = solve_triangular(
            gp.factor, cross.T, lower=True, check_finite=False).T
    return scaled, result


def create_catalog(gp, source, points, grid, config, destination):
    """Precompute initial-D covariance using only kernel, geometry, and times."""
    started = time.perf_counter()
    c = config["mission"]
    target_h = c["start_h"] + c["budget_s"] / 3600
    events = []
    for point in points:
        times = source.times[tuple(point)]
        for t in times[(times >= c["start_h"]) & (times <= target_h)]:
            events.append([*point, float(t)])
    events = np.asarray(events, dtype=float)
    keys = [event_key(row, c["start_h"]) for row in events]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate event identity in the recorded availability catalog")
    res = config["evaluation"]["resolution_m"]
    axis = np.arange(res / 2, grid.size, res)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.array([grid.is_reachable(p) for p in xy]).reshape(xx.shape)
    target = np.column_stack([xy[mask.ravel()], np.full(mask.sum(), target_h)])
    se, pe = project(gp, events)
    sq, pq = project(gp, target)
    units = gp.fitted.scaler.y_std ** 2
    ce = (kernel(gp, se, se) - pe @ pe.T) * units
    ce = (ce + ce.T) / 2
    cq = (kernel(gp, sq, se) - pq @ pe.T) * units
    vq = (gp.scale - np.sum(pq * pq, axis=1)) * units
    gram = cq.T @ cq / len(target)
    single_mi = 0.5 * np.log1p(np.diag(ce) / gp.noise)
    single_emse = np.diag(gram) / (np.diag(ce) + gp.noise)
    if min(single_mi.max(), single_emse.max(), vq.min()) <= 0:
        raise FloatingPointError("Non-positive initial objective scale or target variance")
    np.savez_compressed(destination, events=events, target=target, event_cov=ce,
                        target_event_cov=cq, target_variance=vq, gram=gram,
                        noise=np.array(gp.noise), start_h=np.array(c["start_h"]),
                        history_count=np.array(len(gp.x)),
                        mi_scale=np.array(single_mi.max()),
                        emse_scale=np.array(single_emse.max()),
                        xx=xx, yy=yy, mask=mask)
    return dict(event_count=len(events), target_count=len(target), target_h=target_h,
                target_resolution_m=res, initial_mean_target_variance_c2=float(vq.mean()),
                mi_scale_nats=float(single_mi.max()), emse_scale_c2=float(single_emse.max()),
                normalization="Maximum initial single-event gain over the fixed availability catalog. Fixed across policies and replans. Route scores may exceed one; never clipped.",
                catalog_precompute_wall_s=time.perf_counter() - started)


def route_components(event_cov, gram, indices, noise):
    """Vectorized MI and average target-variance reduction for equal-size routes."""
    indices = np.asarray(indices, dtype=int)
    if indices.shape[-1] == 0:
        return np.zeros(len(indices)), np.zeros(len(indices))
    covariance = event_cov[indices[:, :, None], indices[:, None, :]]
    grams = gram[indices[:, :, None], indices[:, None, :]]
    mi = information_gain(covariance, noise)
    reduced = np.trace(np.linalg.solve(covariance + noise * np.eye(indices.shape[1]), grams),
                       axis1=-2, axis2=-1)
    if np.any(reduced < -1e-8):
        raise FloatingPointError("Negative EMSE reduction beyond numerical tolerance")
    return mi, np.maximum(reduced, 0.)


class FrozenCatalog:
    def __init__(self, path):
        with np.load(path) as data:
            for name in ("events", "target", "event_cov", "target_event_cov", "target_variance", "gram"):
                setattr(self, name, data[name].copy())
            for name in ("noise", "start_h", "mi_scale", "emse_scale"):
                setattr(self, name, float(data[name]))
            self.history_count = int(data["history_count"])
        self.lookup = {event_key(row, self.start_h): i for i, row in enumerate(self.events)}
        self.version = None
        self.refresh_wall_s = 0.

    def indices(self, query):
        try:
            return [self.lookup[event_key(row, self.start_h)] for row in query]
        except KeyError as error:
            raise ValueError("Scoring requested an event outside the fixed public availability catalog") from error

    def condition(self, gp):
        if self.version == gp.version:
            return
        started = time.perf_counter()
        if not np.isclose(gp.noise, self.noise, rtol=1e-12, atol=1e-12):
            raise ValueError("The fitted observation noise changed")
        scaler = gp.fitted.scaler
        online = gp.x[self.history_count:] * scaler.x_std + scaler.x_mean
        active = self.indices(online)
        if len(active) != gp.version or len(active) != len(set(active)):
            raise ValueError("Online GP observations do not match unique acquired events")
        if not active:
            self.current_cov = self.event_cov
            self.current_gram = self.gram
            self.current_target_variance = self.target_variance
        else:
            factor = cholesky(self.event_cov[np.ix_(active, active)] + self.noise * np.eye(len(active)), lower=True)
            ue = solve_triangular(factor, self.event_cov[:, active].T, lower=True, check_finite=False).T
            uq = solve_triangular(factor, self.target_event_cov[:, active].T, lower=True, check_finite=False).T
            self.current_cov = self.event_cov - ue @ ue.T
            v = self.target_event_cov.T @ uq / len(self.target)
            w = uq.T @ uq / len(self.target)
            self.current_gram = self.gram - v @ ue.T - ue @ v.T + (ue @ w) @ ue.T
            self.current_target_variance = self.target_variance - np.sum(uq * uq, axis=1)
            self.current_cov = (self.current_cov + self.current_cov.T) / 2
            self.current_gram = (self.current_gram + self.current_gram.T) / 2
        self.version = gp.version
        self.refresh_wall_s += time.perf_counter() - started

    def components(self, gp, queries):
        self.condition(gp)
        indices = np.array([self.indices(query) for query in queries], dtype=int)
        return route_components(self.current_cov, self.current_gram, indices, self.noise)

    def score(self, policy, mi, emse, weights):
        if policy == "mi":
            return mi
        if policy == "emse":
            return emse
        if policy != "fusion":
            raise ValueError(policy)
        return weights["mi"] * mi / self.mi_scale + weights["emse_reduction"] * emse / self.emse_scale
