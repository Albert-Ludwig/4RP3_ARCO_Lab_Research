"""Fixed-target synthetic evaluation, isolated from planning decisions."""

import numpy as np

import reuse
from synthetic_field import temperature_true
from evaluation import distribution_metrics


class Evaluation:
    def __init__(self, grid, resolution, target_h):
        axis = np.arange(resolution / 2, grid.size, resolution)
        self.xx, self.yy = np.meshgrid(axis, axis)
        xy = np.column_stack([self.xx.ravel(), self.yy.ravel()])
        self.mask = np.array([grid.is_reachable(p) for p in xy]).reshape(self.xx.shape)
        self.query = np.column_stack([xy[self.mask.ravel()],
                                      np.full(self.mask.sum(), target_h)])
        self.truth = temperature_true(self.query)
        self.rows, self.means, self.variances = [], [], []

    def record(self, gp, elapsed, count):
        pred = gp.predict(self.query)
        row = {"elapsed_s": elapsed, "measurement_count": count,
               **distribution_metrics(self.truth, pred["mean"], pred["latent_var"]),
               "mean_variance_c2": float(pred["latent_var"].mean())}
        self.rows.append(row)
        self.means.append(pred["mean"])
        self.variances.append(pred["latent_var"])
        return {"rmse_c": row["rmse_c"], "mean_variance_c2": row["mean_variance_c2"]}

    def field(self, values):
        field = np.full(self.mask.shape, np.nan)
        field[self.mask] = values
        return field

    def save(self, path):
        np.savez_compressed(path, query=self.query, truth=self.truth,
                            means=np.array(self.means), variances=np.array(self.variances),
                            xx=self.xx, yy=self.yy, mask=self.mask)
