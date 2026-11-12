"""Online conditioning around the unchanged 10.16 GPyTorch model."""

import time

import gpytorch
import numpy as np
from scipy.linalg import cholesky, solve_triangular
import torch

import reuse
from gp_model import fit_gp, load_gp


def arrays(records):
    if not records or not all(r.valid() for r in records):
        raise ValueError("Require valid measurements")
    return (np.array([[r.x_m, r.y_m, r.t_h] for r in records]),
            np.array([r.temperature_c for r in records]))


class GPModel:
    def __init__(self, fitted):
        self.fitted = fitted
        self.query_s = 0.0
        self.query_batches = 0
        self.version = 0
        self.refresh()

    @classmethod
    def train(cls, records, start_h, settings, seed):
        x, y = arrays(records)
        if np.any(x[:, 2] > start_h):
            raise ValueError("Historical data contain future observations")
        fitted = fit_gp(x, y, None, None, settings, seed, select_by_validation=False)
        fitted.selection_method = "fixed 300-step budget from config; historical data only".replace(
            "300", str(settings["iterations"]))
        return cls(fitted)

    def refresh(self):
        model = self.fitted.model
        self.x = model.train_inputs[0].detach().numpy().copy()
        self.y = model.train_targets.detach().numpy().copy()
        self.length = np.concatenate([k.lengthscale.detach().numpy().ravel()
                                      for k in model.rbf_components()])
        self.scale = float(model.covar_module.outputscale.detach())
        self.noise_scaled = float(self.fitted.likelihood.noise.detach())
        self.noise = self.noise_scaled * self.fitted.scaler.y_std**2
        with torch.no_grad():
            kernel = model.covar_module(model.train_inputs[0]).to_dense().numpy()
        self.factor = cholesky(kernel + self.noise_scaled * np.eye(len(self.x)), lower=True)

    def joint(self, x):
        """Authoritative GPyTorch latent posterior, in °C and °C²."""
        q = torch.as_tensor(self.fitted.scaler.transform_x(x), dtype=torch.float64)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False), \
                gpytorch.settings.max_cholesky_size(10000):
            posterior = self.fitted.model(q)
            s = self.fitted.scaler
            return (posterior.mean.numpy() * s.y_std + s.y_mean,
                    posterior.covariance_matrix.numpy() * s.y_std**2)

    def covariance_many(self, x):
        """Exact batched covariance for short routes, shape (batch, points, 3).

        Uses the same fitted kernel, training data, noise and scaler. This avoids
        constructing thousands of small GPyTorch distribution objects in GRG.
        No approximation, extra fit, explicit inverse, or ground truth is used.
        """
        started = time.perf_counter()
        q = self.fitted.scaler.transform_x(np.asarray(x))
        batch, count, _ = q.shape
        flat = q.reshape(-1, 3)
        delta = (flat[:, None, :] - self.x[None, :, :]) / self.length
        cross = self.scale * np.exp(-0.5 * np.sum(delta**2, axis=-1))
        projected = solve_triangular(self.factor, cross.T, lower=True,
                                     check_finite=False).T.reshape(batch, count, -1)
        delta = (q[:, :, None, :] - q[:, None, :, :]) / self.length
        prior = self.scale * np.exp(-0.5 * np.sum(delta**2, axis=-1))
        cov = prior - projected @ np.swapaxes(projected, 1, 2)
        self.query_s += time.perf_counter() - started
        self.query_batches += 1
        return (cov + np.swapaxes(cov, 1, 2)) * (0.5 * self.fitted.scaler.y_std**2)

    def predict(self, x):
        return self.fitted.predict(x)

    def append(self, record):
        x, y = arrays([record])
        s, model = self.fitted.scaler, self.fitted.model
        tx = torch.as_tensor(s.transform_x(x), dtype=torch.float64)
        ty = torch.as_tensor(s.transform_y(y), dtype=torch.float64)
        model.set_train_data(torch.cat([model.train_inputs[0], tx]),
                             torch.cat([model.train_targets, ty]), strict=False)
        model.eval()
        self.version += 1
        self.refresh()

    def parameters(self):
        return {name: p.detach().numpy().copy() for name, p in self.fitted.model.named_parameters()}

    def save(self, path):
        self.fitted.save(path)

    @classmethod
    def load(cls, path):
        return cls(load_gp(path))
