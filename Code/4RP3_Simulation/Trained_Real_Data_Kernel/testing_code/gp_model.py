"""Extracted/adapted from GP-Planning_reference Notebook 06, SpaceTimeGP.

The existing ConstantMean + ScaleKernel(spatial ARD RBF * temporal RBF)
and ExactGP/Adam/ExactMarginalLogLikelihood workflow are retained. The product
is algebraically a 3D ARD RBF. Added: train-only scaling, validation checkpoint
selection, explicit latent/noisy posteriors, CPU float64, batching and persistence.
The one-dimensional temporal slice supports single-station real measurements.
This module receives measurements only; it has no access to synthetic truth.
"""

from dataclasses import dataclass
import copy
import time

import gpytorch
import numpy as np
import torch


class SpaceTimeGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dims=3):
        super().__init__(train_x, train_y, likelihood)
        self.input_dims = input_dims
        self.mean_module = gpytorch.means.ConstantMean()
        if input_dims == 3:
            k_sp = gpytorch.kernels.RBFKernel(
                ard_num_dims=2, active_dims=torch.tensor([0, 1]))
            k_tm = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([2]))
            self.covar_module = gpytorch.kernels.ScaleKernel(k_sp * k_tm)
        elif input_dims == 1:
            # Single-station data identify only time, not spatial lengthscales.
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=1))
        else:
            raise ValueError("Supported inputs are t (1D) or x,y,t (3D)")

    def rbf_components(self):
        kernel = self.covar_module.base_kernel
        return list(kernel.kernels) if self.input_dims == 3 else [kernel]

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


@dataclass
class TrainScaler:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, train_x, train_y):
        return cls(train_x.mean(axis=0), train_x.std(axis=0).clip(1e-8),
                   float(train_y.mean()), max(float(train_y.std()), 1e-8))

    def transform_x(self, x):
        return (np.asarray(x) - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (np.asarray(y) - self.y_mean) / self.y_std

    def as_dict(self):
        return {"x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                "y_mean": self.y_mean, "y_std": self.y_std,
                "fit_source": "training measurements only"}


@dataclass
class FittedGP:
    model: SpaceTimeGP
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    scaler: TrainScaler
    history: list
    selected_iteration: int
    fit_seconds: float
    selection_method: str = "minimum validation noisy marginal NLPD; no test selection"

    def predict(self, query_x, batch_size=256):
        """Conditioned posterior, marginal moments in original degrees Celsius.

        Batch queries avoid constructing huge full grid covariances. No approximate
        fast predictive variances: use exact Cholesky for these small experiments.
        """
        self.model.eval()
        self.likelihood.eval()
        query = self.scaler.transform_x(query_x)
        means, variances, noisy_variances = [], [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False), \
                gpytorch.settings.max_cholesky_size(10000):
            for start in range(0, len(query), batch_size):
                x = torch.as_tensor(query[start:start + batch_size], dtype=torch.float64)
                latent = self.model(x)
                observed = self.likelihood(latent)
                means.append(latent.mean.cpu().numpy())
                variances.append(latent.variance.cpu().numpy())
                noisy_variances.append(observed.variance.cpu().numpy())
        return {
            "mean": np.concatenate(means) * self.scaler.y_std + self.scaler.y_mean,
            "latent_var": np.concatenate(variances).clip(1e-12) * self.scaler.y_std**2,
            "observed_var": np.concatenate(noisy_variances).clip(1e-12) * self.scaler.y_std**2,
        }

    def hyperparameters(self):
        ls = np.concatenate([k.lengthscale.detach().numpy().ravel()
                             for k in self.model.rbf_components()])
        result = {
            "kernel": ("ScaleKernel(RBF_ARD_xy * RBF_t) == 3D ARD RBF"
                       if self.model.input_dims == 3 else "ScaleKernel(RBF_t), single-location temporal GP"),
            "lengthscales_scaled": ls.tolist(),
            "output_variance_c2": float(self.model.covar_module.outputscale.detach())
                                  * self.scaler.y_std**2,
            "noise_variance_c2": float(self.likelihood.noise.detach()) * self.scaler.y_std**2,
            "mean_c": float(self.model.mean_module.constant.detach())
                      * self.scaler.y_std + self.scaler.y_mean,
            "selected_iteration": self.selected_iteration,
            "selection": self.selection_method,
            "fit_seconds": self.fit_seconds,
            "scaler": self.scaler.as_dict(),
        }
        if self.model.input_dims == 3:
            result["lengthscales_m_m_h"] = (ls * self.scaler.x_std).tolist()
        else:
            result["lengthscale_h"] = float(ls[0] * self.scaler.x_std[0])
        return result

    def save(self, path):
        torch.save({"model_state": self.model.state_dict(),
                    "train_x_scaled": self.model.train_inputs[0],
                    "train_y_scaled": self.model.train_targets,
                    "scaler": self.scaler.as_dict(),
                    "selected_iteration": self.selected_iteration,
                    "fit_seconds": self.fit_seconds,
                    "input_dims": self.model.input_dims,
                    "selection_method": self.selection_method}, path)


def fit_gp(train_x, train_y, validation_x, validation_y, settings, seed=0,
           select_by_validation=True):
    """Fit using noisy training labels; validation labels select checkpoints only.

    All inputs are arrays without ground-truth fields. Test data are not accepted.
    Hyperparameters (including noise) are learned; generating noise is not supplied.
    With select_by_validation=False, use a previously fixed iteration budget and
    do not inspect validation labels (e.g. refit before a held-out forecast).
    """
    train_x, train_y = np.asarray(train_x), np.asarray(train_y)
    if train_x.ndim != 2 or train_x.shape[1] not in (1, 3) or train_y.shape != (len(train_x),):
        raise ValueError("Require (N,1) or (N,3) inputs and (N,) labels")
    arrays = [train_x, train_y]
    if select_by_validation:
        if validation_x is None or validation_y is None:
            raise ValueError("Checkpoint selection requires a separate validation set")
        arrays.extend([validation_x, validation_y])
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Require finite measurements")
    input_dims = train_x.shape[1]
    torch.manual_seed(seed)
    torch.set_num_threads(settings["threads"])
    scaler = TrainScaler.fit(train_x, train_y)
    tx = torch.as_tensor(scaler.transform_x(train_x), dtype=torch.float64)
    ty = torch.as_tensor(scaler.transform_y(train_y), dtype=torch.float64)
    if select_by_validation:
        vx = torch.as_tensor(scaler.transform_x(validation_x), dtype=torch.float64)
        vy = torch.as_tensor(scaler.transform_y(validation_y), dtype=torch.float64)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = SpaceTimeGP(tx, ty, likelihood, input_dims=input_dims).double()
    for k in model.rbf_components():
        k.lengthscale = settings["initial_lengthscale_scaled"]
    model.covar_module.outputscale = 1.0
    likelihood.noise = settings["initial_noise_variance_scaled"]
    model.mean_module.constant = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    history, best_score, best_state, best_iteration = [], np.inf, None, 0
    started = time.perf_counter()
    with gpytorch.settings.max_cholesky_size(10000):
        for iteration in range(1, settings["iterations"] + 1):
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            loss = -mll(model(tx), ty)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GP marginal likelihood")
            loss.backward()
            optimizer.step()
            row = {"iteration": iteration, "negative_mll_scaled": float(loss.detach())}
            if select_by_validation and (iteration == 1
                    or iteration % settings["validation_every"] == 0
                    or iteration == settings["iterations"]):
                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
                    pv = likelihood(model(vx))
                    var = pv.variance.clamp_min(1e-12)
                    score = float((0.5 * (torch.log(2 * torch.pi * var)
                                      + (vy - pv.mean)**2 / var)).mean())
                row["validation_noisy_nlpd_scaled"] = score
                if np.isfinite(score) and score < best_score:
                    best_score, best_iteration = score, iteration
                    best_state = copy.deepcopy(model.state_dict())
            history.append(row)
    if not select_by_validation:
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = settings["iterations"]
    if best_state is None:
        raise FloatingPointError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.train()  # invalidate any cached posterior from the last checkpoint
    model.eval()
    likelihood.eval()
    selection = ("minimum validation noisy marginal NLPD; no test selection"
                 if select_by_validation else "fixed iteration count selected before refit; no test selection")
    return FittedGP(model, likelihood, scaler, history, best_iteration,
                    time.perf_counter() - started, selection)


def load_gp(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = SpaceTimeGP(checkpoint["train_x_scaled"], checkpoint["train_y_scaled"],
                        likelihood, input_dims=checkpoint.get("input_dims", 3)).double()
    model.load_state_dict(checkpoint["model_state"])
    s = checkpoint["scaler"]
    scaler = TrainScaler(np.array(s["x_mean"]), np.array(s["x_std"]), s["y_mean"], s["y_std"])
    model.eval()
    likelihood.eval()
    return FittedGP(model, likelihood, scaler, [], checkpoint["selected_iteration"],
                    checkpoint["fit_seconds"], checkpoint.get("selection_method",
                    "minimum validation noisy marginal NLPD; no test selection"))
