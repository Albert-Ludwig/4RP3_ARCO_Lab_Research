"""Stationary 2D exact GP (GPyTorch) and the fixed/learned fit-and-evaluate driver.

This is the only heavy module (imports torch/gpytorch). The experiment runner uses
it to produce raw metrics and dense-grid predictions; the plotting/table scripts do
NOT import this module — they read the CSVs the runner writes.
"""

import numpy as np
import torch
import gpytorch

from .solvers import FixedKernel


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, constrained=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if constrained:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=2,
                    lengthscale_constraint=gpytorch.constraints.Interval(0.03, 0.6),
                )
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def fit_gp_predict(
    train_points,
    train_values,
    eval_points,
    eval_truth_flat,
    kernel: FixedKernel,
    mode: str = "fixed",
    standardize: bool = False,
    train_iters: int = 320,
    learning_rate: float = 0.02,
    device=None,
):
    """Fit a stationary GP to (train_points, train_values) and evaluate on eval_points.

    mode='fixed' freezes the kernel to ``kernel`` (no training); 'constrained' and
    'learned' optimize the marginal log likelihood (constrained adds interval priors).
    Returns a dict of dense predictions and scalar metrics (RMSE vs the noise-free
    eval truth, mean variance/entropy, calibration, learned hyperparameters).
    """
    device = device or torch.device("cpu")
    train_x = torch.tensor(np.asarray(train_points), dtype=torch.float32, device=device)
    train_y = torch.tensor(np.asarray(train_values), dtype=torch.float32, device=device)
    test_x = torch.tensor(np.asarray(eval_points), dtype=torch.float32, device=device)

    if standardize:
        y_mean = train_y.mean()
        y_std = train_y.std().clamp_min(1e-6)
        train_y_model = (train_y - y_mean) / y_std
    else:
        y_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        y_std = torch.tensor(1.0, dtype=torch.float32, device=device)
        train_y_model = train_y

    constrained = mode == "constrained"
    if constrained:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(1e-4, 0.02)
        ).to(device)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    model = ExactGPModel(train_x, train_y_model, likelihood, constrained=constrained).to(device)

    model.covar_module.base_kernel.lengthscale = kernel.lengthscale
    model.covar_module.outputscale = kernel.outputscale
    likelihood.noise = kernel.noise_variance
    model.mean_module.constant = train_y_model.mean()

    if mode == "fixed":
        model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        model.covar_module.raw_outputscale.requires_grad_(False)
        likelihood.raw_noise.requires_grad_(False)
    else:
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(train_iters):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y_model)
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent = model(test_x)
        mean_s = latent.mean
        var_s = latent.variance
        predictive = likelihood(model(test_x))
        pred_var_s = predictive.variance

    pred_mean = (mean_s * y_std + y_mean).detach().cpu().numpy()
    pred_var = (var_s * (y_std ** 2)).detach().cpu().numpy()
    predictive_var = (pred_var_s * (y_std ** 2)).detach().cpu().numpy()

    pred_std = np.sqrt(np.maximum(pred_var, 1e-12))
    truth = np.asarray(eval_truth_flat)
    rmse = float(np.sqrt(np.mean((pred_mean - truth) ** 2)))
    mean_var = float(np.mean(pred_var))
    mean_entropy = float(np.mean(0.5 * np.log(2.0 * np.pi * np.e * np.maximum(pred_var, 1e-12))))

    abs_error = np.abs(pred_mean - truth)
    corr = float(np.corrcoef(abs_error, pred_std)[0, 1])

    lower = pred_mean - 1.96 * pred_std
    upper = pred_mean + 1.96 * pred_std
    coverage_95 = float(np.mean((truth >= lower) & (truth <= upper)))

    z = (truth - pred_mean) / np.maximum(pred_std, 1e-8)
    z_mean = float(np.mean(z))

    lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    outputscale = float(model.covar_module.outputscale.detach().cpu().item())
    noise_var = float(likelihood.noise.detach().cpu().item())
    mean_constant_scaled = float(model.mean_module.constant.detach().cpu().item())
    mean_constant = float(mean_constant_scaled * y_std.item() + y_mean.item())

    return {
        "mean": pred_mean,
        "var": pred_var,
        "predictive_var": predictive_var,
        "std": pred_std,
        "rmse": rmse,
        "mean_var": mean_var,
        "mean_entropy": mean_entropy,
        "abs_error": abs_error,
        "corr_err_unc": corr,
        "coverage_95": coverage_95,
        "z_mean": z_mean,
        "lengthscale": float(np.mean(lengthscale)),
        "lengthscale_ard": lengthscale,
        "outputscale": outputscale,
        "noise_var": noise_var,
        "mean_constant": mean_constant,
    }
