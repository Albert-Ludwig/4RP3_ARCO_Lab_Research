"""Correctness checks for migration, posterior semantics and leakage boundaries."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import gpytorch
import numpy as np
import torch

from evaluation import distribution_metrics
from experiment import config_load, independent_splits
from experiment_paths import WORKSPACE_ROOT
from gp_model import SpaceTimeGP, fit_gp, load_gp
from synthetic_field import sample_measurements, temperature_true


class BaselineChecks(unittest.TestCase):
    def test_migrated_class_matches_original_and_three_dimensional_ard(self):
        path = WORKSPACE_ROOT / "GP-Planning_reference/notebooks/06_temporal_nonstationary.ipynb"
        notebook = json.loads(path.read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if "class SpaceTimeGP(" in "".join(c.get("source", [])))
        original_class = next(n for n in ast.parse(source).body
                              if isinstance(n, ast.ClassDef) and n.name == "SpaceTimeGP")
        namespace = {"torch": torch, "gpytorch": gpytorch}
        # Execute only the original model class, never the Notebook data/plot code.
        exec(compile(ast.Module(body=[original_class], type_ignores=[]), str(path), "exec"), namespace)
        torch.manual_seed(5)
        x, y = torch.randn(9, 3, dtype=torch.float64), torch.randn(9, dtype=torch.float64)
        old = namespace["SpaceTimeGP"](x, y, gpytorch.likelihoods.GaussianLikelihood()).double()
        new = SpaceTimeGP(x, y, gpytorch.likelihoods.GaussianLikelihood()).double()
        new.load_state_dict(old.state_dict())
        torch.testing.assert_close(old.covar_module(x).to_dense(), new.covar_module(x).to_dense())
        direct = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3)).double()
        spatial, temporal = new.covar_module.base_kernel.kernels
        spatial.lengthscale = torch.tensor([0.4, 1.2], dtype=torch.float64)
        temporal.lengthscale = 0.7
        direct.base_kernel.lengthscale = torch.tensor([0.4, 1.2, 0.7], dtype=torch.float64)
        direct.outputscale = new.covar_module.outputscale.detach()
        torch.testing.assert_close(new.covar_module(x).to_dense(), direct(x).to_dense())

    def test_generation_reproducible_and_noise_configurable(self):
        bounds = [[0, 30], [0, 30], [0, 24]]
        a = sample_measurements(30, 3, bounds, 0.2)
        b = sample_measurements(30, 3, bounds, 0.2)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.observed, b.observed)
        np.testing.assert_allclose(a.observed - a.truth, 0.2 * a.noise_z)
        np.testing.assert_array_equal(a.with_noise(0).observed, a.truth)
        self.assertGreater(np.ptp(temperature_true([[10, 10, 0], [10, 10, 12]])), 0.1)

    def test_blocked_splits_and_no_shared_coordinates(self):
        config = config_load()
        for scenario in ["main", "spatial", "temporal"]:
            train, val, test = independent_splits(config, scenario, 3)
            x = np.vstack([train.x, val.x, test.x])
            self.assertEqual(len(np.unique(x, axis=0)), len(x))
            if scenario == "spatial":
                self.assertLessEqual(train.x[:, 0].max(), 18)
                self.assertGreaterEqual(test.x[:, 0].min(), 22)
            if scenario == "temporal":
                self.assertLess(train.x[:, 2].max(), val.x[:, 2].min())
                self.assertLess(val.x[:, 2].max(), test.x[:, 2].min())

    def test_posterior_noise_scaling_checkpoint_and_reproducibility(self):
        config = config_load()
        config.update(train_count=30, validation_count=16, test_count=12)
        config["training"].update(iterations=25, validation_every=5)
        train, val, test = independent_splits(config, "main", 12)
        model = fit_gp(train.x, train.observed, val.x, val.observed, config["training"], seed=7)
        np.testing.assert_allclose(model.scaler.x_mean, train.x.mean(axis=0))
        self.assertAlmostEqual(model.scaler.y_mean, train.observed.mean())
        p = model.predict(test.x)
        noise = model.hyperparameters()["noise_variance_c2"]
        np.testing.assert_allclose(p["observed_var"] - p["latent_var"], noise, rtol=1e-8)
        prior_var = model.hyperparameters()["output_variance_c2"]
        self.assertTrue((p["latent_var"] < prior_var).all())
        self.assertTrue((p["latent_var"] > 0).all())
        candidates = [r for r in model.history if "validation_noisy_nlpd_scaled" in r]
        best = min(candidates, key=lambda r: r["validation_noisy_nlpd_scaled"])
        self.assertEqual(model.selected_iteration, best["iteration"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            model.save(path)
            restored = load_gp(path).predict(test.x)
            for key in p:
                np.testing.assert_allclose(p[key], restored[key], atol=1e-10)
        rerun = fit_gp(train.x, train.observed, val.x, val.observed, config["training"], seed=7)
        np.testing.assert_allclose(p["mean"], rerun.predict(test.x)["mean"], atol=1e-10)

    def test_metric_distribution_definition(self):
        m = distribution_metrics(np.zeros(10), np.zeros(10), np.ones(10))
        self.assertEqual(m["rmse_c"], 0)
        self.assertEqual(m["coverage_95"], 1)
        self.assertAlmostEqual(m["nlpd"], 0.5 * np.log(2 * np.pi))
        self.assertAlmostEqual(m["interval_width_95_c"], 3.919927969080108)


if __name__ == "__main__":
    unittest.main(verbosity=2)
