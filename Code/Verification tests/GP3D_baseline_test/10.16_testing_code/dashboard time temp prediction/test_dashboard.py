"""Checks for the real-data split/refit protocol and shared 1D/3D GP machinery."""

import json
from pathlib import Path
import tempfile
import unittest

import gpytorch
import numpy as np
import torch

from dashboard_experiment import ROOT, load_data, split_indices
from experiment_paths import SYNTHETIC_RESULT
from gp_model import SpaceTimeGP, fit_gp, load_gp


class DashboardChecks(unittest.TestCase):
    def test_chronological_refit_and_gap_partitions(self):
        config = json.loads((ROOT / "dashboard_config.json").read_text())
        rows, dates, x, y, audit = load_data(ROOT / config["input_csv"], config["temperature_column"])
        parts = split_indices(x, config)
        self.assertEqual(len(rows), 352)
        self.assertEqual(audit["sampling_interval_min"], [10.0])
        gap, fc = parts["gap"], parts["forecast"]
        self.assertEqual([len(gap[k]) for k in ["train", "validation", "test"]], [316, 12, 24])
        self.assertEqual([len(fc[k]) for k in ["development_train", "validation", "refit_train", "test"]],
                         [216, 72, 288, 64])
        self.assertLess(x[fc["refit_train"]].max(), x[fc["test"]].min())
        self.assertFalse(np.intersect1d(fc["refit_train"], fc["test"]).size)
        np.testing.assert_array_equal(fc["refit_train"], np.r_[fc["development_train"], fc["validation"]])

    def test_one_dimensional_kernel_equals_fixed_location_slice(self):
        t = torch.linspace(0, 2, 10, dtype=torch.float64)[:, None]
        xyz = torch.cat([torch.zeros(10, 2, dtype=torch.float64), t], dim=1)
        y = torch.zeros(10, dtype=torch.float64)
        one = SpaceTimeGP(t, y, gpytorch.likelihoods.GaussianLikelihood(), input_dims=1).double()
        three = SpaceTimeGP(xyz, y, gpytorch.likelihoods.GaussianLikelihood()).double()
        one.covar_module.base_kernel.lengthscale = 0.7
        three.rbf_components()[1].lengthscale = 0.7
        one.covar_module.outputscale = 1.3
        three.covar_module.outputscale = 1.3
        torch.testing.assert_close(one.covar_module(t).to_dense(), three.covar_module(xyz).to_dense())

    def test_fixed_budget_refit_and_temporal_checkpoint(self):
        config = json.loads((ROOT / "dashboard_config.json").read_text())
        settings = {**config["training"], "iterations": 10, "validation_every": 5}
        x = np.linspace(0, 3, 20)[:, None]
        y = np.sin(x[:, 0])
        fitted = fit_gp(x, y, None, None, settings, select_by_validation=False)
        self.assertEqual(fitted.selected_iteration, 10)
        self.assertFalse(any("validation_noisy_nlpd_scaled" in r for r in fitted.history))
        self.assertIn("lengthscale_h", fitted.hyperparameters())
        self.assertNotIn("lengthscales_m_m_h", fitted.hyperparameters())
        query = np.array([[3.1], [3.2]])
        pred = fitted.predict(query)
        self.assertTrue(np.isfinite(pred["mean"]).all())
        np.testing.assert_allclose(pred["observed_var"] - pred["latent_var"],
                                   fitted.hyperparameters()["noise_variance_c2"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "temporal.pt"
            fitted.save(path)
            restored = load_gp(path)
            self.assertEqual(restored.model.input_dims, 1)
            np.testing.assert_allclose(restored.predict(query)["mean"], pred["mean"], atol=1e-10)

    def test_previous_3d_checkpoints_still_load_unchanged(self):
        root = SYNTHETIC_RESULT / "main"
        old = load_gp(root / "model.pt")
        stored = np.load(root / "test_predictions.npz")
        np.testing.assert_allclose(old.predict(stored["xyt"][:10])["mean"],
                                   stored["mean"][:10], atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
