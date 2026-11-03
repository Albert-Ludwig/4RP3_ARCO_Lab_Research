"""Analytic fixtures that distinguish the objectives and test normalization."""

import unittest
import bootstrap
import numpy as np
from objectives import FrozenCatalog, route_components


class ObjectiveChecks(unittest.TestCase):
    def setUp(self):
        self.noise = .2
        self.cov = np.diag([4., 1.])
        self.cross = np.array([[.1, .9]])
        self.gram = self.cross.T @ self.cross
        self.catalog = object.__new__(FrozenCatalog)
        mi, emse = route_components(self.cov, self.gram, [[0], [1]], self.noise)
        self.mi, self.emse = mi, emse
        self.catalog.mi_scale, self.catalog.emse_scale = mi.max(), emse.max()

    def test_objectives_prefer_different_locations(self):
        self.assertEqual(int(self.mi.argmax()), 0)
        self.assertEqual(int(self.emse.argmax()), 1)

    def test_single_event_closed_forms(self):
        np.testing.assert_allclose(self.mi, .5 * np.log1p(np.diag(self.cov) / self.noise))
        np.testing.assert_allclose(self.emse, [.01 / 4.2, .81 / 1.2])

    def test_joint_reduction_matches_conditioned_target_variance(self):
        _, reduction = route_components(self.cov, self.gram, [[0, 1]], self.noise)
        posterior = 1. - self.cross @ np.linalg.solve(self.cov + self.noise * np.eye(2), self.cross.T)
        self.assertAlmostEqual(float(reduction[0]), 1. - float(posterior[0, 0]))

    def test_fusion_50_50_and_extreme_weights(self):
        fused = self.catalog.score("fusion", self.mi, self.emse, {"mi": .5, "emse_reduction": .5})
        np.testing.assert_allclose(fused, .5 * self.mi / self.mi.max() + .5 * self.emse / self.emse.max())
        self.assertEqual(int(fused.argmax()), 1)
        self.assertEqual(int(self.catalog.score("fusion", self.mi, self.emse, {"mi": 1., "emse_reduction": 0.}).argmax()), 0)
        self.assertEqual(int(self.catalog.score("fusion", self.mi, self.emse, {"mi": 0., "emse_reduction": 1.}).argmax()), 1)

    def test_temperature_unit_rescaling_leaves_normalized_fusion_unchanged(self):
        scale = 100.
        mi, emse = route_components(self.cov * scale, self.gram * scale ** 2, [[0], [1]], self.noise * scale)
        np.testing.assert_allclose(mi, self.mi)
        np.testing.assert_allclose(emse / emse.max(), self.emse / self.emse.max())

    def test_empty_route_has_zero_gain(self):
        mi, emse = route_components(self.cov, self.gram, np.empty((2, 0), dtype=int), self.noise)
        np.testing.assert_array_equal(mi, [0., 0.])
        np.testing.assert_array_equal(emse, [0., 0.])

    def test_joint_mi_accounts_for_correlated_observations(self):
        cov = np.array([[1., .99], [.99, 1.]])
        joint, _ = route_components(cov, np.eye(2), [[0, 1]], self.noise)
        individual, _ = route_components(cov, np.eye(2), [[0], [1]], self.noise)
        self.assertLess(joint[0], individual.sum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
