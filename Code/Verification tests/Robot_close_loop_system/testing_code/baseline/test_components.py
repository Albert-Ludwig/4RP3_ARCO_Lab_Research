"""Deterministic tests A–C: navigation, GP conditioning, time-aware GRG."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from gp import GPModel
from grg import GRGPlanner, information_gain
from interfaces import Measurement
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from temperature import SyntheticTemperatureSource

CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())


def small_gp():
    rng = np.random.default_rng(8)
    x = rng.uniform([1, 1, 0], [29, 29, 12], (24, 3))
    data = SyntheticTemperatureSource(x, 0, 0, 8).load_history()
    gp = GPModel.train(data, 12, {**CONFIG["gp"], "iterations": 35}, 8)
    return gp, data


class TestAEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = GridMap(CONFIG["map"])
        cls.points = cls.grid.candidates(48, 18)

    def test_map_and_footprint(self):
        g = self.grid
        self.assertEqual(g.size, 30)
        self.assertEqual(len(g.obstacles), 2)
        self.assertTrue(all(p.area <= 10 for p in g.obstacles))
        self.assertFalse(g.obstacles[0].intersects(g.obstacles[1]))
        self.assertAlmostEqual(g.radius, np.hypot(.3, .2) + .1)
        for polygon in g.obstacles:
            self.assertTrue(g.inflated.covers(polygon))
        self.assertFalse(g.is_free([.1, 2]))

    def test_candidates_reachable_repeatable(self):
        np.testing.assert_array_equal(self.points, self.grid.candidates(48, 18))
        self.assertEqual(len(np.unique(self.points, axis=0)), 48)
        for p in self.points:
            self.assertTrue(self.grid.is_free(p))
            path, distance = self.grid.route(self.grid.home, p)
            self.assertTrue(len(path) > 0 and np.isfinite(distance))

    def test_edges_and_no_corner_cutting(self):
        for a, b in self.grid.graph.edges:
            self.assertTrue(self.grid.segment_free(self.grid.xy(a), self.grid.xy(b)))
            if a[0] != b[0] and a[1] != b[1]:
                self.assertIn((a[0], b[1]), self.grid.graph)
                self.assertIn((b[0], a[1]), self.grid.graph)

    def test_motion_cost_and_return(self):
        nav = PythonNavigationSimulator(self.grid, .35)
        target = self.points[3]
        expected = nav.estimate_travel_cost(self.grid.home, target)
        result = nav.navigate_to(target)
        self.assertEqual(result.status, "arrived")
        self.assertAlmostEqual(result.travel_s, expected)
        self.assertAlmostEqual(result.travel_s, result.distance_m / .35)
        self.assertGreater(len(nav.trajectory), len(result.path))
        self.assertTrue(all(self.grid.is_free([r["x_m"], r["y_m"]]) for r in nav.trajectory))
        self.assertEqual(nav.navigate_to(self.grid.home).status, "arrived")
        np.testing.assert_allclose(nav.state.xy, self.grid.home)

    def test_invalid_requests(self):
        nav = PythonNavigationSimulator(self.grid, .35)
        for target in [[10, 9], [-1, 0], [2.1, 2], [float("nan"), 2]]:
            before = nav.get_robot_state()
            self.assertEqual(nav.navigate_to(target).status, "unreachable")
            self.assertEqual(nav.get_robot_state(), before)


class TestBPosterior(unittest.TestCase):
    def setUp(self):
        self.gp, self.data = small_gp()
        self.q = np.array([[8, 9, 12.1], [21, 19, 12.2], [8.1, 9.1, 12.11]])

    def test_joint_matches_conditioning_and_scaled_units(self):
        gp = self.gp
        mean, cov = gp.joint(self.q)
        model, s = gp.fitted.model, gp.fitted.scaler
        x = model.train_inputs[0]
        q = torch.tensor(s.transform_x(self.q), dtype=torch.float64)
        with torch.no_grad():
            a = model.covar_module(x).to_dense().numpy() + gp.noise_scaled * np.eye(len(x))
            b = model.covar_module(q, x).to_dense().numpy()
            c = model.covar_module(q).to_dense().numpy()
            m = float(model.mean_module.constant)
        expected_mean = (m + b @ np.linalg.solve(a, gp.y - m)) * s.y_std + s.y_mean
        expected_cov = (c - b @ np.linalg.solve(a, b.T)) * s.y_std**2
        np.testing.assert_allclose(mean, expected_mean, atol=1e-9)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-9)
        np.testing.assert_allclose(gp.covariance_many(self.q[None])[0], cov, atol=1e-9)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-10)
        self.assertTrue(np.all(np.diag(cov) < np.diag(c) * s.y_std**2))
        pred = gp.predict(self.q)
        np.testing.assert_allclose(pred["observed_var"] - pred["latent_var"], gp.noise)

    def test_append_fixed_parameters_and_variance(self):
        gp = self.gp
        parameters = gp.parameters()
        scaler = copy.deepcopy(gp.fitted.scaler.as_dict())
        old_x = gp.x.copy()
        mean, cov = gp.joint(self.q)
        gp.append(Measurement(*self.q[0], mean[0] + .4, "test"))
        new_mean, new_cov = gp.joint(self.q)
        self.assertGreater(np.max(np.abs(new_mean - mean)), .01)
        self.assertTrue(np.all(np.diag(new_cov) <= np.diag(cov) + 1e-10))
        np.testing.assert_array_equal(gp.x[:-1], old_x)
        self.assertEqual(scaler, gp.fitted.scaler.as_dict())
        for key, value in parameters.items():
            np.testing.assert_array_equal(value, gp.parameters()[key])
        # A second successful update retains the first online measurement.
        gp.append(Measurement(*self.q[1], mean[1], "test"))
        self.assertEqual(len(gp.x), len(old_x) + 2)

    def test_temperature_changes_mean_not_covariance(self):
        a, b = self.gp, copy.deepcopy(self.gp)
        a.append(Measurement(*self.q[0], 18., "test"))
        b.append(Measurement(*self.q[0], 25., "test"))
        ma, ca = a.joint(self.q)
        mb, cb = b.joint(self.q)
        self.assertGreater(np.max(np.abs(ma - mb)), 1)
        np.testing.assert_allclose(ca, cb, atol=1e-12)

    def test_future_history_and_invalid_measurement_rejected(self):
        future = self.data + [Measurement(1, 2, 13, 20, "test")]
        with self.assertRaises(ValueError):
            GPModel.train(future, 12, CONFIG["gp"], 0)
        with self.assertRaises(ValueError):
            self.gp.append(Measurement(1, 2, 12, np.nan, "test"))
        self.assertLessEqual(max(r.t_h for r in self.data), 12)


class TestCGRG(unittest.TestCase):
    def setUp(self):
        self.gp, _ = small_gp()
        self.points = np.array([[2, 2], [7, 9], [20, 24], [28, 8], [8, 28]], float)
        self.travel = np.linalg.norm(self.points[:, None] - self.points[None, :], axis=-1) / .35
        self.planner = GRGPlanner(self.points, self.travel, 12, 5)

    def test_information_gain_correlations(self):
        cov = np.array([[1, .95], [.95, 1]])
        self.assertAlmostEqual(float(information_gain(cov, .1)),
                               .5 * np.linalg.slogdet(np.eye(2) + cov / .1)[1])
        self.assertLess(information_gain(cov, .1), information_gain(np.eye(2), .1))

    def test_route_covariance_is_posterior_and_time_dependent(self):
        route = [0, 1, 2, 0]
        query, times, finish = self.planner.schedule(route, 0)
        self.assertAlmostEqual(times[1], times[0] + 5 + self.travel[1, 2])
        self.assertAlmostEqual(query[1, 2], 12 + times[1] / 3600)
        expected = information_gain(self.gp.joint(query)[1], self.gp.noise)
        self.assertAlmostEqual(self.planner.score_routes([route], self.gp, 0)[0], expected, places=8)
        early = self.planner.score_routes([route], self.gp, 0)[0]
        late = self.planner.score_routes([route], self.gp, 12 * 3600)[0]
        self.assertGreater(abs(late - early), .01)
        slow = GRGPlanner(self.points, self.travel * 200, 12, 1000)
        forward, backward = slow.score_routes([route, [0, 2, 1, 0]], self.gp, 0)
        self.assertGreater(abs(forward - backward), 1e-5)

    def test_recursive_search_budget_and_updated_posterior(self):
        plan = self.planner.plan(self.gp, 0, 0, 900, set())
        self.assertEqual(len(plan.route[1:-1]), 3)
        self.assertLessEqual(plan.finish_s, 900)
        self.assertEqual(plan.route[-1], 0)
        query = self.planner.schedule(plan.route, 0)[0]
        score = self.planner.score_routes([plan.route], self.gp, 0)[0]
        self.gp.append(Measurement(*query[0], 20, "test"))
        self.assertLess(self.planner.score_routes([plan.route], self.gp, 0)[0], score)
        self.assertEqual(self.planner.plan(self.gp, 0, 0, 900, set()).gp_version, 1)
        tight = self.planner.plan(self.gp, 0, 0, 1, set())
        self.assertEqual(tight.route, [0])
        expensive = self.travel.copy()
        expensive[1:, :] = 1e6
        expensive[:, 1:] = 1e6
        other = GRGPlanner(self.points, expensive, 12, 5)
        self.assertEqual(other.plan(self.gp, 0, 0, 900, set(), blocked={0}).route, [0])

    def test_visited_nodes_remain_eligible(self):
        visited = set(range(1, len(self.points)))
        self.assertTrue(self.planner.allow_revisit)
        self.assertGreater(len(self.planner.plan(self.gp, 0, 0, 900, visited, blocked={0}).route), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
