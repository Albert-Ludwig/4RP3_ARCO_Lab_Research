"""Tests D–F: closed-loop execution, evaluation, and controlled failures."""

import ast
import copy
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from evaluate import Evaluation
from grg import GRGPlanner, Plan
from interfaces import Measurement, NavigationResult
from mission import MissionController
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from test_components import CONFIG, small_gp


class Sensor:
    def __init__(self, mode="ok"):
        self.mode, self.calls = mode, []

    def load_history(self):
        return []

    def measure(self, position, timestamp_h):
        self.calls.append((position.copy(), timestamp_h))
        value = np.nan if self.mode == "invalid" else 20.0
        shift = 1 if self.mode == "wrong_time" else 0
        return Measurement(*position, timestamp_h + shift, value, "test_sensor")


class FailedNavigation(PythonNavigationSimulator):
    def navigate_to(self, goal):
        if not np.array_equal(goal, self.grid.home):
            return NavigationResult("failed", self.state, 0., 0., np.empty((0, 2)))
        return super().navigate_to(goal)


class DelayedNavigation(PythonNavigationSimulator):
    def navigate_to(self, goal):
        result = super().navigate_to(goal)
        if not np.array_equal(goal, self.grid.home) and result.status == "arrived":
            self.wait(2)
            result = replace(result, state=self.state, travel_s=result.travel_s + 2)
        return result


class PartialFailure(PythonNavigationSimulator):
    def navigate_to(self, goal):
        if not np.array_equal(goal, self.grid.home):
            path, _ = self.grid.route(self.state.xy, goal)
            result = super().navigate_to(path[1])
            return replace(result, status="failed")
        return super().navigate_to(goal)


class EmptyPlanner:
    def __init__(self, points):
        self.points, self.allow_revisit = points, False

    def plan(self, gp, current, elapsed, remaining, visited, blocked=()):
        return Plan([], [], elapsed, 0, 0, 0, gp.version)


class MissionFixture:
    @classmethod
    def setUpClass(cls):
        cls.grid = GridMap(CONFIG["map"])
        cls.points = cls.grid.candidates(10, 8)
        nav = PythonNavigationSimulator(cls.grid, .35)
        cls.travel = travel_matrix(nav, cls.points)
        cls.template, _ = small_gp()

    def make_mission(self, budget=600, nav_class=PythonNavigationSimulator, sensor=None):
        gp = copy.deepcopy(self.template)
        c = {**CONFIG["mission"], "budget_s": budget}
        nav = nav_class(self.grid, .35)
        planner = GRGPlanner(self.points, self.travel, 12, 5)
        sensor = sensor or Sensor()
        mission = MissionController(gp, planner, nav, sensor, c)
        return mission


class TestDClosedLoop(MissionFixture, unittest.TestCase):
    def test_full_loop_and_repeated_run(self):
        a, b = self.make_mission(), self.make_mission()
        first, second = a.run(), b.run()
        self.assertGreaterEqual(len(a.online), 3)
        self.assertTrue(first["returned_home"])
        self.assertLessEqual(first["elapsed_s"], 600)
        self.assertEqual(a.gp.version, len(a.online))
        self.assertEqual(len(a.gp.x), len(self.template.x) + len(a.online))
        self.assertEqual([p["gp_version"] for p in a.plans], list(range(len(a.online) + 1)))
        self.assertEqual(a.online, b.online)
        self.assertEqual(a.nav.trajectory, b.nav.trajectory)
        self.assertEqual([p["route"] for p in a.plans], [p["route"] for p in b.plans])
        for record, (xy, timestamp) in zip(a.online, a.sensor.calls):
            np.testing.assert_array_equal([record.x_m, record.y_m], xy)
            self.assertEqual(record.t_h, timestamp)

    def test_actual_navigation_clock_used(self):
        mission = self.make_mission(nav_class=DelayedNavigation)
        result = mission.run()
        rows = [r for r in mission.log if r["gp_updated"]]
        self.assertTrue(result["returned_home"])
        self.assertTrue(rows)
        for row, record in zip(rows, mission.online):
            delay = 0 if np.allclose([record.x_m, record.y_m], self.grid.home) else 2
            self.assertAlmostEqual(row["actual_travel_s"] - row["estimated_travel_s"], delay)
            self.assertAlmostEqual(record.t_h, 12 + row["arrival_s"] / 3600)


class TestEEvaluation(MissionFixture, unittest.TestCase):
    def test_fixed_query_variance_and_latent_coverage(self):
        mission = self.make_mission()
        evaluation = Evaluation(self.grid, 2, 12.25)
        mission.observer = evaluation.record
        mission.run()
        variance = np.array(evaluation.variances)
        self.assertTrue(np.all(np.diff(variance, axis=0) <= 1e-9))
        self.assertTrue(np.all(evaluation.query[:, 2] == 12.25))
        last = evaluation.rows[-1]
        error = evaluation.means[-1] - evaluation.truth
        self.assertAlmostEqual(last["rmse_c"], np.sqrt(np.mean(error**2)))
        expected = np.mean(np.abs(error) <= 1.959963984540054 * np.sqrt(variance[-1]))
        self.assertAlmostEqual(last["coverage_95"], expected)


class TestFInterfaces(MissionFixture, unittest.TestCase):
    def test_failed_navigation_does_not_measure(self):
        mission = self.make_mission(nav_class=FailedNavigation)
        result = mission.run()
        # Successful Home samples are permitted; failed destinations are not.
        self.assertTrue(all(np.array_equal(xy, self.grid.home) for xy, _ in mission.sensor.calls))
        self.assertEqual(mission.gp.version, len(mission.online))
        self.assertTrue(result["returned_home"])
        self.assertTrue(any(r["status"] == "navigation_failed" for r in mission.log))

    def test_invalid_sensor_and_wrong_timestamp(self):
        for mode in ["invalid", "wrong_time"]:
            mission = self.make_mission(sensor=Sensor(mode))
            result = mission.run()
            self.assertTrue(mission.sensor.calls)
            self.assertEqual(mission.gp.version, 0)
            self.assertTrue(result["returned_home"])

    def test_partial_navigation_failure_returns_from_actual_position(self):
        mission = self.make_mission(nav_class=PartialFailure)
        result = mission.run()
        self.assertTrue(all(np.array_equal(xy, self.grid.home) for xy, _ in mission.sensor.calls))
        self.assertGreater(result["travel_distance_m"], 0)
        self.assertEqual(result["stop_reason"], "navigation_failed_after_motion")
        self.assertTrue(result["returned_home"])

    def test_empty_plan_and_insufficient_budget(self):
        mission = self.make_mission()
        mission.planner = EmptyPlanner(self.points)
        self.assertTrue(mission.run()["returned_home"])
        self.assertEqual(mission.sensor.calls, [])
        mission = self.make_mission(budget=1)
        self.assertTrue(mission.run()["returned_home"])
        self.assertEqual(mission.sensor.calls, [])

    def test_empty_navigation_path(self):
        grid = copy.copy(self.grid)
        grid.route = lambda start, goal: (np.empty((0, 2)), np.inf)
        nav = PythonNavigationSimulator(grid, .35)
        self.assertEqual(nav.navigate_to([3, 3]).status, "unreachable")
        self.assertEqual(nav.state.elapsed_s, 0)

    def test_core_modules_have_no_truth_or_ros_import(self):
        for name in ["gp.py", "grg.py", "mission.py"]:
            tree = ast.parse((Path(__file__).parent / name).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any(s in i for i in imports for s in [
                "synthetic_field", "temperature", "evaluate", "navigation", "rclpy", "nav2"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
