"""Deterministic candidate-exclusion tests; no real sensor failures are generated."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from experiment import ROOT

from mission import MissionController
from failures import CandidateFailures
from gp import GPModel
from interfaces import Measurement, NavigationResult
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from grg import GRGPlanner, Plan

HERE = Path(__file__).resolve().parent
EVIDENCE = {}


class FixedTargetPlanner(GRGPlanner):
    """Test-only planner to make failure, skip and retry calls deterministic."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def plan(self, gp, current, elapsed, remaining, visited=(), blocked=()):
        self.calls.append({"blocked": sorted(blocked), "current": current, "elapsed_s": elapsed})
        if 1 in blocked:
            route = [current, 0, 0] if getattr(self, "alternative", False) else ([current, 0] if current else [0])
        else:
            route = [current, 1, 0]
        _, arrivals, finish = self.schedule(route, elapsed)
        return Plan(route, arrivals, finish, 0., 0., 0, gp.version)


class Sensor:
    def measure(self, position, timestamp_h):
        return Measurement(*position, timestamp_h, 20., "test")


class FailOnceNavigation(PythonNavigationSimulator):
    def __init__(self, grid, status="timeout"):
        super().__init__(grid, .35)
        self.failure_status = status
        self.failed = False
        self.target_attempts = 0

    def navigate_to(self, goal):
        if not np.array_equal(goal, self.grid.home):
            self.target_attempts += 1
            if not self.failed:
                self.failed = True
                return NavigationResult(self.failure_status, self.state, 0., 0., np.empty((0, 2)))
        return super().navigate_to(goal)


class FailureListTests(unittest.TestCase):
    def test_temporary_failure_excludes_exactly_next_planning_call(self):
        failures = CandidateFailures()
        failures.mark_temporary(4, 7, "timeout")
        self.assertEqual(failures.temporary[4].retry_at_plan, 9)
        self.assertIn(4, failures.blocked_for(8))
        self.assertNotIn(4, failures.blocked_for(9))
        self.assertNotIn(4, failures.temporary)

    def test_permanent_requires_static_evidence_and_never_expires(self):
        failures = CandidateFailures()
        for reason in ["timeout", "blocked", "unreachable", "local_failure", "measurement_failed"]:
            with self.assertRaises(ValueError):
                failures.mark_permanent(4, reason)
        for node, reason in [(1, "static_collision"), (2, "static_no_path")]:
            failures.mark_permanent(node, reason)
        self.assertEqual(failures.blocked_for(1000), {1, 2})

    def test_repeated_temporary_failures_are_not_promoted(self):
        failures = CandidateFailures()
        for round_index in range(0, 20, 2):
            failures.mark_temporary(4, round_index, "local_failure")
            self.assertIn(4, failures.blocked_for(round_index + 1))
            self.assertNotIn(4, failures.blocked_for(round_index + 2))
        self.assertEqual(failures.permanent, {})

    def test_static_evidence_supersedes_temporary_and_cannot_be_downgraded(self):
        failures = CandidateFailures()
        failures.mark_temporary(4, 0, "timeout")
        failures.mark_permanent(4, "static_no_path")
        failures.mark_temporary(4, 10, "local_failure")
        self.assertEqual(failures.permanent, {4: "static_no_path"})
        self.assertEqual(failures.temporary, {})

    def test_measurement_failure_hook_has_the_same_one_plan_cooldown(self):
        failures = CandidateFailures()
        failures.measurement_failed(3, 2)
        self.assertIn(3, failures.blocked_for(3))
        self.assertNotIn(3, failures.blocked_for(4))
        self.assertEqual(failures.permanent, {})


class FailureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text())
        cls.grid = GridMap(cls.config["map"])
        cls.points = np.array([[2, 2], [7, 9]], float)
        cls.travel = travel_matrix(PythonNavigationSimulator(cls.grid, .35), cls.points)
        cls.template = GPModel.load(ROOT / "10.18_result/baseline/main/initial_gp.pt")

    def make_mission(self, nav, sensor=None, points=None):
        points = self.points if points is None else points
        travel = np.linalg.norm(points[:, None] - points[None, :], axis=-1) / .35
        planner = FixedTargetPlanner(points, travel, 12, 5)
        config = {**self.config["mission"], "budget_s": 600, "max_iterations": 3}
        return MissionController(copy.deepcopy(self.template), planner, nav, sensor or Sensor(), config)

    def test_transient_navigation_failures_skip_one_call_then_retry(self):
        traces = []
        for status in ["pedestrian_blocked", "local_failure", "timeout", "unreachable", "failed", "blocked"]:
            with self.subTest(status=status):
                nav = FailOnceNavigation(self.grid, status)
                mission = self.make_mission(nav)
                mission.planner.alternative = True
                summary = mission.run()
                self.assertEqual([1 in r["blocked"] for r in mission.planner.calls], [False, True, False])
                self.assertEqual([r["status"] for r in mission.log],
                                 ["navigation_failed", "sampled", "sampled"])
                self.assertEqual(nav.target_attempts, 2)
                self.assertEqual(summary["measurements"], 2)
                self.assertTrue(summary["returned_home"])
                self.assertEqual(mission.failures.permanent, {})
                self.assertEqual(mission.failures.temporary, {})
                self.assertAlmostEqual(summary["mission_time_s"], summary["elapsed_s"], places=8)
                traces.append({"status": status, "calls": mission.planner.calls,
                               "log_statuses": [r["status"] for r in mission.log]})
        EVIDENCE["navigation_cooldowns"] = traces

    def test_confirmed_map_failure_is_permanent_after_navigation_failure(self):
        class ConfirmedNavigation(FailOnceNavigation):
            def static_failure_reason(self, goal):
                if self.failed and not np.array_equal(goal, self.grid.home):
                    return self.confirmed_reason
                return None

        for reason in ["static_collision", "static_no_path"]:
            nav = ConfirmedNavigation(self.grid, "unreachable")
            nav.confirmed_reason = reason
            mission = self.make_mission(nav)
            summary = mission.run()
            self.assertEqual(mission.failures.permanent, {1: reason})
            self.assertEqual(mission.failures.temporary, {})
            self.assertIn(1, mission.failures.blocked_for(100))
            self.assertTrue(summary["returned_home"])
            self.assertEqual(summary["measurements"], 0)

    def test_obstacle_candidate_is_excluded_before_first_plan(self):
        nav = PythonNavigationSimulator(self.grid, .35)
        points = np.array([[2, 2], [10, 9]], float)
        mission = self.make_mission(nav, points=points)
        self.assertEqual(mission.failures.permanent, {1: "static_collision"})
        summary = mission.run()
        self.assertIn(1, mission.planner.calls[0]["blocked"])
        self.assertEqual(summary["measurements"], 0)
        self.assertTrue(summary["returned_home"])

    def test_static_path_absence_is_distinct_from_generic_unreachable(self):
        grid = copy.copy(self.grid)
        original_route = grid.route

        def route(start, goal):
            if np.array_equal(goal, self.points[1]):
                return np.empty((0, 2)), np.inf
            return original_route(start, goal)

        grid.route = route
        nav = PythonNavigationSimulator(grid, .35)
        self.assertEqual(nav.static_failure_reason(self.points[1]), "static_no_path")
        self.assertIsNone(PythonNavigationSimulator(self.grid, .35).static_failure_reason(self.points[1]))
        self.assertIsNone(nav.static_failure_reason([2.1, 2]))

    def test_existing_invalid_measurement_entry_uses_temporary_not_permanent(self):
        class FailOnceSensor(Sensor):
            calls = 0

            def measure(self, position, timestamp_h):
                self.calls += 1
                return Measurement(*position, timestamp_h, 20., "test", "failed" if self.calls == 1 else "ok")

        mission = self.make_mission(PythonNavigationSimulator(self.grid, .35), FailOnceSensor())
        mission.planner.alternative = True
        summary = mission.run()
        self.assertEqual([1 in r["blocked"] for r in mission.planner.calls], [False, True, False])
        self.assertEqual([r["status"] for r in mission.log],
                         ["invalid_measurement", "sampled", "sampled"])
        self.assertEqual(summary["measurements"], 2)
        self.assertEqual(mission.gp.version, 2)
        self.assertEqual(mission.failures.permanent, {})
        self.assertTrue(summary["returned_home"])

    def test_no_alternative_returns_without_artificial_empty_cycles(self):
        mission = self.make_mission(FailOnceNavigation(self.grid))
        summary = mission.run()
        self.assertEqual(len(mission.planner.calls), 2)
        self.assertEqual([r["status"] for r in mission.log],
                         ["navigation_failed", "no_feasible_sampling_route"])
        self.assertEqual(summary["measurements"], 0)
        self.assertEqual(mission.nav.target_attempts, 1)
        self.assertTrue(summary["returned_home"])

    def test_partial_motion_failure_keeps_existing_safe_return_boundary(self):
        class PartialFailure(PythonNavigationSimulator):
            def navigate_to(self, goal):
                if not np.array_equal(goal, self.grid.home):
                    path, _ = self.grid.route(self.state.xy, goal)
                    result = super().navigate_to(path[1])
                    return replace(result, status="local_failure")
                return super().navigate_to(goal)

        mission = self.make_mission(PartialFailure(self.grid, .35))
        summary = mission.run()
        self.assertEqual(summary["stop_reason"], "navigation_failed_after_motion")
        self.assertTrue(summary["returned_home"])
        self.assertEqual(mission.failures.permanent, {})
        self.assertIn(1, mission.failures.temporary)
        self.assertEqual(summary["measurements"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
