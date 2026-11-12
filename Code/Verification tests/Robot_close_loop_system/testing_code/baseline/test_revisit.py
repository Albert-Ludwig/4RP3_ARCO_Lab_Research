"""Verify the authorized revisit policy without replacing the GP or reward."""

import copy
import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import textwrap
import unittest

import numpy as np
import torch

import experiment
from experiment import ROOT
from grg import GRGPlanner

from gp import GPModel
from grg import information_gain
from interfaces import Measurement
from mission import MissionController
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from timing import MissionTime

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
EVIDENCE = {}


class Sensor:
    def measure(self, position, timestamp_h):
        return Measurement(*position, timestamp_h, 20., "test")


class RevisitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "config.json").read_text())
        cls.grid = GridMap(cls.config["map"])
        cls.points = np.array([[2, 2], [7, 9], [20, 24]], float)
        cls.travel = travel_matrix(PythonNavigationSimulator(cls.grid, .35), cls.points)
        cls.template = GPModel.load(ROOT / "10.18_result/baseline/main/initial_gp.pt")

    def setUp(self):
        self.gp = copy.deepcopy(self.template)
        self.planner = GRGPlanner(self.points, self.travel, 12, 5, depth=2)

    def test_matching_configuration_and_original_components(self):
        old = json.loads((ROOT / "10.18_result/baseline/main/config.json").read_text())
        current = copy.deepcopy(self.config)
        self.assertTrue(current["mission"].pop("allow_revisit"))
        old["mission"].pop("allow_revisit")
        self.assertEqual(current, old)
        self.assertIs(experiment.GRGPlanner, GRGPlanner)
        self.assertIs(experiment.GPModel, GPModel)
        self.assertIs(experiment.MissionController, MissionController)

    def test_previously_sampled_nodes_remain_eligible(self):
        plan = self.planner.plan(self.gp, 0, 0, 900, visited={1, 2}, blocked={0})
        self.assertTrue(self.planner.allow_revisit)
        self.assertGreater(len(plan.arrivals_s), 0)
        self.assertTrue(set(plan.route[1:-1]) <= {1, 2})

    def test_prefix_nodes_can_be_sampled_at_later_times(self):
        plan = self.planner.plan(self.gp, 0, 0, 900, visited={1, 2}, blocked={0})
        samples = plan.route[1:-1]
        self.assertLess(len(set(samples)), len(samples))
        query, arrivals, _ = self.planner.schedule(plan.route, 0)
        self.assertEqual(len(query), len(samples))
        self.assertTrue(np.all(np.diff(arrivals) > 0))
        for node in set(samples):
            times = [t for v, t in zip(samples, arrivals) if v == node]
            self.assertEqual(len(times), len(set(times)))
        EVIDENCE["route"] = {"route": plan.route, "arrivals_s": arrivals,
                             "finish_s": plan.finish_s, "gain_nats": plan.gain}

    def test_joint_reward_keeps_every_spatiotemporal_observation(self):
        route = [0, 1, 2, 1, 0]
        query, arrivals, finish = self.planner.schedule(route, 43)
        self.assertEqual(query.shape, (3, 3))
        self.assertTrue(np.array_equal(query[0, :2], query[2, :2]))
        self.assertLess(query[0, 2], query[2, 2])
        mean, covariance = self.gp.joint(query)
        expected = float(information_gain(covariance, self.gp.noise))
        actual = self.planner.score_routes([route], self.gp, 43)[0]
        self.assertAlmostEqual(actual, expected, places=8)
        elapsed = 43.
        for i, (a, b) in enumerate(zip(route, route[1:])):
            elapsed += self.travel[a, b]
            if b != 0:
                self.assertAlmostEqual(arrivals[i], elapsed)
                elapsed += 5
        self.assertAlmostEqual(finish, elapsed)

    def test_blocked_unreachable_and_budget_constraints_remain(self):
        blocked = self.planner.plan(self.gp, 0, 0, 900, (), blocked={0, 1, 2})
        self.assertEqual(blocked.route, [0])
        tight = self.planner.plan(self.gp, 0, 0, 1, ())
        self.assertEqual(tight.route, [0])
        travel = self.travel.copy()
        travel[2, :] = np.inf
        travel[:, 2] = np.inf
        travel[2, 2] = 0
        planner = GRGPlanner(self.points, travel, 12, 5)
        plan = planner.plan(self.gp, 0, 0, 900, {1}, blocked={0})
        self.assertTrue(plan.arrivals_s)
        self.assertEqual(set(plan.route[1:-1]), {1})
        self.assertNotIn(2, plan.route)
        self.assertTrue(np.isfinite(plan.finish_s))

    def test_revisited_measurement_updates_posterior_without_exclusion(self):
        before = self.planner.plan(self.gp, 0, 0, 900, (), blocked={0, 2})
        q = self.planner.schedule(before.route, 0)[0][0]
        self.gp.append(Measurement(*q, 20., "test"))
        after = self.planner.plan(self.gp, 0, 0, 900, {1}, blocked={0, 2})
        self.assertEqual(after.route, before.route)
        self.assertLess(after.gain, before.gain)
        self.assertEqual(after.gp_version, 1)
        self.assertEqual(len(self.gp.x), 101)

    def test_full_loop_assimilates_actual_revisits_and_returns(self):
        parameters = self.gp.parameters()
        nav = PythonNavigationSimulator(self.grid, .35)
        config = {**self.config["mission"], "budget_s": 600}
        mission = MissionController(self.gp, self.planner, nav, Sensor(), config)
        summary = mission.run()
        positions = [(float(m.x_m), float(m.y_m)) for m in mission.online]
        self.assertGreater(len(positions), len(set(positions)))
        self.assertTrue(summary["returned_home"])
        self.assertLessEqual(summary["elapsed_s"], 600)
        self.assertEqual(self.gp.version, len(mission.online))
        self.assertEqual(len(self.gp.x), 100 + len(mission.online))
        self.assertEqual(summary["online_computation_time_s"], 0.)
        self.assertAlmostEqual(summary["sampling_time_s"], 5 * len(mission.online))
        self.assertAlmostEqual(summary["mission_time_s"], summary["travel_time_s"] +
                               summary["sampling_time_s"] + summary["online_computation_time_s"])
        self.assertAlmostEqual(summary["mission_time_s"], summary["elapsed_s"], places=8)
        self.assertAlmostEqual(summary["time_breakdown"]["total_s"], summary["elapsed_s"], places=8)
        for row in mission.log:
            self.assertEqual(row["online_computation_time_s"], 0.)
            self.assertAlmostEqual(row["time_breakdown_start"]["total_s"], row["elapsed_s"], places=8)
        self.assertTrue(np.all(np.diff([m.t_h for m in mission.online]) > 0))
        for name, value in parameters.items():
            np.testing.assert_array_equal(value, self.gp.parameters()[name])
        self.assertTrue(all(self.grid.segment_free([a["x_m"], a["y_m"]], [b["x_m"], b["y_m"]])
                            for a, b in zip(nav.trajectory, nav.trajectory[1:])))
        EVIDENCE["mission"] = {"summary": summary,
            "measurements": [vars(m) for m in mission.online],
            "new_observations": len(positions), "unique_spatial_positions": len(set(positions))}

    def test_experiment_uses_consolidated_modules(self):
        self.assertIs(experiment.GRGPlanner, GRGPlanner)
        self.assertIs(experiment.MissionController, MissionController)
        self.assertIs(experiment.PythonNavigationSimulator, PythonNavigationSimulator)
        self.assertEqual(GRGPlanner.__module__, "grg")
        self.assertFalse((HERE.parent / "revisit_enabled").exists())

    def test_home_is_a_planned_measurement_even_at_start(self):
        plan = self.planner.plan(self.gp, 0, 0, 5, {0}, blocked={1, 2})
        self.assertEqual(plan.route, [0, 0, 0])
        self.assertEqual(plan.arrivals_s, [0.])
        self.assertAlmostEqual(plan.finish_s, 5)
        query = self.planner.schedule(plan.route, 0)[0]
        self.assertEqual(query.shape, (1, 3))
        np.testing.assert_allclose(query[0], [2, 2, 12])
        self.assertAlmostEqual(plan.gain, float(information_gain(self.gp.joint(query)[1], self.gp.noise)), places=8)
        tight = self.planner.plan(self.gp, 0, 0, 5 - 1e-5, (), blocked={1, 2})
        self.assertEqual(tight.route, [0])

    def test_intermediate_home_and_final_return_have_distinct_roles(self):
        route = [0, 1, 0, 2, 0]
        query, arrivals, finish = self.planner.schedule(route, 19)
        self.assertEqual(len(query), 3)
        np.testing.assert_allclose(query[1, :2], self.points[0])
        self.assertAlmostEqual(arrivals[1], 19 + self.travel[0, 1] + 5 + self.travel[1, 0])
        self.assertAlmostEqual(finish, 19 + sum(self.travel[a, b] for a, b in zip(route, route[1:])) + 15)
        expected = float(information_gain(self.gp.joint(query)[1], self.gp.noise))
        self.assertAlmostEqual(self.planner.score_routes([route], self.gp, 19)[0], expected, places=8)
        self.assertEqual(len(self.planner.schedule([0, 1, 0], 0)[0]), 1)
        self.assertEqual(len(self.planner.schedule([0, 1, 0], 0, sample_last=True)[0]), 2)

    def test_home_sampling_after_travel_reserves_dwell_and_return(self):
        back = self.travel[1, 0]
        direct = self.planner.plan(self.gp, 1, 47, back, {0, 1}, blocked={1, 2})
        sample = self.planner.plan(self.gp, 1, 47, back + 5, {0, 1}, blocked={1, 2})
        self.assertEqual(direct.route, [1, 0])
        self.assertEqual(direct.arrivals_s, [])
        self.assertEqual(sample.route, [1, 0, 0])
        self.assertAlmostEqual(sample.arrivals_s[0], 47 + back)
        self.assertAlmostEqual(sample.finish_s, 47 + back + 5)

    def test_home_only_closed_loop_updates_gp_without_extra_final_sample(self):
        points = self.points[:1]
        planner = GRGPlanner(points, np.zeros((1, 1)), 12, 5, depth=2)
        nav = PythonNavigationSimulator(self.grid, .35)
        config = {**self.config["mission"], "budget_s": 10}
        mission = MissionController(self.gp, planner, nav, Sensor(), config)
        summary = mission.run()
        self.assertEqual(len(mission.online), 2)
        self.assertEqual(self.gp.version, 2)
        self.assertEqual(len(self.gp.x), 102)
        self.assertTrue(summary["returned_home"])
        self.assertAlmostEqual(summary["elapsed_s"], 10)
        self.assertEqual(summary["time_breakdown"], {"travel_s": 0., "sampling_s": 10.,
                                                  "online_computation_s": 0., "total_s": 10.})
        np.testing.assert_allclose([m.t_h for m in mission.online], [12, 12 + 5 / 3600])
        self.assertTrue(all(0 in row["candidates"] for row in mission.log))
        EVIDENCE["home_only"] = {"summary": summary, "measurements": [vars(m) for m in mission.online]}

    def test_recursive_home_timing_and_joint_reward(self):
        for depth in [1, 2, 3]:
            for current, elapsed, remaining in [(0, 0, 30), (1, 55, 250)]:
                planner = GRGPlanner(self.points, self.travel, 12, 5, depth=depth)
                plan = planner.plan(self.gp, current, elapsed, remaining, {0, 1})
                query, arrivals, finish = planner.schedule(plan.route, elapsed)
                clock = float(elapsed)
                expected = []
                for i in range(1, len(plan.route)):
                    clock += self.travel[plan.route[i - 1], plan.route[i]]
                    if i < len(plan.route) - 1:
                        expected.append([*self.points[plan.route[i]], 12 + clock / 3600])
                        clock += 5
                np.testing.assert_allclose(query, np.asarray(expected).reshape(-1, 3))
                self.assertAlmostEqual(finish, clock)
                self.assertLessEqual(finish, elapsed + remaining + 1e-8)
                reward = float(information_gain(self.gp.joint(query)[1], self.gp.noise)) if len(query) else 0
                self.assertAlmostEqual(plan.gain, reward, places=8)

    def test_pure_return_does_not_force_a_home_measurement(self):
        from grg import Plan

        class OneSamplePlanner(GRGPlanner):
            def plan(self, gp, current, elapsed, remaining, visited=(), blocked=()):
                route = [current, 1, 0] if not visited else [current, 0]
                _, arrivals, finish = self.schedule(route, elapsed)
                return Plan(route, arrivals, finish, 0., 0., 0, gp.version)

        planner = OneSamplePlanner(self.points, self.travel, 12, 5)
        mission = MissionController(self.gp, planner, PythonNavigationSimulator(self.grid, .35),
                                    Sensor(), {**self.config["mission"], "budget_s": 600})
        summary = mission.run()
        self.assertTrue(summary["returned_home"])
        self.assertEqual(len(mission.online), 1)
        np.testing.assert_allclose([mission.online[0].x_m, mission.online[0].y_m], self.points[1])

    def test_home_endpoint_roles_match_with_reward_cache_disabled(self):
        class NoReuseDict(dict):
            def __contains__(self, key):
                return False

        replacements = []

        class ReplaceCache(ast.NodeTransformer):
            def visit_Assign(self, node):
                if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "cache" and isinstance(node.value, ast.Dict)):
                    node.value = ast.Call(func=ast.Name(id="NoReuseDict", ctx=ast.Load()), args=[], keywords=[])
                    replacements.append(True)
                return self.generic_visit(node)

        tree = ast.parse(textwrap.dedent(inspect.getsource(GRGPlanner.plan)))
        tree = ast.fix_missing_locations(ReplaceCache().visit(tree))
        self.assertEqual(len(replacements), 1)
        scope = {**GRGPlanner.plan.__globals__, "NoReuseDict": NoReuseDict}
        exec(compile(tree, "<home-cache-verification-only>", "exec"), scope)
        for current, elapsed, remaining, blocked in [(0, 0, 30, {1, 2}), (0, 0, 900, ()),
                                                      (1, 80, 500, ())]:
            a = self.planner.plan(self.gp, current, elapsed, remaining, {0, 1}, blocked)
            b = scope["plan"](self.planner, self.gp, current, elapsed, remaining, {0, 1}, blocked)
            self.assertEqual(a.route, b.route)
            np.testing.assert_allclose(a.arrivals_s, b.arrivals_s)
            self.assertAlmostEqual(a.finish_s, b.finish_s)
            self.assertAlmostEqual(a.gain, b.gain, places=8)

    def test_mission_time_structure_reserves_online_computation(self):
        timing = MissionTime(travel_s=12., sampling_s=5.)
        self.assertEqual(timing.online_computation_s, 0.)
        self.assertEqual(timing.total_s, 17.)
        self.assertEqual(timing.as_dict(), {"travel_s": 12., "sampling_s": 5.,
                                             "online_computation_s": 0., "total_s": 17.})
        # A future adapter can represent a charge; no current controller sets one.
        reserved = MissionTime(travel_s=12., sampling_s=5., online_computation_s=3.)
        self.assertEqual(reserved.total_s, 20.)

    def test_planning_wall_runtime_does_not_consume_mission_budget(self):
        class WallRuntimePlanner(GRGPlanner):
            def plan(self, *args, **kwargs):
                return replace(super().plan(*args, **kwargs), runtime_s=100000.)

        outputs = []
        for planner in [self.planner, WallRuntimePlanner(self.points, self.travel, 12, 5)]:
            mission = MissionController(copy.deepcopy(self.template), planner,
                PythonNavigationSimulator(self.grid, .35), Sensor(),
                {**self.config["mission"], "budget_s": 600})
            summary = mission.run()
            self.assertEqual(summary["online_computation_time_s"], 0.)
            self.assertAlmostEqual(summary["mission_time_s"], summary["elapsed_s"], places=8)
            outputs.append((summary, mission.online, mission.nav.trajectory))
        self.assertEqual(outputs[0][1:], outputs[1][1:])
        self.assertAlmostEqual(outputs[0][0]["elapsed_s"], outputs[1][0]["elapsed_s"])
        self.assertGreater(outputs[1][0]["grg_runtime_s"], outputs[1][0]["budget_s"])

    def test_invalid_measurement_dwell_is_still_accounted(self):
        class InvalidSensor(Sensor):
            def measure(self, position, timestamp_h):
                return Measurement(*position, timestamp_h, np.nan, "test")

        planner = GRGPlanner(self.points[:1], np.zeros((1, 1)), 12, 5)
        mission = MissionController(self.gp, planner, PythonNavigationSimulator(self.grid, .35),
                                    InvalidSensor(), {**self.config["mission"], "budget_s": 10})
        summary = mission.run()
        self.assertEqual(summary["measurements"], 0)
        self.assertEqual(self.gp.version, 0)
        # Rejected acquisition consumes dwell. With no other eligible target,
        # terminate on the next call; do not spin empty cooldown cycles.
        self.assertEqual(summary["time_breakdown"], {"travel_s": 0., "sampling_s": 5.,
                                                      "online_computation_s": 0., "total_s": 5.})
        self.assertEqual(summary["elapsed_s"], 5.)
        self.assertTrue(summary["returned_home"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
