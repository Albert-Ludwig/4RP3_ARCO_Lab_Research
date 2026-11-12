"""Closed-loop control with Home sampling and classified candidate exclusions."""

from dataclasses import asdict
import numpy as np

from grg import information_gain
from interfaces import Navigation, TemperatureSource
from timing import MissionTime
from failures import CandidateFailures, PERMANENT_REASONS


class MissionController:
    def __init__(self, gp, planner, nav: Navigation, sensor: TemperatureSource,
                 config, observer=None):
        self.gp, self.planner, self.nav, self.sensor = gp, planner, nav, sensor
        self.config, self.observer = config, observer
        self.online, self.log, self.plans = [], [], []
        self.distance, self.total_gain = 0.0, 0.0
        self.measurement_wait_s = 0.0
        self.dwell_s = 0.0
        # The ROS/Nav2 integration may populate online computation later.
        # Python simulation leaves it at zero; plan.runtime_s is wall time only.
        self.timing = MissionTime()
        self.failures = CandidateFailures()
        proof = getattr(self.nav, "static_failure_reason", None)
        if callable(proof):
            for node, point in enumerate(self.planner.points):
                reason = proof(point)
                if reason in PERMANENT_REASONS:
                    self.failures.mark_permanent(node, reason)
        self.stop_reason = "not_started"

    def navigation_failed(self, node, planning_round, status):
        proof = getattr(self.nav, "static_failure_reason", None)
        reason = proof(self.planner.points[node]) if callable(proof) else None
        if reason in PERMANENT_REASONS:
            self.failures.mark_permanent(node, reason)
            return "permanent", reason
        self.failures.mark_temporary(node, planning_round, status)
        return "temporary", status

    def run(self):
        c = self.config
        current, visited = 0, set()
        if self.observer:
            self.observer(self.gp, self.nav.get_robot_state().elapsed_s, 0)
        for iteration in range(c["max_iterations"]):
            blocked = self.failures.blocked_for(iteration)
            state = self.nav.get_robot_state()
            remaining = c["budget_s"] - state.elapsed_s
            plan = self.planner.plan(self.gp, current, state.elapsed_s, remaining, visited, blocked)
            self.plans.append({"iteration": iteration, "elapsed_s": state.elapsed_s, **asdict(plan)})
            row = {"iteration": iteration, "x_m": state.x_m, "y_m": state.y_m,
                   "elapsed_s": state.elapsed_s, "remaining_s": remaining,
                   "candidates": [i for i in range(len(self.planner.points)) if i not in blocked],
                   "route": plan.route, "route_gain_nats": plan.gain,
                   "plan_s": plan.runtime_s, "gp_version": self.gp.version,
                   "online_computation_time_s": 0.0, "sampling_time_s": 0.0,
                   "time_breakdown_start": self.timing.as_dict(),
                   "blocked_nodes": sorted(blocked),
                   "candidate_failures_start": self.failures.snapshot(),
                   "gp_updated": False}
            if not plan.route or len(plan.route) < 3:
                self.stop_reason = "no_feasible_sampling_route"
                row["status"] = self.stop_reason
                self.log.append(row)
                break
            if plan.finish_s > c["budget_s"] + 1e-8:
                self.stop_reason = "infeasible_plan"
                row["status"] = self.stop_reason
                self.log.append(row)
                break
            node = plan.route[1]
            goal = self.planner.points[node]
            estimate = self.nav.estimate_travel_cost(state.xy, goal)
            back = self.nav.estimate_travel_cost(goal, self.planner.points[0])
            row.update(next_node=node, estimated_travel_s=estimate)
            if not np.isfinite(estimate + back):
                kind, reason = self.navigation_failed(node, iteration, "travel_estimate_unavailable")
                row.update(status="navigation_unavailable", failure_kind=kind, failure_reason=reason,
                           candidate_failures_after=self.failures.snapshot())
                self.log.append(row)
                continue
            if estimate + c["sample_s"] + back > remaining + 1e-8:
                self.stop_reason = "return_reserve"
                row["status"] = self.stop_reason
                self.log.append(row)
                break
            result = self.nav.navigate_to(goal)
            actual = self.nav.get_robot_state()
            self.distance += result.distance_m
            self.timing.travel_s += result.travel_s
            row.update(navigation_status=result.status, actual_travel_s=result.travel_s,
                       arrival_x_m=actual.x_m, arrival_y_m=actual.y_m,
                       arrival_s=actual.elapsed_s, arrival_t_h=c["start_h"] + actual.elapsed_s / 3600)
            if result.status != "arrived" or not np.allclose(actual.xy, goal, atol=1e-8, rtol=0):
                status = result.status if result.status != "arrived" else "arrival_pose_mismatch"
                kind, reason = self.navigation_failed(node, iteration, status)
                row.update(status="navigation_failed", failure_kind=kind, failure_reason=reason,
                           candidate_failures_after=self.failures.snapshot())
                self.log.append(row)
                if not np.allclose(actual.xy, state.xy, atol=1e-8, rtol=0):
                    self.stop_reason = "navigation_failed_after_motion"
                    break
                continue
            current = node
            back = self.nav.estimate_travel_cost(actual.xy, self.planner.points[0])
            if actual.elapsed_s + c["sample_s"] + back > c["budget_s"] + 1e-8:
                row["status"] = "arrival_return_reserve"
                self.log.append(row)
                self.stop_reason = row["status"]
                break
            # Recorded-data sources may require acquisition waiting. Only
            # timestamps are consulted before acquisition, never temperatures.
            next_time = getattr(self.sensor, "next_time", None)
            measurement_h = next_time(actual.xy, row["arrival_t_h"]) if callable(next_time) else row["arrival_t_h"]
            wait_s = max(0., (measurement_h - row["arrival_t_h"]) * 3600)
            if not np.isfinite(wait_s) or actual.elapsed_s + wait_s + c["sample_s"] + back > c["budget_s"] + 1e-8:
                self.failures.measurement_failed(node, iteration)
                row["status"] = "measurement_unavailable_within_budget"
                self.log.append(row)
                continue
            if wait_s > 0:
                self.nav.wait(wait_s)
            self.timing.sampling_s += wait_s
            self.measurement_wait_s += wait_s
            record = self.sensor.measure(actual.xy, measurement_h)
            self.nav.wait(c["sample_s"])
            self.timing.sampling_s += c["sample_s"]
            self.dwell_s += c["sample_s"]
            row.update(sampling_time_s=wait_s + c["sample_s"], measurement_wait_s=wait_s,
                       measurement_t_h=measurement_h, dwell_s=c["sample_s"],
                       time_breakdown_after_sampling=self.timing.as_dict())
            matches = (record.valid() and np.allclose([record.x_m, record.y_m], actual.xy,
                                                     atol=1e-8, rtol=0)
                       and abs(record.t_h - measurement_h) < 1e-10)
            if not matches:
                row["status"] = "invalid_measurement"
                self.failures.measurement_failed(node, iteration)
                row.update(failure_kind="temporary", failure_reason="measurement_failed",
                           candidate_failures_after=self.failures.snapshot())
                self.log.append(row)
                continue
            query = np.array([[record.x_m, record.y_m, record.t_h]])
            gain = float(information_gain(self.gp.joint(query)[1], self.gp.noise))
            self.gp.append(record)
            self.online.append(record)
            self.total_gain += gain
            visited.add(node)
            row.update(status="sampled", temperature_c=record.temperature_c,
                       measurement_source=record.source, measurement_status=record.status,
                       gp_updated=True, gain_nats=gain, total_gain_nats=self.total_gain,
                       after_sample_s=self.nav.get_robot_state().elapsed_s)
            if self.observer:
                row.update(self.observer(self.gp, self.nav.get_robot_state().elapsed_s, len(self.online)))
            self.log.append(row)
        else:
            self.stop_reason = "iteration_limit"
        state = self.nav.get_robot_state()
        back = self.nav.estimate_travel_cost(state.xy, self.planner.points[0])
        if np.isfinite(back) and state.elapsed_s + back <= c["budget_s"] + 1e-8:
            result = self.nav.navigate_to(self.planner.points[0])
            self.distance += result.distance_m
            self.timing.travel_s += result.travel_s
            return_status = result.status
        else:
            return_status = "no_feasible_return"
        final = self.nav.get_robot_state()
        return {"measurements": len(self.online), "planning_calls": len(self.plans),
                "travel_distance_m": self.distance, "travel_time_s": self.timing.travel_s,
                "sampling_time_s": self.timing.sampling_s,
                "measurement_wait_time_s": self.measurement_wait_s, "dwell_time_s": self.dwell_s,
                "online_computation_time_s": self.timing.online_computation_s,
                "mission_time_s": self.timing.total_s,
                "time_breakdown": self.timing.as_dict(),
                "candidate_failures": self.failures.snapshot(),
                "elapsed_s": final.elapsed_s, "budget_s": c["budget_s"],
                "returned_home": bool(return_status == "arrived"
                                      and np.allclose(final.xy, self.planner.points[0])),
                "return_status": return_status, "stop_reason": self.stop_reason,
                "information_gain_nats": self.total_gain,
                "grg_runtime_s": sum(p["runtime_s"] for p in self.plans)}
