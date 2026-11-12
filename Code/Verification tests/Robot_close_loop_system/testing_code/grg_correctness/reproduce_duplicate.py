"""Reproduce the no-revisit defect with the saved GP and actual obstacle map."""

from dataclasses import asdict
import json

import numpy as np
import torch

from reference import ROOT, BASE
from gp import GPModel
from grg import GRGPlanner
from navigation import GridMap, PythonNavigationSimulator, travel_matrix


def reproduce():
    torch.set_num_threads(1)
    config = json.loads((BASE / "config.json").read_text())
    gp = GPModel.load(ROOT / "10.18_result/baseline/main/initial_gp.pt")
    points = np.array([[2, 2], [7, 9], [20, 24], [28, 8]], float)
    grid = GridMap(config["map"])
    travel = travel_matrix(PythonNavigationSimulator(grid, .35), points)
    plan = GRGPlanner(points, travel, 12, 5, depth=3).plan(gp, 0, 0, 900, set())
    samples = plan.route[1:-1]
    repeated = sorted(v for v in set(samples) if samples.count(v) > 1)
    return {"points": points.tolist(), "depth": 3, "remaining_s": 900, "sample_s": 5,
            "speed_m_s": .35, "allow_revisit": True, "travel_source": "unchanged obstacle map and grid paths",
            "plan": asdict(plan), "repeated_nodes": repeated,
            "constraint_satisfied": plan.finish_s <= 900 and all(np.diff(plan.arrivals_s)>0)}


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2))
