"""Execute tests, the main mission, a repeat, and two additional seeds."""

import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

import numpy as np

from experiment import ROOT, run_experiment, save_csv, save_json
from reuse import BASELINE


class TestResult(unittest.TextTestResult):
    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed",
                             "runtime_s": time.perf_counter() - self.started})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append({"test": test.id(), "status": "failed"})

    def addError(self, test, error):
        super().addError(test, error)
        self.records.append({"test": test.id(), "status": "error"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []


def run_tests(folder):
    folder.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=TestResult).run(suite)
    (folder / "tests.log").write_text(stream.getvalue())
    save_json(folder / "tests.json", {"passed": result.wasSuccessful(),
                                      "count": result.testsRun, "tests": result.records})
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise AssertionError("Automated tests failed")


def compare_repeat(first, second):
    checks = {}
    for name in ["historical_measurements.csv", "online_measurements.csv", "robot_trajectory.csv",
                 "metrics.csv", "candidates.csv"]:
        checks[name] = (first / name).read_bytes() == (second / name).read_bytes()
    a, b = np.load(first / "reconstruction.npz"), np.load(second / "reconstruction.npz")
    for key in a.files:
        checks[f"reconstruction.{key}"] = np.array_equal(a[key], b[key])
    plans_a = json.loads((first / "planned_routes.json").read_text())
    plans_b = json.loads((second / "planned_routes.json").read_text())
    for plans in [plans_a, plans_b]:
        for plan in plans:
            plan.pop("runtime_s")
    checks["ordered_routes_and_rewards"] = plans_a == plans_b
    checks["travel_costs"] = np.array_equal(np.load(first / "travel_time_s.npy"),
                                            np.load(second / "travel_time_s.npy"))
    return {"passed": all(checks.values()), "checks": checks,
            "excluded": "Wall-clock runtimes and checkpoint container metadata"}


def main():
    folder = ROOT / "10.18_result" / "baseline" / "current"
    verification = folder / "verification"
    run_tests(verification)
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    debug = copy.deepcopy(config)
    debug["mission"].update(candidate_count=12, budget_s=300)
    debug["gp"]["iterations"] = 80
    run_experiment(debug, folder / "debug", figures=False)
    rows = []
    for name, seed in [("main", config["seed"]), ("repeat", config["seed"]),
                       ("seed_19", 19), ("seed_20", 20)]:
        case = copy.deepcopy(config)
        case["seed"] = seed
        summary = run_experiment(case, folder / name, figures=name == "main")
        rows.append({"run": name, **summary})
        print(f"{name}: {summary['measurements']} samples, "
              f"RMSE {summary['initial_rmse_c']:.4f} -> {summary['final_rmse_c']:.4f} °C, "
              f"home={summary['returned_home']}", flush=True)
    reproducibility = compare_repeat(folder / "main", folder / "repeat")
    save_json(verification / "reproducibility.json", reproducibility)
    save_csv(folder / "seed_summary.csv", rows)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                            capture_output=True, text=True).stdout
    (verification / "environment_freeze.txt").write_text(freeze)
    save_json(verification / "environment.json", {"python": sys.version, "platform": platform.platform(),
                                                 "executable": sys.executable})
    sources = [*Path(__file__).parent.glob("*.py"), Path(__file__).with_name("config.json"),
               *(BASELINE / name for name in ["gp_model.py", "synthetic_field.py", "evaluation.py"])]
    save_json(verification / "source_hashes.json", {
        str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)})
    assert reproducibility["passed"], "Fixed-seed reproduction failed"
    print("All unit tests, mission checks, and fixed-seed comparisons passed.")


if __name__ == "__main__":
    main()
