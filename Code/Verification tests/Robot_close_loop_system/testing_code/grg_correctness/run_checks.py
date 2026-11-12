"""Run isolated checks and existing regressions; retain failures as evidence."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from reference import ROOT
import test_recursive

HERE = Path(__file__).resolve().parent
RESULT = ROOT / "10.18_result" / "grg_correcness" / "current"


def save(name, data):
    (RESULT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def snapshot():
    files = [p for p in (ROOT / "testing_code").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for folder in [ROOT/"10.18_result/baseline",ROOT/"10.18_result/grg_depth_benchmark",ROOT/"10.18_result/grg_correcness"]:
        files.extend(p for p in folder.rglob("*") if p.is_file() and "intel_lab" not in p.parts and not p.is_relative_to(RESULT))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


class CheckResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "PASS"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.records.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.records.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(error, test)})


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_recursive)
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=CheckResult).run(suite)
    (RESULT / "new_tests.log").write_text(stream.getvalue())
    save("new_tests.json", {"passed": result.wasSuccessful(), "count": result.testsRun,
                            "failures": len(result.failures), "errors": len(result.errors), "tests": result.records})
    save("evidence.json", test_recursive.EVIDENCE)
    regressions = []
    for name, arguments in [
        ("baseline", ["-m", "unittest", "discover", "-s", "testing_code/baseline", "-p", "test_*.py", "-v"]),
        ("benchmark", ["testing_code/grg_depth_benchmark/test_benchmark.py"])]:
        run = subprocess.run([sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True, timeout=60)
        (RESULT / f"{name}_regression.log").write_text(run.stdout + run.stderr)
        regressions.append({"suite": name, "command": [sys.executable, *arguments],
                            "passed": run.returncode == 0, "returncode": run.returncode})
    save("regressions.json", regressions)
    after = snapshot()
    save("preservation.json", {"passed": before == after, "file_count": len(before),
                               "before": before, "after": after})
    audit = []
    benchmark = ROOT / "10.18_result/grg_depth_benchmark/runs"
    for path in sorted(benchmark.glob("seed_*/depth_*/planned_routes.json")):
        for index, plan in enumerate(json.loads(path.read_text())):
            samples = plan["route"][1:-1]
            duplicates = sorted(v for v in set(samples) if samples.count(v) > 1)
            audit.append({"run": str(path.parent.relative_to(benchmark)), "call": index,
                          "route": plan["route"], "repeated_nodes": duplicates})
    save("saved_route_audit.json", {"planning_calls": len(audit),
                                   "calls_with_duplicates": sum(bool(r["repeated_nodes"]) for r in audit),
                                   "calls": audit})
    save("obstacle_map_reproduction.json", test_recursive.EVIDENCE["reproduction"])
    save("numerical_summary.json", {"maximum_mi_difference_nats": max(test_recursive.EVIDENCE["mi_errors"], default=0.),
        "recursive_subproblems_checked":len(test_recursive.EVIDENCE["recursive_calls"]),
        "cache_states_checked":len(test_recursive.EVIDENCE["cache"]),
        "bounded_exhaustive_cases":len(test_recursive.EVIDENCE["exhaustive"])})
    print(stream.getvalue())
    print("Regressions:", regressions)
    print("Existing files unchanged:", before == after)
    print("Saved benchmark calls with duplicates:", sum(bool(r["repeated_nodes"]) for r in audit), "/", len(audit))
    # A confirmed defect remains a real failing test, not an expected-success label.
    return 0 if result.wasSuccessful() and all(r["passed"] for r in regressions) and before == after else 1


if __name__ == "__main__":
    sys.exit(main())
