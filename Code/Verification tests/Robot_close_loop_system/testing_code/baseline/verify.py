"""Validate the unified current baseline and preserve historical experiment data."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiment import ROOT
import test_revisit
import test_failures

HERE = Path(__file__).resolve().parent
RESULT = ROOT / "10.18_result/baseline/current/verification"


def snapshot():
    files = []
    for folder in [ROOT / "testing_code", ROOT / "10.18_result/baseline",
                   ROOT / "10.18_result/grg_depth_benchmark", ROOT / "10.18_result/grg_correcness"]:
        files.extend(p for p in folder.rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts
                     and not p.is_relative_to(ROOT / "10.18_result/baseline/current"))
    files.append(ROOT / "10.18_test_report.ipynb")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (RESULT / "tests.log").write_text(stream.getvalue())
    records = {"passed": result.wasSuccessful(), "count": result.testsRun,
               "failures": [str(t) for t, _ in result.failures], "errors": [str(t) for t, _ in result.errors]}
    run = subprocess.run([sys.executable, "testing_code/grg_depth_benchmark/test_benchmark.py"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    (RESULT / "historical_benchmark.log").write_text(run.stdout + run.stderr)
    after = snapshot()
    data = {"tests": records, "historical_benchmark_checks_passed": run.returncode == 0,
            "files_unchanged_during_tests": before == after, "file_count": len(before),
            "evidence": {**test_revisit.EVIDENCE, "failure_policy": test_failures.EVIDENCE},
            "preserved_sha256": before,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(HERE.glob("*")) if p.suffix in {".py", ".json", ".txt"}}}
    (RESULT / "checks.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(stream.getvalue())
    print("Historical benchmark checks passed:", run.returncode == 0)
    print("Files unchanged during tests:", before == after)
    assert result.wasSuccessful() and run.returncode == 0 and before == after


if __name__ == "__main__":
    main()
