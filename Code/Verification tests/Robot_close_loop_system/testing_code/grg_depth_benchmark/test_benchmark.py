"""Controls, timeout enforcement, and saved-result checks for the benchmark."""

import json
import hashlib
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_benchmark import RESULT, ROOT, supervised


class DepthBenchmarkTests(unittest.TestCase):
    def test_original_baseline_results_are_preserved(self):
        repo = ROOT.parents[1]
        # This folder tree is identical in the personal and ARCO repositories;
        # their commit IDs differ because their parent histories differ.
        tree = "0980afb6e74db3b0f0ddd1d9bf152c7037b23c29"
        relocation = json.loads((ROOT / "10.18_result/baseline/verification/relocation.json").read_text())
        moved = {item["old"]: item for item in relocation["files"]}
        listing = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", tree], text=True)
        for line in listing.splitlines():
            details, name = line.split("\t", 1)
            if name in ["README.md", "10.18_test_report.ipynb"]:
                continue
            # Runtime code was intentionally replaced by the user-approved
            # current baseline. Preserve historical data, not old policies.
            if name.startswith("testing_code/"):
                continue
            item = moved.get(name)
            data = (ROOT / (item["new"] if item else name)).read_bytes()
            if item and item["replacements"]:
                original = subprocess.check_output(["git", "-C", str(repo), "show", tree + ":" + name])
                self.assertEqual(hashlib.sha256(original).hexdigest(), item["original_sha256"], name)
                expected = original.decode()
                for before, after in item["replacements"]:
                    expected = expected.replace(before, after)
                self.assertEqual(expected.encode(), data, name)
            else:
                digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                self.assertEqual(digest, details.split()[2], name)
        import nbformat
        saved = subprocess.check_output(["git", "-C", str(repo), "show", tree + ":10.18_test_report.ipynb"], text=True)
        baseline = nbformat.reads(saved, as_version=4)
        for cell in baseline.cells:
            for before, after in relocation["notebook_replacements"]:
                if before.startswith('if root.name'):
                    after = after.replace('\n        root', '\n    root').replace('\n    if root', '\nif root')
                cell.source = cell.source.replace(before, after)
        archived = ROOT / "10.18_result/baseline/archived_notebooks/10.18_synthetic_report.ipynb"
        current = nbformat.read(archived if archived.exists() else ROOT / "10.18_test_report.ipynb", as_version=4)
        additions = {"grg_depth_benchmark", "intel_lab", "grg_policy_verification"}
        preserved = [c for c in current.cells if not additions.intersection(c.metadata.get("tags", []))]
        self.assertEqual(baseline.cells, preserved)

    def test_planning_timeout_stops_worker(self):
        with tempfile.TemporaryDirectory(prefix="grg-watchdog-") as temp:
            folder = Path(temp)
            script = ("import json,time; from pathlib import Path; "
                      f"p=Path({str(folder / 'events.jsonl')!r}); "
                      "p.write_text(json.dumps(dict(event='plan_start',call=0,elapsed_s=0,"
                      "wall_s=time.perf_counter()))+'\\n'); time.sleep(30)")
            outcome = supervised([sys.executable, "-c", script], folder, plan_limit=.2, mission_limit=5)
            self.assertEqual(outcome["status"], "TIMEOUT")
            self.assertEqual(outcome["reason"], "planning_call_limit")
            self.assertGreaterEqual(outcome["timed_out_call_s"], .2)
            self.assertLess(outcome["wall_s"], 3)

    def test_mission_wall_timeout_stops_worker(self):
        with tempfile.TemporaryDirectory(prefix="grg-wall-") as temp:
            outcome = supervised([sys.executable, "-c", "import time; time.sleep(30)"],
                                 Path(temp), plan_limit=90, mission_limit=.2)
            self.assertEqual(outcome["reason"], "mission_wall_limit")
            self.assertLess(outcome["wall_s"], 3)

    def test_depth_two_reproduces_baseline(self):
        self.assertTrue(json.loads((RESULT / "verification/baseline_reproduction.json").read_text())["passed"])

    def test_only_depth_changes_and_initial_checkpoint_is_shared(self):
        for seed in [18, 19, 20, 21, 22]:
            shared = RESULT / "shared" / f"seed_{seed}"
            base = json.loads((shared / "config.json").read_text())
            checkpoint = (shared / "initial_gp.pt").read_bytes()
            for depth in [2, 3, 5]:
                folder = RESULT / "runs" / f"seed_{seed}" / f"depth_{depth}"
                if not (folder / "config.json").is_file():
                    continue
                config = json.loads((folder / "config.json").read_text())
                self.assertEqual(config["grg"]["depth"], depth)
                config["grg"]["depth"] = base["grg"]["depth"]
                self.assertEqual(config, base)
                self.assertEqual((folder / "initial_gp.pt").read_bytes(), checkpoint)

    def test_completed_missions_and_waypoint_bound(self):
        completed = 0
        for path in (RESULT / "runs").glob("seed_*/depth_*/summary.json"):
            summary = json.loads(path.read_text())
            if summary["status"] != "SUCCESS":
                continue
            completed += 1
            checks = json.loads(path.with_name("verification.json").read_text())
            self.assertTrue(checks["passed"])
            self.assertLessEqual(summary["max_planned_waypoints"], 2**summary["depth"] - 1)
            self.assertEqual(summary["navigation_failures"], 0)
            self.assertEqual(summary["repeated_planned_visits"], 0)
            self.assertLessEqual(summary["max_planning_s"], 90)
        self.assertGreaterEqual(completed, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
