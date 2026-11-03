"""Add three 70:30 missions without altering or rerunning the earlier five arms."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bootstrap as paths
import numpy as np
import run_strict_compare as strict
import strict_protocol
from experiment import save_json
from verify_strict import verify

POLICY = "fusion_70_30"


def read(path):
    return json.loads(Path(path).read_text())


def worker(window):
    config = read(window.parent.parent / "config_frozen.json")
    assert config["weights_mi_emse"][POLICY] == [.7, .3]
    # Runtime injection is confined to this worker; frozen old sources stay intact.
    strict_protocol.settings = lambda: config
    strict.settings = lambda: config
    strict.worker(window, POLICY)
    with (window / POLICY / "objective_components.csv").open() as stream:
        scores = list(csv.DictReader(stream))
    valid = all(np.isclose(float(r["objective_score"]),
        .7 * float(r["normalized_mi"]) + .3 * float(r["normalized_emse"]),
        atol=1e-8, rtol=1e-7) and r["policy"] == POLICY for r in scores)
    save_json(window / POLICY / "weight_verification.json", dict(passed=bool(scores) and valid,
        mi_weight=.7, emse_weight=.3, checked_plans=len(scores)))
    assert scores and valid, "Saved route scores do not use the requested 70:30 weights"


def supervise(window, config):
    folder = window / POLICY
    folder.mkdir()
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--window", str(window)]
    begin, reason = time.time(), None
    with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        while process.poll() is None:
            active = None
            if (folder / "events.jsonl").exists():
                for line in (folder / "events.jsonl").read_text().splitlines():
                    try:
                        event = json.loads(line)
                        active = event if event["event"] == "plan_start" else None
                    except json.JSONDecodeError:
                        pass
            if active and time.time() - active["timestamp_s"] > config["planning_wall_limit_s"]:
                reason = "planning_wall_limit"
            if time.time() - begin > config["mission_wall_limit_s"]:
                reason = "mission_wall_limit"
            if reason:
                process.kill()
                process.wait()
                break
            time.sleep(.1)
    result = dict(status="TIMEOUT" if reason else ("SUCCESS" if process.returncode == 0 else "ERROR"),
        reason=reason, returncode=process.returncode, wall_s=time.time() - begin)
    save_json(folder / "execution.json", result)
    print(window.name, POLICY, result["status"], round(result["wall_s"], 2), "s", flush=True)
    if result["status"] != "SUCCESS":
        raise RuntimeError(f"Inspect preserved logs in {folder}")


def main():
    previous = paths.ROOT / "11.2_result" / read(paths.ROOT / "11.2_result/latest_strict.json")["run"]
    parent_manifest = read(previous / "manifest.json")
    assert read(previous / "verification.json")["passed"]
    for relative, digest in read(previous / "artifacts_sha256.json").items():
        assert paths.sha256(previous / relative) == digest, relative
    for relative, digest in parent_manifest["source_sha256"].items():
        assert paths.sha256(paths.ROOT / relative) == digest, relative
    config = read(previous / "config_frozen.json")
    assert config["policies"] == ["mi", "emse", "fusion", "fusion_40_60", "fusion_60_40"]
    config["policies"].append(POLICY)
    config["weights_mi_emse"][POLICY] = [.7, .3]
    config["selection"] = "Exploratory 70:30 follow-up requested after viewing the first five arms; identical windows and labels, not independent validation or a prespecified six-arm study."
    root = paths.ROOT / "11.2_result" / datetime.now(timezone.utc).strftime("strict_plus_70_30_%Y%m%dT%H%M%SZ")
    root.mkdir(exist_ok=False)
    (root / "windows").mkdir()
    protected = dict(parent_manifest["protected_inputs_sha256"])
    for p in list(previous.rglob("*")) + [paths.ROOT / "README_strict.md", paths.ROOT / "11.2_strict_test_report.ipynb", paths.ROOT / "11.2_result/latest_strict.json"]:
        if p.is_file():
            protected[str(p.relative_to(paths.WORKSPACE))] = paths.sha256(p)
    sources = {str(p.relative_to(paths.ROOT)): paths.sha256(p) for p in paths.HERE.glob("*.py")
               if not p.name.startswith("build_")}
    sources[str(strict_protocol.STRICT_CONFIG.relative_to(paths.ROOT))] = paths.sha256(strict_protocol.STRICT_CONFIG)
    save_json(root / "manifest.json", dict(protected_inputs_sha256=protected, source_sha256=sources,
        prior_three_policy_run=parent_manifest["prior_three_policy_run"],
        prior_five_policy_run=str(previous.relative_to(paths.ROOT)),
        historical_checkpoint_sha256=paths.sha256(paths.CHECKPOINT), environment=parent_manifest["environment"],
        selection_note=config["selection"], newly_executed_policies=[POLICY],
        reused_policies=["history_only"] + config["policies"][:-1]))
    save_json(root / "config_frozen.json", config)
    for name in ["test_objectives.py", "test_strict_protocol.py"]:
        with (root / f"{Path(name).stem}.log").open("w") as stream:
            completed = subprocess.run([sys.executable, "-B", str(paths.HERE / name)], stdout=stream, stderr=subprocess.STDOUT)
        assert completed.returncode == 0, name
    save_json(root / "unit_tests.json", dict(passed=True, suites=["test_objectives.py", "test_strict_protocol.py"]))
    copied = {}
    for source in sorted((previous / "windows").iterdir()):
        window = root / "windows" / source.name
        shutil.copytree(source, window)
        for p in source.rglob("*"):
            if p.is_file() and p.relative_to(source).as_posix() != "protocol.json":
                copied[str((window / p.relative_to(source)).relative_to(root))] = paths.sha256(p)
        protocol = read(source / "protocol.json")
        protocol["weights_mi_emse"] = config["weights_mi_emse"]
        protocol["extension"] = dict(new_policy=POLICY, reused_window=str(source.relative_to(paths.ROOT)),
            note="Common inputs and earlier arms copied byte-for-byte; only 70:30 is newly executed.")
        save_json(window / "protocol.json", protocol)
        supervise(window, config)
    checks = dict(copied_inputs_and_previous_arms_identical=all(paths.sha256(root / p) == h for p, h in copied.items()),
        requested_weight_scores=all(read(w / POLICY / "weight_verification.json")["passed"] for w in (root / "windows").iterdir()))
    save_json(root / "extension_verification.json", dict(passed=all(checks.values()), checks=checks,
        copied_artifacts_sha256=copied, newly_executed_missions=3))
    assert all(checks.values()), checks
    verify(root)
    save_json(paths.ROOT / "11.2_result/latest_70_30.json", dict(run=root.name))
    print("Completed 70:30 extension:", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--window", type=Path)
    args = parser.parse_args()
    worker(args.window) if args.worker else main()
