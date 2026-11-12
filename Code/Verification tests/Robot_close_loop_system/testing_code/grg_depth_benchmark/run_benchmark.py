"""Sequential paired depth benchmark with an external process watchdog."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = ROOT / "10.18_result" / "grg_depth_benchmark"
WORKER = HERE / "worker.py"
SEEDS = [18, 19, 20, 21, 22]
DEPTHS = [2, 3, 5]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_events(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A worker may be appending its final line.
    return rows


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervised(command, folder, plan_limit=90, mission_limit=900):
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reason, active = None, None
    with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        while process.poll() is None:
            now = time.perf_counter()
            events = read_events(folder / "events.jsonl")
            active = None
            for row in events:
                if row["event"] == "plan_start":
                    active = row
                elif row["event"] == "plan_end":
                    active = None
            if active and now - active["wall_s"] > plan_limit:
                reason = "planning_call_limit"
            elif now - started > mission_limit:
                reason = "mission_wall_limit"
            if reason:
                stop_process(process)
                break
            time.sleep(.05)
    elapsed = time.perf_counter() - started
    info = {"status": "TIMEOUT" if reason else ("SUCCESS" if process.returncode == 0 else "ERROR"),
            "reason": reason, "wall_s": elapsed, "returncode": process.returncode,
            "planning_limit_s": plan_limit, "mission_limit_s": mission_limit}
    if reason and active:
        info.update(timed_out_call=active["call"],
                    timed_out_call_s=time.perf_counter() - active["wall_s"],
                    elapsed_simulation_s=active["elapsed_s"])
    save(folder / "execution.json", info)
    return info


def run_case(mode, seed, depth, folder):
    if (folder / "execution.json").is_file():
        previous = json.loads((folder / "execution.json").read_text())
        if previous["status"] in ["SUCCESS", "TIMEOUT"]:
            summary = json.loads((folder / "summary.json").read_text())
            previous["status"] = summary["status"]
            print(f"Reuse recorded {mode}: seed={seed}, depth={depth}, {previous['status']}", flush=True)
            return previous
        raise RuntimeError(f"Inspect and archive failed output before retrying: {folder}")
    execution = supervised([sys.executable, str(WORKER), mode, "--seed", str(seed),
                            "--depth", str(depth), "--output", str(folder)], folder)
    if execution["status"] != "SUCCESS":
        save(folder / "summary.json", {"seed": seed, "depth": depth, **execution})
    else:
        execution["status"] = json.loads((folder / "summary.json").read_text())["status"]
    print(f"{mode}: seed={seed}, depth={depth}, {execution['status']}, "
          f"wall={execution['wall_s']:.2f}s", flush=True)
    return execution


def baseline_check(folder):
    import csv
    import numpy as np
    original = ROOT / "10.18_result" / "baseline" / "main"
    a = json.loads((original / "mission_summary.json").read_text())
    b = json.loads((folder / "summary.json").read_text())
    checks = {key: bool(np.isclose(a[key], b[key], atol=1e-10, rtol=1e-10)) for key in [
        "measurements", "planning_calls", "travel_distance_m", "travel_time_s", "elapsed_s",
        "initial_rmse_c", "final_rmse_c", "initial_variance_c2", "final_variance_c2"]}
    for name in ["online_measurements.csv", "robot_trajectory.csv", "metrics.csv"]:
        checks[name] = (original / name).read_bytes() == (folder / name).read_bytes()
    routes_a = json.loads((original / "planned_routes.json").read_text())
    routes_b = json.loads((folder / "planned_routes.json").read_text())
    checks["routes"] = [r["route"] for r in routes_a] == [r["route"] for r in routes_b]
    checks["return_home"] = a["returned_home"] == b["returned_home"] is True
    report = {"passed": all(checks.values()), "checks": checks,
              "excluded": "Wall-clock runtime; not expected to reproduce exactly"}
    save(RESULT / "verification" / "baseline_reproduction.json", report)
    if not report["passed"]:
        raise AssertionError("Depth 2 did not reproduce the baseline; stop comparisons")


def main():
    config = json.loads((ROOT / "testing_code/baseline/config.json").read_text())
    if config["mission"]["allow_revisit"]:
        raise SystemExit("This saved benchmark used the old no-revisit policy. "
                         "Current baseline enables revisits. Historical results are retained; "
                         "use testing_code/baseline/verify.py to validate current code.")
    RESULT.mkdir(parents=True, exist_ok=True)
    tracked = [p for p in (ROOT / "testing_code" / "baseline").glob("*") if p.suffix in [".py", ".json", ".txt"]]
    original_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    save(RESULT / "experiment_config.json", {"seeds": SEEDS, "depths": DEPTHS,
                                            "planning_limit_s": 90, "mission_wall_limit_s": 900,
                                            "baseline_commit": "bbd018151b38a4af718f0e215edc3de7e4d86d9f",
                                            "only_changed_model_setting": "grg.depth"})
    for seed in SEEDS:
        subprocess.run([sys.executable, str(WORKER), "prepare", "--seed", str(seed)], check=True)
    baseline = RESULT / "runs" / "seed_18" / "depth_2"
    assert run_case("mission", 18, 2, baseline)["status"] == "SUCCESS"
    baseline_check(baseline)
    feasible = []
    for depth in DEPTHS:
        outcome = run_case("screen", 18, depth, RESULT / "screening" / f"depth_{depth}")
        if outcome["status"] == "SUCCESS":
            feasible.append(depth)
    # A timed-out depth is not retried with altered search settings. Complete
    # the paired experiment for every depth that passed the initial screening.
    for seed in SEEDS:
        for depth in DEPTHS:
            if seed == 18 and depth == 2:
                continue
            folder = RESULT / "runs" / f"seed_{seed}" / f"depth_{depth}"
            if depth not in feasible:
                save(folder / "summary.json", {"seed": seed, "depth": depth,
                                               "status": "NOT_RUN_SCREEN_TIMEOUT",
                                               "reason": "Seed 18 initial screen did not pass; no mission attempted"})
                continue
            run_case("mission", seed, depth, folder)
    checks = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
              for name, digest in original_hashes.items()}
    save(RESULT / "verification" / "baseline_sources_unchanged.json", {
        "passed": all(checks.values()), "checks": checks, "sha256": original_hashes})
    if not all(checks.values()):
        raise AssertionError("Baseline source changed")
    print("Benchmark executions complete. Generate the report with report.py.", flush=True)


if __name__ == "__main__":
    main()
