"""Run one prepared scenario and policy inside a sourced ROS 2 Jazzy environment."""

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

from prepare import (
    FINAL_TEST, NAV2, POLICIES, SCENARIO_FOLDERS, experiment_directory,
    prepare, read_json, sha256, write_json,
)

INPUT_FILES = ("initial.json", "field.json", "protocol.json", "candidates.json")
TOPICS = (
    "/clock", "/odom", "/tf", "/tf_static", "/simulation/ground_truth/odom", "/cmd_vel",
    "/cl/query", "/cl/temperature",
)


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def wait_for_mission(output, processes, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for name, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {process.returncode}. See {name}.log.")
        try:
            mission = read_json(output / "mission.json")
            if mission["status"] != "window_completed":
                return mission
            read_json(output / "worker_status.json")
            return mission
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(.2)
    raise TimeoutError(f"No completed mission within the {timeout_s:g} s wall-clock guard")


def run(bundle, scenario, policy, record, gui, timeout_s):
    scenario_input = bundle / "scenarios" / scenario
    spec = read_json(scenario_input / "scenario.json")
    protocol = read_json(scenario_input / "protocol.json")
    protocol["policies"] = [policy]
    geometry = bundle / "navigation" / spec["geometry"]
    cache = geometry / "graph.json"
    shared = bundle / "shared"
    group = FINAL_TEST / SCENARIO_FOLDERS[scenario]
    output = group / "results" / bundle.name / policy
    recording = group / "recordings" / bundle.name / policy / "rosbag"
    with (bundle / ".run.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        for name in ("initial.json", "field.json"):
            shutil.copy2(shared / name, output / name)
        shutil.copy2(geometry / "candidates.json", output / "candidates.json")
        write_json(output / "protocol.json", protocol)
        if cache.is_file():
            shutil.copy2(cache, output / "graph.json")
        trace = dict(
            scenario=scenario, policy=policy, inputs={name: sha256(output / name) for name in INPUT_FILES},
            navigation={name: sha256(geometry / name) for name in ("map.yaml", "map.pgm", "world.sdf")},
            graph_source="bundle cache" if cache.is_file() else "ComputePathToPose",
            status="running",
        )
        write_json(output / "run_status.json", trace)
        env = os.environ.copy()
        env.update(
            CL_OUTPUT=str(output), CL_MATRIX_THREADS=str(protocol["matrix_threads"]),
            OMP_NUM_THREADS=str(protocol["matrix_threads"]), OMP_DYNAMIC="FALSE",
            OPENBLAS_NUM_THREADS="1", GZ_PARTITION=f"welle_{uuid4().hex}",
        )
        env.setdefault("ROS_DOMAIN_ID", "106")
        env.setdefault("ROS_HOME", "/ros-data")

        try:
            with ExitStack() as stack:
                processes = {}

                def start(name, command):
                    log = stack.enter_context((output / f"{name}.log").open("w", encoding="utf-8"))
                    child = subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    stack.callback(stop, child)
                    processes[name] = child

                start("scene", ["ros2", "launch", str(NAV2 / "launch/scene.launch.py"),
                                f"fixture_dir:={geometry}", f"gui:={str(gui).lower()}"])
                if record:
                    recording.parent.mkdir(parents=True, exist_ok=True)
                    start("bag", ["ros2", "bag", "record", "-s", "mcap", "-o",
                                  str(recording), *TOPICS])
                start("temperature", [sys.executable, str(NAV2 / "tools/srf_monitor.py")])
                start("monitor", [sys.executable, str(NAV2 / "tools/srf_monitor.py"), "monitor"])
                start("controller", ["ros2", "run", "well_e_closed_loop", "closed_loop"])
                mission = wait_for_mission(output, processes, timeout_s)

            trace.update(
                status=mission["status"], within_mission_budget=mission["within_mission_budget"],
                home_returned=mission["home_returned"], mission_summary=mission,
            )
            if (mission["status"] != "window_completed" or not mission["within_mission_budget"]
                    or not mission["home_returned"] or mission["navigation_failures"]):
                if mission["status"] == "window_completed":
                    trace["status"] = "overtime" if mission["home_returned"] else "failed"
                raise RuntimeError("Mission did not complete the on-time return protocol")

            graph = output / "graph.json"
            if not cache.exists():
                staged_graph = geometry / ".graph.pending"
                shutil.copy2(graph, staged_graph)
                staged_graph.replace(cache)
            trace["graph_sha256"] = sha256(graph)
            trace["status"] = "accepted"
        except (Exception, KeyboardInterrupt) as exc:
            trace["error"] = str(exc) or type(exc).__name__
            if trace["status"] == "running":
                trace["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            raise
        finally:
            write_json(output / "run_status.json", trace)
    print(f"Saved {scenario}/{policy} to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIO_FOLDERS)
    parser.add_argument("--kernel", type=Path, required=True, help="Newly trained RBF hyperparameters.json")
    parser.add_argument("--name", default="final", help="Run name shared by A, B and C")
    parser.add_argument("--policy", choices=POLICIES, help="Run one policy instead of all three")
    parser.add_argument("--record", action="store_true", help="Record ROS topics to an MCAP bag")
    parser.add_argument("--gui", action="store_true", help="Show the Gazebo GUI")
    parser.add_argument("--wall-guard-s", type=float, default=1800)
    args = parser.parse_args()
    kernel = args.kernel.resolve()
    if not kernel.is_file():
        parser.error(f"Trained kernel file does not exist: {kernel}")
    bundle = experiment_directory(args.name)
    if not bundle.exists():
        prepare(bundle, kernel)
    else:
        provenance = read_json(bundle / "shared/provenance.json")
        if provenance.get("initial_observation_count") != 0:
            parser.error("This run name belongs to an older initialization protocol. Choose a new --name.")
        if provenance["kernel_sha256"] != sha256(kernel):
            parser.error("This run name is already paired with another kernel. Choose a new --name.")
    group = FINAL_TEST / SCENARIO_FOLDERS[args.scenario]
    policies = (args.policy,) if args.policy else POLICIES
    for policy in policies:
        status = group / "results" / bundle.name / policy / "run_status.json"
        if status.is_file() and read_json(status)["status"] == "accepted":
            print(f"Already completed: {args.scenario}/{policy}")
            continue
        run(bundle, args.scenario, policy, args.record, args.gui, args.wall_guard_s)


if __name__ == "__main__":
    main()
