"""Freeze the shared field and GP and prepare navigation inputs without starting ROS."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

FINAL_TEST = Path(__file__).resolve().parents[1]
FINAL_CODE = FINAL_TEST.parent
REPO = FINAL_CODE.parent.parent
NAV2 = FINAL_CODE / "Nav2_ROS"
VERIFIED = (REPO / "Code/Verification tests/WIVR_test/results"
            / "run_20261124T031956Z/latest_gp")
POLICIES = ("mi", "emse", "wivr")
SCENARIO_FOLDERS = {"A": "Baseline", "B": "Test B", "C": "Test C"}
sys.path.insert(0, str(NAV2 / "tools"))
from srf_field import TemperatureField


def experiment_directory(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Use an experiment name with letters, digits, dots, underscores or hyphens")
    return FINAL_TEST / "results" / name


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kernel(path):
    hp = read_json(path)
    length = np.asarray(hp["lengthscales_m_m_s"], dtype=float)
    noise = float(hp["noise_variance_c2"])
    if (length.shape != (3,) or not np.isfinite(length).all() or (length <= 0).any()
            or not np.isfinite(noise) or noise <= 0):
        raise ValueError("Expected positive RBF lengths in m, m, s and positive noise variance in °C²")
    if "kernel" in hp and "rbf" not in hp["kernel"].lower():
        raise ValueError("The closed-loop implementation expects fitted RBF hyperparameters")
    return dict(length=length.tolist(), noise=noise)


def make_arena(path, extent, settings):
    path.mkdir()
    xmin, ymin = np.asarray(settings["map_field_origin_m"]) - settings["clearance_m"]
    xmax, ymax = np.asarray(settings["map_field_origin_m"]) + extent
    resolution = float(settings["map_resolution_m"])
    width, height = xmax - xmin, ymax - ymin
    cols, rows = round(width / resolution), round(height / resolution)
    border = bytes(cols)
    free_row = b"\x00" + b"\xfe" * (cols - 2) + b"\x00"
    (path / "map.pgm").write_bytes(
        f"P5\n{cols} {rows}\n255\n".encode() + border + free_row * (rows - 2) + border
    )
    (path / "map.yaml").write_text(
        f"image: map.pgm\nresolution: {resolution}\norigin: [{xmin}, {ymin}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n",
        encoding="utf-8",
    )

    world = ET.parse(NAV2 / "worlds/world.sdf")
    mid_x, mid_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    plane = world.find("./world/model[@name='ground_plane']")
    if width > 39 or height > 39:
        ET.SubElement(plane, "pose").text = f"{mid_x} {mid_y} 0 0 0 0"
        for size in plane.findall("./link/*/geometry/plane/size"):
            size.text = f"{width + 2} {height + 2}"
    walls = {
        "west_wall": (xmin - .05, mid_y, .1, height + .2),
        "east_wall": (xmax + .05, mid_y, .1, height + .2),
        "south_wall": (mid_x, ymin - .05, width, .1),
        "north_wall": (mid_x, ymax + .05, width, .1),
    }
    for name, (x, y, dx, dy) in walls.items():
        wall = world.find(f"./world/model[@name='{name}']")
        wall.find("pose").text = f"{x} {y} 0.5 0 0 0"
        for size in wall.findall("./link/*/geometry/box/size"):
            size.text = f"{dx} {dy} 1.0"
    world.write(path / "world.sdf", encoding="utf-8", xml_declaration=True)


def prepare(bundle, kernel_path):
    settings = read_json(FINAL_TEST / "settings.json")
    specs = [
        read_json(FINAL_TEST / folder / "code/scenario.json")
        for folder in SCENARIO_FOLDERS.values()
    ]
    small, large = specs[0]["extent_m"], specs[2]["extent_m"]
    mean = float(settings["prior_mean_c"])
    variance = float(settings["prior_variance_c2"])
    if not np.isfinite(mean):
        raise ValueError("The GP prior mean must be finite")
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("The GP prior variance must be positive and finite")
    field_source = NAV2 / "fixtures/field.json"
    field = read_json(field_source)
    if small != field["working_extent_m"]:
        raise ValueError("Baseline must use the verified arena extent")
    reference_points = np.asarray(read_json(VERIFIED / "candidates.json"), dtype=float)
    if (reference_points.shape != (settings["candidate_count"] + 1, 2)
            or not np.array_equal(reference_points[0], settings["home_m"])):
        raise ValueError("Candidate count and Home must match the verified arena")
    for key in ("historical_reference", "status", "visualization"):
        field.pop(key, None)
    field.update(
        working_extent_m=large,
        working_origin_m=[-settings["clearance_m"]] * 2,
        mission_interval_s=[0, max(s["mission_budget_s"] for s in specs)],
    )
    initial = dict(read_kernel(kernel_path), mean=mean, variance=variance, x=[], y=[])

    protocol = read_json(VERIFIED / "protocol.json")
    for key in ("reference_fixture_run", "history_seed", "history_count", "history_interval_s"):
        protocol.pop(key, None)
    protocol.update(
        candidate_count=settings["candidate_count"],
        home_m=settings["home_m"],
        field_seeds=[field["field_seed"]],
        initial_observation_count=0,
        prior_mean_c=mean,
        prior_variance_c2=variance,
        scoring_cell_m=settings["scoring_cell_m"],
        scoring_time_step_s=settings["scoring_time_step_s"],
        search_guard_s=settings["search_guard_s"],
        evaluation_cell_m=settings["evaluation_cell_m"],
        evaluation_time_step_s=settings["evaluation_time_step_s"],
        update_interval_s=settings["gp_update_interval_s"],
        update_intervals_s=[settings["gp_update_interval_s"]],
        initial_gp="RBF prior with fixed mean and variance and zero pre-mission observations.",
        run_end="Reserve return time within the scenario budget and sense through Home arrival.",
        reconstruction="Final posterior on the scenario and common-support grids.",
        execution_mode="Background latest-GP search with arrival-only waypoint switching.",
    )

    bundle.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=bundle.parent) as stage:
        output = Path(stage) / "bundle"
        output.mkdir()
        shared = output / "shared"
        shared.mkdir()
        write_json(shared / "field.json", field)
        write_json(shared / "initial.json", initial)
        write_json(shared / "kernel.json", read_json(kernel_path))
        write_json(shared / "settings.json", settings)
        write_json(shared / "provenance.json", dict(
            kernel_path=str(kernel_path), kernel_sha256=sha256(kernel_path),
            initial_observation_count=0, prior_mean_c=mean, prior_variance_c2=variance,
            field_reference_sha256=sha256(field_source),
            verification_graph_sha256=sha256(VERIFIED / "graph.json"),
            field_sha256=sha256(shared / "field.json"), initial_sha256=sha256(shared / "initial.json"),
        ))

        navigation = output / "navigation"
        navigation.mkdir()
        small_arena = navigation / specs[0]["geometry"]
        small_arena.mkdir()
        for name in ("map.yaml", "map.pgm", "candidates.json", "graph.json"):
            shutil.copy2(VERIFIED / name, small_arena / name)
        shutil.copy2(NAV2 / "worlds/world.sdf", small_arena / "world.sdf")
        large_arena = navigation / specs[2]["geometry"]
        make_arena(large_arena, large, settings)
        # Preserve the verified waypoint pattern in the larger field frame.
        scaled_points = reference_points * (np.asarray(large) / np.asarray(small))
        write_json(large_arena / "candidates.json", scaled_points.tolist())
        run_inputs = output / "scenarios"
        run_inputs.mkdir()
        for spec in specs:
            case = run_inputs / spec["id"]
            case.mkdir()
            write_json(case / "scenario.json", spec)
            budget = spec["mission_budget_s"]
            if budget % settings["scoring_time_step_s"]:
                raise ValueError("The mission horizon must end on the scoring time grid")
            write_json(case / "protocol.json", dict(
                protocol, extent_m=spec["extent_m"], navigation_budget_s=budget,
            ))
        output.rename(bundle)
    print(f"Prepared {bundle} with a {mean:g} C GP prior and zero initial observations")
