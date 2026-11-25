"""Compute full and matched-support RMSE from stored missions with the same truth field."""

import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

from prepare import (
    FINAL_TEST, POLICIES, REPO, SCENARIO_FOLDERS, TemperatureField,
    experiment_directory, read_json, write_json,
)

sys.path.insert(0, str(REPO / "Code/Verification tests/Continuous_sensing_test/testing_code"))
from components import BatchGP


def gp_from(data):
    params = dict(
        lengthscales_m_m_s=data["length"], output_variance_c2=data["variance"],
        noise_variance_c2=data["noise"], mean_c=data["mean"],
    )
    ids = [str(i) for i in range(len(data["x"]))]
    return BatchGP(np.asarray(data["x"]), np.asarray(data["y"]), params, ids)


def points(extent, cell):
    grids = [np.arange(cell / 2, edge, cell) for edge in extent]
    x, y = np.meshgrid(*grids, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel()))


def rmse(gp, field, xy, duration, step):
    last_t = float(np.max(gp.x[:, 2], initial=-np.inf))
    squared = 0.0
    count, beyond = 0, 0
    for t in np.arange(0, duration + step / 2, step):
        for start in range(0, len(xy), 256):
            spatial = xy[start:start + 256]
            q = np.column_stack((spatial, np.full(len(spatial), t)))
            residual = gp.posterior(q)[0] - field.truth(q)
            squared += float(residual @ residual)
            count += len(q)
            if t > last_t:
                beyond += len(q)
    return dict(rmse_c=float(np.sqrt(squared / count)), queries=count, after_last_sample=beyond)


def evaluate(bundle):
    shared = bundle / "shared"
    field = TemperatureField(read_json(shared / "field.json"))
    settings = read_json(shared / "settings.json")
    specs = {case: read_json(bundle / "scenarios" / case / "scenario.json") for case in ("A", "B", "C")}
    space = settings["evaluation_cell_m"]
    step = settings["evaluation_time_step_s"]
    small = specs["A"]["extent_m"]
    short_time = specs["B"]["mission_budget_s"]
    long_time = specs["A"]["mission_budget_s"]
    results = {}
    for case, spec in specs.items():
        xy = points(spec["extent_m"], space)
        grids = {"full": (xy, spec["mission_budget_s"])}
        if case in ("A", "B"):
            grids["common_time"] = (xy, short_time)
        if case in ("A", "C"):
            inside = (xy < small).all(axis=1)
            grids["common_space"] = (xy[inside], long_time)
            if case == "C":
                grids["added_area"] = (xy[~inside], long_time)
        results[case] = {}
        for policy in POLICIES:
            run_path = FINAL_TEST / SCENARIO_FOLDERS[case] / "results" / bundle.name / policy
            status_file = run_path / "run_status.json"
            if not status_file.exists():
                results[case][policy] = {"status": "not_run"}
                continue
            status = read_json(status_file)
            row = {
                "status": status["status"],
                "within_mission_budget": status.get("within_mission_budget"),
                "home_returned": status.get("home_returned"),
            }
            if status["status"] == "accepted":
                gp = gp_from(read_json(run_path / "final_gp.json"))
                row["domains"] = {key: rmse(gp, field, *grid, step) for key, grid in grids.items()}
            results[case][policy] = row
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="final", help="Run name shared by A, B and C")
    args = parser.parse_args()
    bundle = experiment_directory(args.name)
    target = bundle / "evaluation.json"
    write_json(target, evaluate(bundle))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
