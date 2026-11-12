"""Refresh GP-only result tables and figures without retraining saved models."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiment_paths import (SYNTHETIC_CODE, SYNTHETIC_RESULT, DASHBOARD_CODE,
                              DASHBOARD_RESULT)
from gp_model import load_gp
from evaluation import distribution_metrics


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh():
    torch.set_num_threads(1)
    roots = [SYNTHETIC_RESULT, DASHBOARD_RESULT]
    protected = [p for root in roots for p in root.rglob("*") if p.is_file()
                 and (p.suffix in {".pt", ".npz"}
                      or p.name in {"train.csv", "validation.csv", "test.csv",
                                    "development_train.csv", "refit_train.csv"}
                      or p.name.startswith("WELLWestCampus_"))]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    counts = {}
    for root in roots:
        tables = list(root.rglob("metrics.csv"))
        if root == SYNTHETIC_RESULT:
            tables.append(root / "robustness_runs.csv")
        for path in tables:
            rows = [r for r in read_rows(path) if r["model"].endswith("_gp")]
            assert rows, path
            write_rows(path, rows)
        if root == DASHBOARD_RESULT:
            fields = ["source_row", "elapsed_h", "observed_c", "gp_mean_c",
                      "latent_variance_c2", "observation_variance_c2",
                      "lower_95_c", "upper_95_c"]
            for path in root.rglob("predictions.csv"):
                write_rows(path, [{k: r[k] for k in fields} for r in read_rows(path)])
            config = json.loads((DASHBOARD_CODE / "dashboard_config.json").read_text())
            (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        audited = 0
        for path in root.rglob("test_predictions.npz"):
            with np.load(path) as saved:
                model = load_gp(path.parent / "model.pt")
                query = saved["xyt"] if "xyt" in saved else saved["time_h"][:, None]
                pred = model.predict(query)
                for key in ["mean", "latent_var", "observed_var"]:
                    np.testing.assert_allclose(pred[key], saved[key], rtol=1e-9, atol=1e-10)
                actual = saved["observed"] if "observed" in saved else saved["observed_c"]
                values = distribution_metrics(actual, pred["mean"], pred["observed_var"])
                table = path.parent / "metrics.csv"
                if table.exists():
                    row = next(r for r in read_rows(table)
                               if r.get("target", "noisy_observation") == "noisy_observation")
                    for k, value in values.items():
                        np.testing.assert_allclose(value, float(row[k]), rtol=1e-8, atol=1e-9)
                audited += 1
        counts[root.name] = audited

    sys.path.insert(0, str(DASHBOARD_CODE))
    from dashboard_experiment import load_data, split_indices, plot_case, setup_style
    config = json.loads((DASHBOARD_RESULT / "config.json").read_text())
    source = DASHBOARD_RESULT / json.loads((DASHBOARD_RESULT / "data_audit.json").read_text())["source_name"]
    _, _, x, y, _ = load_data(source, config["temperature_column"])
    splits = split_indices(x, config)
    setup_style()
    for case, parts in splits.items():
        fitted = load_gp(DASHBOARD_RESULT / case / "model.pt")
        plot_case(case, fitted, x, y, parts, config, DASHBOARD_RESULT / "figures")

    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after, "Frozen model/data/prediction artifact changed"
    for root, code in [(SYNTHETIC_RESULT, SYNTHETIC_CODE), (DASHBOARD_RESULT, DASHBOARD_CODE)]:
        path = root / "verification.json"
        record = json.loads(path.read_text())
        record.pop("saved_gp_and_baseline_metrics_recomputed", None)
        record["gp_only_record_refresh"] = {
            "models_retrained": False, "frozen_models_data_predictions_unchanged": True,
            "protected_files_checked": sum(root in p.parents for p in protected),
            "checkpoints_reloaded_and_predictions_verified": counts[root.name],
            "gp_metrics_verified_against_saved_predictions": True,
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in code.glob("*.py")},
        }
        path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"frozen_artifacts_unchanged": len(protected),
                      "checkpoints_verified": counts}, indent=2))


if __name__ == "__main__":
    refresh()
