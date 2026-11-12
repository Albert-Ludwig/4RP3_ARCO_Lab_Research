"""Read Intel records and select actual observations across days and sensors."""

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
BASE = ROOT / "testing_code/baseline"
sys.path.insert(0, str(BASE))
from experiment import save_csv, save_json
from interfaces import Measurement

DATA = WORKSPACE / "Data/Intel_lab_data"
PROCESSED = DATA / "processed/three_day_2000"
ORIGIN = datetime(2004, 2, 29)
TRAIN_END_H = 72.0
TEST_END_H = 96.0


def read_records():
    locations = {int(m): (float(x), float(y))
                 for m, x, y in np.loadtxt(DATA / "raw/mote_locs.txt")}
    records, counts = [], Counter()
    with (DATA / "raw/data.txt").open() as stream:
        for row_id, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split(" ")
            if len(fields) != 8 or not "2004-02-29" <= fields[0] < "2004-03-04":
                continue
            counts["raw_window_rows"] += 1
            if not fields[3] or not fields[4]:
                counts["missing_sensor_or_temperature"] += 1
                continue
            try:
                mote, temp = int(fields[3]), float(fields[4])
                time = datetime.fromisoformat(fields[0] + "T" + fields[1])
            except ValueError:
                counts["parse_failure"] += 1
                continue
            if mote not in locations:
                counts["unknown_coordinates"] += 1
                continue
            # Fixed, explicit indoor plausibility screen. No thresholds are
            # fitted on test labels, and no missing value is imputed.
            if not np.isfinite(temp) or not 0 <= temp <= 50:
                counts["temperature_outside_fixed_0_50_c"] += 1
                continue
            hour = (time - ORIGIN).total_seconds() / 3600
            x, y = locations[mote]
            records.append(dict(record_id=row_id, sensor_id=mote, x_m=x, y_m=y,
                                t_h=hour, timestamp=time.isoformat(sep=" "),
                                temperature_c=temp, source="intel_lab", status="ok"))
    records.sort(key=lambda r: (r["t_h"], r["sensor_id"], r["record_id"]))
    return records, dict(counts)


def evenly_spaced(group, count, lower, upper):
    """Match equal-time targets to distinct real records; do not interpolate."""
    times = np.array([r["t_h"] for r in group])
    selected = set()
    for target in lower + (np.arange(count) + .5) * (upper - lower) / count:
        rank = np.argsort(np.abs(times - target), kind="stable")
        index = next(int(i) for i in rank if int(i) not in selected)
        selected.add(index)
    return [group[i] for i in sorted(selected)]


def prepare():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    records, counts = read_records()
    history = [r for r in records if r["t_h"] < TRAIN_END_H]
    test = [r for r in records if TRAIN_END_H <= r["t_h"] < TEST_END_H]
    rng, selected, allocation = np.random.default_rng(18), [], []
    day_counts = [667, 667, 666]
    for day, quota in enumerate(day_counts):
        groups = defaultdict(list)
        for row in history:
            if day * 24 <= row["t_h"] < (day + 1) * 24:
                groups[row["sensor_id"]].append(row)
        sensors = sorted(groups)
        extra = set(rng.permutation(sensors)[:quota % len(sensors)].tolist())
        for sensor in sensors:
            n = quota // len(sensors) + int(sensor in extra)
            assert len(groups[sensor]) >= n
            selected.extend(evenly_spaced(groups[sensor], n, day * 24, (day + 1) * 24))
            allocation.append(dict(day=day + 1, sensor_id=sensor,
                                   available=len(groups[sensor]), selected=n))
    selected.sort(key=lambda r: (r["t_h"], r["sensor_id"]))
    assert len(selected) == len({r["record_id"] for r in selected}) == 2000
    assert max(r["t_h"] for r in selected) < min(r["t_h"] for r in test)
    save_csv(PROCESSED / "train_2000.csv", selected)
    save_csv(PROCESSED / "test_day4.csv", test)
    save_csv(PROCESSED / "train_allocation.csv", allocation)
    summary = dict(source_repository="https://github.com/linsea423/Intel_Lab_Data",
                   original_source="https://db.csail.mit.edu/labdata/labdata.html",
                   source_commit="f46a8102e0ae93c98969e919c0b68897a202e0f3",
                   origin="2004-02-29 00:00:00; original timezone unspecified",
                   train_interval="[2004-02-29 00:00, 2004-03-03 00:00)",
                   test_interval="[2004-03-03 00:00, 2004-03-04 00:00)",
                   train_count=len(selected), test_count=len(test),
                   clean_history_pool=len(history), train_sensor_count=len({r["sensor_id"] for r in selected}),
                   test_sensor_count=len({r["sensor_id"] for r in test}),
                   train_rows_per_day=day_counts,
                   sampling="Day quotas 667/667/666; near-equal sensor quotas per day; nearest distinct records to equally spaced time targets; seed 18 assigns remainder quotas.",
                   cleaning="Known coordinates, finite temperature, fixed 0 to 50 C plausibility screen. Preserve blanks when reading; no imputation, spatial scaling, or interpolation.",
                   counts=counts)
    save_json(PROCESSED / "split.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return selected, test


def measurements(rows):
    return [Measurement(r["x_m"], r["y_m"], r["t_h"], r["temperature_c"],
                        "intel_lab", "ok") for r in rows]


def load_split():
    import csv
    def load(name):
        with (PROCESSED / name).open() as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            for key in ["record_id", "sensor_id"]:
                row[key] = int(row[key])
            for key in ["x_m", "y_m", "t_h", "temperature_c"]:
                row[key] = float(row[key])
        return rows
    return load("train_2000.csv"), load("test_day4.csv")


if __name__ == "__main__":
    prepare()
