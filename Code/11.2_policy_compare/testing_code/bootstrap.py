"""Read-only reuse of the 10.18 experiment; keep new outputs here."""

import hashlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORKSPACE = ROOT.parents[1]
OLD = ROOT.parent / "10.18_Robot_close_loop_system"
CHECKPOINT = OLD / "10.18_result/baseline/intel_lab/forecast/initial_gp.pt"
HISTORICAL = OLD / "10.18_result/baseline/intel_lab/replay"
CONFIG = HERE / "config.json"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "11.2_result/.cache/matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / "11.2_result/.cache/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / "11.2_result/.cache/jupyter"))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(OLD / "testing_code/baseline"))
sys.path.insert(1, str(OLD / "testing_code/intel_lab"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_inputs():
    files = list((OLD / "testing_code/baseline").glob("*.py"))
    files += [OLD / "testing_code/baseline/config.json"]
    files += [OLD / "testing_code/intel_lab" / name for name in ("data.py", "replay.py")]
    shared = ROOT.parent / "10.16_GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction"
    files += list(shared.glob("*.py"))
    data = WORKSPACE / "Data/Intel_lab_data/processed/three_day_2000"
    files += [data / name for name in ("train_2000.csv", "test_day4.csv", "split.json")]
    files += [CHECKPOINT, CHECKPOINT.parent / "hyperparameters.json"]
    files += [HISTORICAL / name for name in ("summary.json", "online_measurements.csv", "robot_trajectory.csv", "metrics.csv", "planned_routes.json")]
    return {str(p.relative_to(WORKSPACE)).replace("\\", "/"): sha256(p) for p in sorted(set(files))}
