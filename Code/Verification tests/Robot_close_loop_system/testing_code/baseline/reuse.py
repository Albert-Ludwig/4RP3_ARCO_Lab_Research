"""Locate the unchanged 10.16 baseline modules in this repository."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT.parent / "10.16_GP3D_baseline_test" / "10.16_testing_code" / "3D_time&spatial temp field rebuild & prediction"
if not (BASELINE / "gp_model.py").is_file():
    raise FileNotFoundError(f"Required baseline is missing: {BASELINE}")
sys.path.append(str(BASELINE))
