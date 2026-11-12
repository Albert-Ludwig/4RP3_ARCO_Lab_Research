"""Locations shared by both experiments; numerical/data settings stay separate."""

from pathlib import Path

SYNTHETIC_NAME = "3D_time&spatial temp field rebuild & prediction"
DASHBOARD_NAME = "dashboard time temp prediction"

SYNTHETIC_CODE = Path(__file__).resolve().parent
CODE_ROOT = SYNTHETIC_CODE.parent
EXPERIMENT_ROOT = CODE_ROOT.parent
WORKSPACE_ROOT = EXPERIMENT_ROOT.parent.parent
RESULT_ROOT = EXPERIMENT_ROOT / "10.16_result"
SYNTHETIC_RESULT = RESULT_ROOT / SYNTHETIC_NAME
DASHBOARD_CODE = CODE_ROOT / DASHBOARD_NAME
DASHBOARD_RESULT = RESULT_ROOT / DASHBOARD_NAME
