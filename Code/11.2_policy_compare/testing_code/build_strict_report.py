"""Compatibility entry point for the single consolidated report."""

import argparse
from pathlib import Path
from build_report import build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    build(parser.parse_args().output)
