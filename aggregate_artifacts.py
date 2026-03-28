"""Root entrypoint for artifact aggregation.

Usage:
  python aggregate_artifacts.py --strict
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script_path = Path(__file__).parent / "scripts" / "aggregate_artifacts.py"
    runpy.run_path(str(script_path), run_name="__main__")
