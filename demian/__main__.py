"""Package entrypoint for `python -m demian`."""
from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "demian.py"),
    run_name="__main__",
)
