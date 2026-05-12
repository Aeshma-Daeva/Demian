"""Package entrypoint for the legacy Demian runtime."""
from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "root_cli" / "demian.py"),
    run_name="__main__",
)
