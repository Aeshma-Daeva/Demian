#!/usr/bin/env python3
"""Validate Demian research artifacts with Pydantic schemas."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.lab_schemas import validate_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSON artifact paths")
    args = parser.parse_args()
    for raw_path in args.paths:
        path = Path(raw_path)
        artifact = validate_artifact(path)
        print(f"{path}: {artifact.__class__.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
