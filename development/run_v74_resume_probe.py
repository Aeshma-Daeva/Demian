"""Run the v7.4 resume-continuity probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import resume_continuity_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", default="demian_native_v7.4")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--pause-steps", type=int, default=128)
    parser.add_argument("--resume-steps", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/substrate_lab/v74_resume_probe_20260429")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = resume_continuity_probe(
        substrate_name=args.substrate,
        hidden_size=args.hidden_size,
        seed=args.seed,
        pause_steps=args.pause_steps,
        resume_steps=args.resume_steps,
        device=args.device,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    capsule = payload["capsule_resume"]
    state_only = payload["state_only_resume"]
    body_notebook = payload["body_notebook_resume"]
    notebook = payload["notebook_resume"]
    advantage = payload["continuity_advantage"]
    print(f"capsule final cosine: {capsule['final_cosine_vs_uninterrupted']:.6f}")
    print(f"state-only final cosine: {state_only['final_cosine_vs_uninterrupted']:.6f}")
    print(f"body+notebook final cosine: {body_notebook['final_cosine_vs_uninterrupted']:.6f}")
    print(f"notebook final cosine: {notebook['final_cosine_vs_uninterrupted']:.6f}")
    print(f"trajectory shape ratio: {advantage['trajectory_shape_ratio']:.3f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
