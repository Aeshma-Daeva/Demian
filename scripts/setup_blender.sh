#!/usr/bin/env bash
set -euo pipefail

if command -v blender >/dev/null 2>&1; then
  echo "Blender already available:"
  blender --version | head -n 2
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required by this setup script." >&2
  echo "Install Blender manually, then verify with: blender --version" >&2
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script needs package-manager privileges." >&2
  echo "Run: sudo bash scripts/setup_blender.sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y blender

echo "Blender installed:"
blender --version | head -n 2
echo
echo "Renderer smoke command:"
echo "blender --background --python scripts/render_v9_expression_blender.py -- --input /tmp/v9_5ch_evo_traj_smoke/trajectory_3d.json --out-dir /tmp/v9_expr_blender --save-blend"
