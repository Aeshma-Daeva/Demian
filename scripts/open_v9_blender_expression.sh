#!/usr/bin/env bash
set -euo pipefail

BLEND_FILE="${1:-data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/blender_expression/expression.blend}"

if [[ ! -f "${BLEND_FILE}" ]]; then
  echo "Missing Blender file: ${BLEND_FILE}" >&2
  exit 1
fi

exec blender "${BLEND_FILE}"
