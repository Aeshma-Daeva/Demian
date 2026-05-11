#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="${1:-data/evolution/v9_5ch_release_20260509_full}"
TRAJ_DIR="${2:-data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full}"
INTERVAL="${INTERVAL:-3}"

while true; do
  clear
  date
  echo
  echo "== V9-5CH EVOLUTION =="
  if [[ -d "${ARCHIVE_DIR}" ]]; then
    echo "archive: ${ARCHIVE_DIR}"
    find "${ARCHIVE_DIR}/candidates" -type f 2>/dev/null | wc -l | awk '{print "candidates: "$1}'
    if [[ -f "${ARCHIVE_DIR}/generations.jsonl" ]]; then
      echo
      echo "-- latest generations --"
      tail -5 "${ARCHIVE_DIR}/generations.jsonl"
    fi
    if [[ -f "${ARCHIVE_DIR}/archive.json" ]]; then
      echo
      echo "-- archive candidates --"
      venv/bin/python - <<'PY' "${ARCHIVE_DIR}/archive.json"
import json, sys
from pathlib import Path
rows=json.loads(Path(sys.argv[1]).read_text())
for row in rows:
    m=row["metrics"]
    print(f"{row['id']} rank={row.get('rank_score',0):.4f} rich={m['internal_richness']:.4f} mem={m['memory_specificity']:.4f} rel={m['release_effectiveness']:.6f} duty={m['release_duty_cycle']:.4f}")
    print("  regimes", m["regimes"])
PY
    fi
  else
    echo "missing archive dir: ${ARCHIVE_DIR}"
  fi

  echo
  echo "== TRAJECTORY / BLENDER =="
  if [[ -f "${TRAJ_DIR}/trajectory_3d.json" ]]; then
    ls -lh "${TRAJ_DIR}/trajectory_3d.json"
    venv/bin/python - <<'PY' "${TRAJ_DIR}/trajectory_3d.json"
import json, sys
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text())
print(f"trajectory points={len(j['points'])} events={len(j['events'])} candidates={len(j['metadata'].get('substrates', []))}")
PY
  else
    echo "missing trajectory: ${TRAJ_DIR}/trajectory_3d.json"
  fi
  if [[ -d "${TRAJ_DIR}/blender_expression" ]]; then
    ls -lh "${TRAJ_DIR}/blender_expression"
    if [[ -f "${TRAJ_DIR}/blender_expression/expression_metadata.json" ]]; then
      venv/bin/python - <<'PY' "${TRAJ_DIR}/blender_expression/expression_metadata.json"
import json, sys
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text())
print(f"blender groups={j['group_count']} points={j['point_count']}")
PY
    fi
  fi

  echo
  echo "Refresh: ${INTERVAL}s. Ctrl-C to stop."
  sleep "${INTERVAL}"
done
