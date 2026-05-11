#!/usr/bin/env python3
"""Render a deterministic Blender expression from v9 trajectory JSON.

Run inside Blender:
    blender --background --python scripts/render_v9_expression_blender.py -- \
      --input data/substrate_lab/.../trajectory_3d.json --out-dir /tmp/v9_expr

Outside Blender, ``--dry-run`` validates input and writes metadata only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover - exercised only without Blender.
    bpy = None
    Vector = None


def parse_args() -> argparse.Namespace:
    args = sys.argv
    if "--" in args:
        args = args[args.index("--") + 1:]
    else:
        args = args[1:]
    parser = argparse.ArgumentParser(description="Render v9 expression phenotype")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=12)
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--perturb-family", action="append", default=[])
    parser.add_argument("--perturb-scale", action="append", type=float, default=[])
    return parser.parse_args(args)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _matches_filters(point: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.candidate_id and str(point.get("candidate_id")) not in set(args.candidate_id):
        return False
    if args.seed and int(point.get("seed", -1)) not in set(args.seed):
        return False
    if args.perturb_family and str(point.get("perturb_family")) not in set(args.perturb_family):
        return False
    if args.perturb_scale:
        scale = float(point.get("perturb_scale", 0.0))
        if all(abs(scale - wanted) > 1e-9 for wanted in args.perturb_scale):
            return False
    return True


def group_points(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    metrics = {row["point_id"]: row for row in payload.get("metrics", [])}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in payload.get("points", []):
        if not _matches_filters(point, args):
            continue
        row = dict(point)
        row["metrics"] = metrics.get(point["point_id"], {})
        key = "|".join(
            str(row.get(name, ""))
            for name in ("candidate_id", "seed", "perturb_family", "motif_index", "perturb_scale")
        )
        groups[key].append(row)
    selected = dict(list(groups.items())[:args.max_runs])
    for rows in selected.values():
        rows.sort(key=lambda row: int(row["step"]))
    return selected


def expression_metadata(
    payload: dict[str, Any],
    groups: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "source_note": payload.get("metadata", {}).get("viewer_note", ""),
        "group_count": len(groups),
        "point_count": sum(len(rows) for rows in groups.values()),
        "filters": {
            "candidate_id": args.candidate_id,
            "seed": args.seed,
            "perturb_family": args.perturb_family,
            "perturb_scale": args.perturb_scale,
            "max_runs": args.max_runs,
        },
        "mapping": {
            "fast_surface": "trajectory curve position",
            "slow": "large form radius from slow_state_norm",
            "control": "emission/tension from release_pressure_mean",
            "message": "stroke density/material brightness from message_signature_tail",
            "carrier": "persistent tube thickness from carrier_signature_tail",
            "release": "sparse pulse spheres from release_strength_mean",
        },
        "organization": "Collections are grouped as v9_expression/candidate/seed/run/channel.",
        "groups": [
            {
                "key": key,
                "candidate_id": rows[0].get("candidate_id"),
                "perturb_family": rows[0].get("perturb_family"),
                "motif_index": rows[0].get("motif_index"),
                "steps": len(rows),
            }
            for key, rows in groups.items()
            if rows
        ],
    }


def reset_scene() -> None:
    assert bpy is not None
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 128


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def scale_name(value: Any) -> str:
    return safe_name(str(value).replace(".", "p"))


def get_child_collection(parent: Any, name: str, unique_name: str | None = None):
    assert bpy is not None
    for child in parent.children:
        if child.get("v9_display_name") == name or child.name == name:
            return child
    collection = bpy.data.collections.new(unique_name or name)
    collection["v9_display_name"] = name
    parent.children.link(collection)
    return collection


def move_to_collection(obj: Any, collection: Any) -> None:
    assert bpy is not None
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for current in list(obj.users_collection):
        if current != collection:
            current.objects.unlink(obj)


def run_collections(root: Any, rows: list[dict[str, Any]]):
    first = rows[0]
    candidate = safe_name(first.get("candidate_id", "unknown_candidate"))
    seed = f"seed_{int(first.get('seed', 0)):03d}"
    family = safe_name(first.get("perturb_family", "unknown_family"))
    motif = f"motif_{int(first.get('motif_index', 0)):02d}"
    scale = f"scale_{scale_name(first.get('perturb_scale', 'unknown'))}"
    run = f"{family}_{motif}_{scale}"
    candidate_col = get_child_collection(root, candidate)
    seed_col = get_child_collection(candidate_col, seed, f"{candidate}__{seed}")
    run_unique = f"{candidate}__{seed}__{run}"
    run_col = get_child_collection(seed_col, run, run_unique)
    return {
        "run": run_col,
        "carrier": get_child_collection(run_col, "carrier_path", f"{run_unique}__carrier_path"),
        "slow": get_child_collection(run_col, "slow_form", f"{run_unique}__slow_form"),
        "release": get_child_collection(run_col, "release_pulses", f"{run_unique}__release_pulses"),
        "message": get_child_collection(run_col, "message_traces", f"{run_unique}__message_traces"),
    }


def make_material(name: str, color: tuple[float, float, float, float], emission: float = 0.0):
    assert bpy is not None
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def render_scene(payload: dict[str, Any], groups: dict[str, list[dict[str, Any]]], out_dir: Path, save_blend: bool) -> None:
    assert bpy is not None and Vector is not None
    reset_scene()
    root_col = get_child_collection(bpy.context.scene.collection, "v9_expression")
    curve_mat = make_material("carrier_path", (0.30, 0.70, 1.0, 1.0), 0.1)
    message_mat = make_material("message_trace", (0.35, 0.95, 0.55, 1.0), 0.35)
    release_mat = make_material("release_pulse", (1.0, 0.70, 0.20, 1.0), 0.8)
    base_mat = make_material("slow_form", (0.72, 0.75, 0.80, 1.0), 0.0)

    max_extent = max(
        max(abs(float(point[axis])) for axis in ("x", "y", "z"))
        for rows in groups.values()
        for point in rows
    ) if groups else 1.0
    scale = 4.5 / max(max_extent, 1e-6)

    for group_index, (key, rows) in enumerate(groups.items()):
        if len(rows) < 2:
            continue
        collections = run_collections(root_col, rows)
        offset = Vector(((group_index % 4) * 3.0, (group_index // 4) * 3.0, 0.0))
        curve = bpy.data.curves.new(f"path_{group_index:02d}", type="CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 2
        curve.bevel_depth = 0.018
        poly = curve.splines.new("POLY")
        poly.points.add(len(rows) - 1)
        for idx, point in enumerate(rows):
            coords = Vector((float(point["x"]), float(point["y"]), float(point["z"]))) * scale + offset
            poly.points[idx].co = (coords.x, coords.y, coords.z, 1.0)
        obj = bpy.data.objects.new(f"carrier_path_{group_index:02d}", curve)
        obj.data.materials.append(curve_mat)
        collections["carrier"].objects.link(obj)

        tail = rows[-1]
        route = tail.get("metrics", {}).get("route_metrics", {})
        radius = 0.20 + 0.45 * min(1.0, float(route.get("internal_richness", 0.0)))
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32,
            ring_count=16,
            radius=radius,
            location=offset + Vector((0.0, 0.0, -0.75)),
        )
        base = bpy.context.object
        base.name = f"slow_form_{group_index:02d}"
        base.data.materials.append(base_mat)
        move_to_collection(base, collections["slow"])

        for point in rows:
            route = point.get("metrics", {}).get("route_metrics", {})
            release = float(route.get("release_strength_mean", 0.0))
            if release <= 0.005:
                continue
            coords = Vector((float(point["x"]), float(point["y"]), float(point["z"]))) * scale + offset
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=16,
                ring_count=8,
                radius=0.06 + min(0.30, release * 1.5),
                location=coords,
            )
            pulse = bpy.context.object
            pulse.name = f"release_pulse_{group_index:02d}_{int(point['step']):03d}"
            pulse.data.materials.append(release_mat)
            move_to_collection(pulse, collections["release"])

        for every, point in enumerate(rows[:: max(1, len(rows) // 16)]):
            route = point.get("metrics", {}).get("route_metrics", {})
            carrier = float(route.get("carrier_signature_tail", 0.0))
            if carrier <= 0.0:
                continue
            coords = Vector((float(point["x"]), float(point["y"]), float(point["z"]))) * scale + offset
            bpy.ops.mesh.primitive_cube_add(size=0.05 + min(0.18, carrier * 0.2), location=coords + Vector((0.0, 0.0, 0.12)))
            trace = bpy.context.object
            trace.name = f"message_trace_{group_index:02d}_{every:02d}"
            trace.data.materials.append(message_mat)
            move_to_collection(trace, collections["message"])

    bpy.ops.object.light_add(type="AREA", location=(2.5, -4.0, 7.0))
    bpy.context.object.name = "expression_area_light"
    bpy.context.object.data.energy = 450
    bpy.context.object.data.size = 5
    bpy.ops.object.camera_add(location=(5.0, -7.0, 5.0), rotation=(math.radians(60), 0.0, math.radians(38)))
    bpy.context.scene.camera = bpy.context.object
    if save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "expression.blend"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_payload(input_path)
    groups = group_points(payload, args)
    metadata = expression_metadata(payload, groups, args)
    (out_dir / "expression_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if bpy is None:
        if not args.dry_run:
            raise SystemExit("This script must run inside Blender, or use --dry-run for metadata validation.")
        print(out_dir / "expression_metadata.json")
        return
    render_scene(payload, groups, out_dir, args.save_blend)
    print(out_dir / "expression_metadata.json")


if __name__ == "__main__":
    main()
