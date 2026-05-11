"""v8 vs v7.4 coupling stress: organ activation under environmental threat.

Hypothesis: v7.4 self-policy organs activate under high coupling stress.
v8 tightness modulates uniformly. If v7.4 gates differentiate under stress,
organs are functional, not decorative.
"""

import json
import os
import sys
import math
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
    DemianNativeV74Substrate,
    SelfLoopRunner,
    _make_substrate,
    _fixed_projection,
    _encode_state,
    _decode_code,
    _resolve_substrate_spec,
)

HIDDEN_SIZE = 32
STEPS = 256
SEEDS = [94, 95, 96, 97]
COUPLING_STRENGTHS = [0.0, 0.1, 0.3, 0.5]
COUPLING_DIM = 8
COUPLING_INTERVAL = 1
DEVICE = "cuda"
OUT_DIR = Path("data/substrate_lab/v8_v74_coupling_stress_interval1_20260506")

SUBSTRATES = {
    "demian_native_v8": DemianNativeV8Substrate,
    "demian_native_v7.4": DemianNativeV74Substrate,
}


def run_coupling_single(substrate_name, cls, seed, strength, steps=STEPS):
    """Run coupled pair, return per-step telemetry + summary."""
    torch.manual_seed(seed + 1000)
    model_a = cls(HIDDEN_SIZE)
    model_b = cls(HIDDEN_SIZE)
    model_a.to(DEVICE)
    model_b.to(DEVICE)
    model_a.eval()
    model_b.eval()

    runner_a = SelfLoopRunner(model_a, device=DEVICE)
    runner_b = SelfLoopRunner(model_b, device=DEVICE)

    # Use fixed proj with different seed offset so it's orthogonal to seed init
    proj = _fixed_projection(HIDDEN_SIZE, COUPLING_DIM, seed + 2000)

    state_a = model_a.initial_state(1, runner_a.device)
    state_b = model_b.initial_state(1, runner_b.device)

    # Per-step tracking
    cosines = []
    tightness_a_trace = []
    tightness_b_trace = []
    # v74 organ traces
    v74_gates_a = {k: [] for k in [
        "preserve", "adapt", "recover", "refuse", "quarantine",
        "integrate", "hold", "resolution_open", "step_dt"
    ]}
    v74_gates_b = {k: [] for k in v74_gates_a}
    v74_pressure_a = {k: [] for k in ["viability", "ownership", "tension",
                                        "integrate", "refuse", "recover", "reject",
                                        "hold", "entropy"]}
    v74_pressure_b = {k: [] for k in v74_pressure_a}
    v74_organ_norms_a = {k: [] for k in ["self_potential", "quarantine", "topology"]}
    v74_organ_norms_b = {k: [] for k in v74_organ_norms_a}

    with torch.no_grad():
        for step_idx in range(1, steps + 1):
            state_a = model_a.step(state_a)
            state_b = model_b.step(state_b)

            h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
            h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()

            # Track step aux
            step_aux_a = dict(getattr(model_a, '_step_aux', {}))
            step_aux_b = dict(getattr(model_b, '_step_aux', {}))

            # Tightness trace for v8
            if "tightness_mean" in step_aux_a:
                tightness_a_trace.append(step_aux_a["tightness_mean"])
            if "tightness_mean" in step_aux_b:
                tightness_b_trace.append(step_aux_b["tightness_mean"])

            # v74 organ gates
            if substrate_name == "demian_native_v7.4":
                for model_ref, gates, pressures, organ_norms, aux in [
                    (model_a, v74_gates_a, v74_pressure_a, v74_organ_norms_a, step_aux_a),
                    (model_b, v74_gates_b, v74_pressure_b, v74_organ_norms_b, step_aux_b),
                ]:
                    gates["preserve"].append(aux.get("v74_preserve_gate_mean", 0.0))
                    gates["adapt"].append(aux.get("v74_adapt_gate_mean", 0.0))
                    gates["recover"].append(aux.get("v74_recover_gate_mean", 0.0))
                    gates["refuse"].append(aux.get("v74_refuse_gate_mean", 0.0))
                    gates["quarantine"].append(aux.get("v74_quarantine_gate_mean", 0.0))
                    gates["integrate"].append(aux.get("v74_integrate_gate_mean", 0.0))
                    gates["hold"].append(aux.get("v74_hold_gate_mean", 0.0))
                    gates["resolution_open"].append(aux.get("v74_resolution_open_mean", 0.0))
                    gates["step_dt"].append(aux.get("v74_dynamic_step_dt_mean", 1.0))
                    pressures["viability"].append(aux.get("v74_viability_mean", 0.0))
                    pressures["ownership"].append(aux.get("v74_ownership_mean", 0.0))
                    pressures["tension"].append(aux.get("v74_tension_mean", 0.0))
                    pressures["integrate"].append(aux.get("v74_integrate_pressure_mean", 0.0))
                    pressures["refuse"].append(aux.get("v74_refuse_pressure_mean", 0.0))
                    pressures["recover"].append(aux.get("v74_recover_pressure_mean", 0.0))
                    pressures["reject"].append(aux.get("v74_reject_pressure_mean", 0.0))
                    pressures["hold"].append(aux.get("v74_hold_pressure_mean", 0.0))
                    pressures["entropy"].append(aux.get("v74_pressure_entropy_mean", 0.0))
                    organ_norms["self_potential"].append(aux.get("v74_self_potential_norm", 0.0))
                    organ_norms["quarantine"].append(aux.get("v74_quarantine_norm", 0.0))
                    organ_norms["topology"].append(aux.get("v74_topology_state_norm", 0.0))

            # Coupling injection at interval
            if strength > 0 and step_idx % COUPLING_INTERVAL == 0:
                code_a = _encode_state(h_a, proj)
                code_b = _encode_state(h_b, proj)
                msg_a = _decode_code(code_a, proj).to(runner_a.device, dtype=runner_a.dtype)
                msg_b = _decode_code(code_b, proj).to(runner_b.device, dtype=runner_b.dtype)
                state_a = model_a.inject_coupling_message(state_a, msg_b, strength)
                state_b = model_b.inject_coupling_message(state_b, msg_a, strength)

            cosines.append(float(torch.nn.functional.cosine_similarity(h_a, h_b, dim=0).item()))

    # Build summary
    summary = {
        "coupling_strength": strength,
        "n_steps": steps,
        "initial_cosine": cosines[0] if cosines else 0.0,
        "final_cosine": cosines[-1] if cosines else 0.0,
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "cosine_std": float(np.std(cosines)) if cosines else 0.0,
        "cosine_min": float(min(cosines)) if cosines else 0.0,
        "cosine_max": float(max(cosines)) if cosines else 0.0,
        "cosine_trace": cosines,
    }

    if tightness_a_trace:
        summary["tightness_a_mean"] = float(np.mean(tightness_a_trace))
        summary["tightness_a_std"] = float(np.std(tightness_a_trace))
        summary["tightness_a_final"] = tightness_a_trace[-1]
        summary["tightness_a_trace"] = tightness_a_trace
        summary["tightness_b_mean"] = float(np.mean(tightness_b_trace))
        summary["tightness_b_std"] = float(np.std(tightness_b_trace))
        summary["tightness_b_final"] = tightness_b_trace[-1]
        summary["tightness_b_trace"] = tightness_b_trace

    if substrate_name == "demian_native_v7.4":
        for prefix, gates_dict, pressures_dict, organ_norms_dict in [
            ("a", v74_gates_a, v74_pressure_a, v74_organ_norms_a),
            ("b", v74_gates_b, v74_pressure_b, v74_organ_norms_b),
        ]:
            for gate_name, trace in gates_dict.items():
                if trace:
                    summary["v74_gate_" + gate_name + "_" + prefix + "_mean"] = float(np.mean(trace))
                    summary["v74_gate_" + gate_name + "_" + prefix + "_std"] = float(np.std(trace))
                    summary["v74_gate_" + gate_name + "_" + prefix + "_final"] = trace[-1]
                    summary["v74_gate_" + gate_name + "_" + prefix + "_trace"] = trace

            for pressure_name, trace in pressures_dict.items():
                if trace:
                    summary["v74_pressure_" + pressure_name + "_" + prefix + "_mean"] = float(np.mean(trace))
                    summary["v74_pressure_" + pressure_name + "_" + prefix + "_std"] = float(np.std(trace))
                    summary["v74_pressure_" + pressure_name + "_" + prefix + "_final"] = trace[-1]

            for organ_name, trace in organ_norms_dict.items():
                if trace:
                    summary["v74_organ_" + organ_name + "_" + prefix + "_mean"] = float(np.mean(trace))
                    summary["v74_organ_" + organ_name + "_" + prefix + "_std"] = float(np.std(trace))
                    summary["v74_organ_" + organ_name + "_" + prefix + "_final"] = trace[-1]

    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, cls in SUBSTRATES.items():
        seed_results = {}
        for seed in SEEDS:
            print()
            print("name=" + str(name) + " seed=" + str(seed))
            cases = []
            for strength in COUPLING_STRENGTHS:
                print("  coupling=" + str(strength))
                summary = run_coupling_single(name, cls, seed, strength)
                cases.append(summary)
            seed_results[str(seed)] = {"cases": cases}
        results[name] = seed_results

    # Aggregate across seeds
    aggregate = {}
    for name in SUBSTRATES:
        seed_data = results[name]
        agg = {}
        for strength in COUPLING_STRENGTHS:
            metric_keys = [
                "mean_cosine", "final_cosine", "cosine_std",
                "tightness_a_mean", "tightness_a_final",
            ]
            if name == "demian_native_v7.4":
                metric_keys += [
                    "v74_gate_preserve_a_mean", "v74_gate_adapt_a_mean",
                    "v74_gate_recover_a_mean", "v74_gate_refuse_a_mean",
                    "v74_gate_quarantine_a_mean", "v74_gate_integrate_a_mean",
                    "v74_gate_hold_a_mean", "v74_gate_resolution_open_a_mean",
                    "v74_gate_step_dt_a_mean",
                    "v74_pressure_tension_a_mean", "v74_pressure_viability_a_mean",
                    "v74_pressure_ownership_a_mean", "v74_pressure_entropy_a_mean",
                    "v74_organ_self_potential_a_mean", "v74_organ_quarantine_a_mean",
                    "v74_organ_topology_a_mean",
                ]

            per_strength = {}
            for key in metric_keys:
                vals = []
                for s in SEEDS:
                    case = seed_data[str(s)]["cases"][COUPLING_STRENGTHS.index(strength)]
                    v = case.get(key)
                    if v is not None:
                        vals.append(v)
                if vals:
                    per_strength[key + "_mean"] = float(np.mean(vals))
                    per_strength[key + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            agg[str(strength)] = per_strength
        aggregate[name] = agg

    # Strip traces from disk output (keep traces in memory only)
    output_results = {}
    for name in results:
        output_results[name] = {}
        for seed, seed_data in results[name].items():
            output_results[name][seed] = {
                "cases": [
                    {k: v for k, v in case.items()
                     if not k.endswith("_trace")}
                    for case in seed_data["cases"]
                ]
            }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "steps": STEPS,
                "seeds": SEEDS,
                "coupling_strengths": COUPLING_STRENGTHS,
                "coupling_dim": COUPLING_DIM,
                "coupling_interval": COUPLING_INTERVAL,
                "device": DEVICE,
            },
            "results": output_results,
            "aggregate": aggregate,
        }, f, indent=2)
    print()
    print("Done: " + str(OUT_DIR / "summary.json"))

    # Print quick analysis
    print()
    print("=== Quick analysis ===")
    for name in SUBSTRATES:
        print()
        print("name=" + str(name))
        for strength in COUPLING_STRENGTHS:
            key = str(strength)
            if key in aggregate[name]:
                a = aggregate[name][key]
                line = "  coupling=" + str(strength) + ": "
                for k in ["mean_cosine", "final_cosine"]:
                    mean_k = k + "_mean"
                    std_k = k + "_std"
                    if mean_k in a:
                        line += k + "=" + str(round(a[mean_k], 4))
                        if std_k in a:
                            line += "+-" + str(round(a[std_k], 4))
                        line += " "
                tight_k = "tightness_a_mean_mean"
                if tight_k in a:
                    line += "tight=" + str(round(a[tight_k], 4))
                    tight_sk = "tightness_a_mean_std"
                    if tight_sk in a:
                        line += "+-" + str(round(a[tight_sk], 4))
                    line += " "
                if name == "demian_native_v7.4":
                    for g in ["preserve", "adapt", "recover", "refuse", "quarantine"]:
                        gate_key = "v74_gate_" + g + "_a_mean_mean"
                        if gate_key in a:
                            line += g[:3] + "=" + str(round(a[gate_key], 4)) + " "
                print(line)


if __name__ == "__main__":
    main()
