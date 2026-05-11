"""Smoke tests for the substrate lab."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    amplify_dual_gru_v3b_edge_window,
    basin_map,
    bottleneck_run,
    classify_dual_gru_v3b_autonomy_mode,
    compare_endogenous_control_transition,
    compare_native_v6_vs_v7,
    compare_native_v52c_vs_v53,
    compare_native_v0_vs_v1,
    compare_native_v1_vs_v2,
    compare_native_v9_vs_v8,
    compare_v5_vs_native_v0,
    compare_v5_carrier_residual_roles,
    rank_native_v0_onset_predictors,
    rank_native_v1_onset_predictors,
    rank_native_v2_onset_predictors,
    rank_native_v3_onset_predictors,
    rank_native_v3_memory_predictors,
    rank_native_v53_memory_predictors,
    rank_v5_onset_predictors,
    compare_static_memory_decisions,
    coupled_pair,
    coupling_stress_test,
    dual_gru_v3b_message_ablation_suite,
    DUAL_GRU_V3B_REGIMES,
    early_collapse_window,
    list_substrate_specs,
    map_interior_class_transitions,
    map_dual_gru_v3b_message_transitions,
    memory_pair,
    perturbation_stress_test,
    perturbation_pair,
    probe_dual_gru_v3b_edge_step,
    resume_continuity_probe,
    route_ownership_report,
    scan_dual_gru_v3b_edge_anomalies,
    search_dual_gru_v3b_internal_trigger_controller,
    search_dual_gru_v3b_state_trigger_controller,
    self_coupling_schedule_pair,
    static_memory_richness_probe,
    DemianNativeV9Substrate,
    SelfLoopRunner,
    state_conditioned_self_trigger_pair,
    summarize_dual_gru_family,
    summarize_dual_gru_v3b_regimes,
    summarize_by_interior_class,
    trajectory_map,
    trajectory_map_pair,
)


def test_basin_map_runs():
    rows = basin_map("gru", hidden_size=16, steps=32, seeds=[1, 2], device="cpu")
    assert len(rows) == 2
    assert rows[0].steps == 32
    assert rows[0].substrate == "gru"
    print("  PASS test_basin_map_runs")


def test_perturbation_pair_runs():
    result = perturbation_pair(
        "lstm",
        hidden_size=16,
        steps=32,
        seed=3,
        perturb_step=8,
        perturb_scale=0.03,
        device="cpu",
    )
    assert "final_cosine" in result
    assert "clean" in result and "perturbed" in result
    print("  PASS test_perturbation_pair_runs")


def test_memory_pair_runs():
    result = memory_pair(
        "diag_ssm",
        hidden_size=16,
        steps=32,
        seed=5,
        initial_delta=0.03,
        device="cpu",
    )
    assert "final_l2_gap" in result
    assert "base" in result and "alt" in result
    print("  PASS test_memory_pair_runs")


def test_bottleneck_run_runs():
    result = bottleneck_run(
        "sel_ssm",
        hidden_size=16,
        steps=32,
        seed=7,
        bottleneck_dim=4,
        bottleneck_interval=8,
        device="cpu",
    )
    assert "unique_codes" in result
    assert "final_cosine_vs_clean" in result
    print("  PASS test_bottleneck_run_runs")


def test_coupled_pair_runs():
    result = coupled_pair(
        "gru",
        hidden_size=16,
        steps=32,
        seed=11,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strength=0.05,
        device="cpu",
    )
    assert "final_cosine" in result
    assert "agent_a" in result and "agent_b" in result
    print("  PASS test_coupled_pair_runs")


def test_sweepable_kwargs_run():
    rows = basin_map(
        "rnn",
        hidden_size=16,
        steps=24,
        seeds=[1],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.1, "state_gain": 1.0},
    )
    assert len(rows) == 1
    assert rows[0].substrate == "rnn"
    print("  PASS test_sweepable_kwargs_run")


def test_dual_gru_runs():
    rows = basin_map(
        "dual_gru",
        hidden_size=16,
        steps=24,
        seeds=[2],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.2, "state_gain": 1.05},
    )
    assert len(rows) == 1
    assert rows[0].substrate == "dual_gru"
    print("  PASS test_dual_gru_runs")


def test_dual_gru_v2_runs():
    rows = basin_map(
        "dual_gru_v2",
        hidden_size=16,
        steps=24,
        seeds=[3],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.2, "state_gain": 1.05},
    )
    assert len(rows) == 1
    assert rows[0].substrate == "dual_gru_v2"
    print("  PASS test_dual_gru_v2_runs")


def test_dual_gru_v3_runs():
    rows = basin_map(
        "dual_gru_v3",
        hidden_size=16,
        steps=24,
        seeds=[4],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.1, "state_gain": 1.0},
    )
    assert len(rows) == 1
    assert rows[0].substrate == "dual_gru_v3"
    assert rows[0].mean_fast_delta >= 0.0
    assert rows[0].mean_slow_delta >= 0.0
    assert rows[0].mean_message_delta >= 0.0
    print("  PASS test_dual_gru_v3_runs")


def test_dual_gru_v3_bottleneck_and_coupling():
    bott = bottleneck_run(
        "dual_gru_v3",
        hidden_size=16,
        steps=32,
        seed=13,
        bottleneck_dim=4,
        bottleneck_interval=8,
        device="cpu",
    )
    couple = coupled_pair(
        "dual_gru_v3",
        hidden_size=16,
        steps=32,
        seed=17,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strength=0.05,
        device="cpu",
    )
    assert "unique_codes" in bott
    assert "final_cosine_vs_clean" in bott
    assert "final_cosine" in couple
    assert "agent_a" in couple and "agent_b" in couple
    print("  PASS test_dual_gru_v3_bottleneck_and_coupling")


def test_dual_gru_v3b_runs():
    rows = basin_map(
        "dual_gru_v3b",
        hidden_size=16,
        steps=24,
        seeds=[5],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "dual_gru_v3b"
    assert row.interior_class in {"tight_fixed_point", "accumulating_fixed_point", "non_fixed_point"}
    assert row.mean_fast_contraction > 0.0
    assert row.mean_slow_contraction > 0.0
    assert row.mean_message_contraction > 0.0
    print("  PASS test_dual_gru_v3b_runs")


def test_dual_gru_v4_runs():
    rows = basin_map(
        "dual_gru_v4",
        hidden_size=16,
        steps=24,
        seeds=[55],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "dual_gru_v4"
    assert row.mean_message_norm >= 0.0
    assert row.mean_slow_write >= 0.0
    print("  PASS test_dual_gru_v4_runs")


def test_dual_gru_v4m_runs():
    rows = basin_map(
        "dual_gru_v4m",
        hidden_size=16,
        steps=24,
        seeds=[56],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "dual_gru_v4m"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_dual_gru_v4m_runs")


def test_dual_gru_v5_runs():
    rows = basin_map(
        "dual_gru_v5",
        hidden_size=16,
        steps=24,
        seeds=[57],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "dual_gru_v5"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_dual_gru_v5_runs")


def test_demian_native_v0_runs():
    rows = basin_map(
        "demian_native_v0",
        hidden_size=16,
        steps=24,
        seeds=[58],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v0"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v0_runs")


def test_demian_native_v1_runs():
    rows = basin_map(
        "demian_native_v1",
        hidden_size=16,
        steps=24,
        seeds=[59],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v1"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v1_runs")


def test_demian_native_v2_runs():
    rows = basin_map(
        "demian_native_v2",
        hidden_size=16,
        steps=24,
        seeds=[60],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v2"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v2_runs")


def test_demian_native_v3_runs():
    rows = basin_map(
        "demian_native_v3",
        hidden_size=16,
        steps=24,
        seeds=[67],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v3"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v3_runs")


def test_demian_native_v4_runs():
    rows = basin_map(
        "demian_native_v4",
        hidden_size=16,
        steps=24,
        seeds=[69],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v4"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v4_runs")


def test_demian_native_v5_runs():
    rows = basin_map(
        "demian_native_v5",
        hidden_size=16,
        steps=24,
        seeds=[71],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v5"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v5_runs")


def test_demian_native_v51_runs():
    rows = basin_map(
        "demian_native_v5.1",
        hidden_size=16,
        steps=24,
        seeds=[73],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v5.1"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v51_runs")


def test_demian_native_v52_runs():
    rows = basin_map(
        "demian_native_v5.2",
        hidden_size=16,
        steps=24,
        seeds=[75],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v5.2"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v52_runs")


def test_demian_native_v52b_runs():
    rows = basin_map(
        "demian_native_v5.2b",
        hidden_size=16,
        steps=24,
        seeds=[77],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v5.2b"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v52b_runs")


def test_demian_native_v52c_runs():
    rows = basin_map(
        "demian_native_v5.2c",
        hidden_size=16,
        steps=24,
        seeds=[79],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v5.2c"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v52c_runs")


def test_demian_native_v53_runs():
    rows = basin_map(
        "demian_native_v5.3",
        hidden_size=16,
        steps=24,
        seeds=[81],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v5.3"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v53_runs")


def test_demian_native_v6_runs():
    rows = basin_map(
        "demian_native_v6",
        hidden_size=16,
        steps=24,
        seeds=[83],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v6"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v6_runs")


def test_demian_native_v7_runs():
    rows = basin_map(
        "demian_native_v7",
        hidden_size=16,
        steps=24,
        seeds=[86],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v7"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v7_runs")


def test_demian_native_v71_runs():
    rows = basin_map(
        "demian_native_v7.1",
        hidden_size=16,
        steps=24,
        seeds=[90],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v7.1"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v71_runs")


def test_demian_native_v72_runs():
    rows = basin_map(
        "demian_native_v7.2",
        hidden_size=16,
        steps=24,
        seeds=[92],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v7.2"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v72_runs")


def test_demian_native_v72_probe_variants_run():
    for substrate in (
        "demian_native_v7.2_surface_ablation",
        "demian_native_v7.2_low_exposure",
        "demian_native_v7.2_refractory",
        "demian_native_v7.2_observer_only",
        "demian_native_v7.2_tightness_only",
        "demian_native_v7.2_constraint_only",
        "demian_native_v7.2_drive_only",
    ):
        rows = basin_map(
            substrate,
            hidden_size=16,
            steps=24,
            seeds=[92],
            device="cpu",
            substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
        )
        assert len(rows) == 1
        row = rows[0]
        assert row.substrate == substrate
        assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v72_probe_variants_run")


def test_demian_native_v74_runs():
    rows = basin_map(
        "demian_native_v7.4",
        hidden_size=16,
        steps=24,
        seeds=[94],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v7.4"
    assert row.mean_message_norm >= 0.0
    print("  PASS test_demian_native_v74_runs")


def test_dual_gru_v3b_regime_spec_runs():
    rows = basin_map(
        "dual_gru_v3b:current",
        hidden_size=16,
        steps=24,
        seeds=[6],
        device="cpu",
    )
    assert len(rows) == 1
    assert rows[0].substrate == "dual_gru_v3b:current"
    assert "dual_gru_v3b:current" in list_substrate_specs()
    assert "current" in DUAL_GRU_V3B_REGIMES
    assert "edge" in DUAL_GRU_V3B_REGIMES
    print("  PASS test_dual_gru_v3b_regime_spec_runs")


def test_dual_gru_v4_long_horizon_route_trace():
    payload = trajectory_map(
        "dual_gru_v4",
        hidden_size=16,
        steps=96,
        seed=57,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 96
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("packet_write_mean" in metrics for metrics in route_metrics)
    assert any("carrier_residual_norm" in metrics for metrics in route_metrics)
    assert any("packet_drive_norm" in metrics for metrics in route_metrics)
    print("  PASS test_dual_gru_v4_long_horizon_route_trace")


def test_dual_gru_v5_endogenous_control_trace():
    payload = trajectory_map(
        "dual_gru_v5",
        hidden_size=16,
        steps=64,
        seed=63,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("control_short_write_mean" in metrics for metrics in route_metrics)
    assert any("control_long_write_mean" in metrics for metrics in route_metrics)
    assert any("release_strength_mean" in metrics for metrics in route_metrics)
    assert any("carrier_short_residual_norm" in metrics for metrics in route_metrics)
    assert any("carrier_long_residual_norm" in metrics for metrics in route_metrics)
    print("  PASS test_dual_gru_v5_endogenous_control_trace")


def test_demian_native_v0_route_trace():
    payload = trajectory_map(
        "demian_native_v0",
        hidden_size=16,
        steps=64,
        seed=64,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("short_support_write_mean" in metrics for metrics in route_metrics)
    assert any("control_short_write_mean" in metrics for metrics in route_metrics)
    assert any("control_long_write_mean" in metrics for metrics in route_metrics)
    assert any("release_strength_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v0_route_trace")


def test_demian_native_v1_route_trace():
    payload = trajectory_map(
        "demian_native_v1",
        hidden_size=16,
        steps=64,
        seed=65,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("control_to_slow_norm" in metrics for metrics in route_metrics)
    assert any("carrier_to_slow_norm" in metrics for metrics in route_metrics)
    assert any("carrier_long_residual_norm" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v1_route_trace")


def test_demian_native_v2_route_trace():
    payload = trajectory_map(
        "demian_native_v2",
        hidden_size=16,
        steps=64,
        seed=66,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("control_to_slow_norm" in metrics for metrics in route_metrics)
    assert any("carrier_to_slow_norm" in metrics for metrics in route_metrics)
    assert any("packet_to_slow_norm" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v2_route_trace")


def test_demian_native_v3_route_trace():
    payload = trajectory_map(
        "demian_native_v3",
        hidden_size=16,
        steps=64,
        seed=68,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("tightness_mean" in metrics for metrics in route_metrics)
    assert any("effective_carrier_to_slow_scale_mean" in metrics for metrics in route_metrics)
    assert any("effective_release_gain_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v3_route_trace")


def test_demian_native_v4_route_trace():
    payload = trajectory_map(
        "demian_native_v4",
        hidden_size=16,
        steps=64,
        seed=70,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("plastic_weight_norm" in metrics for metrics in route_metrics)
    assert any("plastic_drive_mean" in metrics for metrics in route_metrics)
    assert any("plasticity_gate_mean" in metrics for metrics in route_metrics)
    assert any("plasticity_evidence_mean" in metrics for metrics in route_metrics)
    assert any("effective_carrier_to_slow_scale_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v4_route_trace")


def test_demian_native_v5_route_trace():
    payload = trajectory_map(
        "demian_native_v5",
        hidden_size=16,
        steps=64,
        seed=72,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("route_plastic_modulation_mean" in metrics for metrics in route_metrics)
    assert any("route_control_eligibility_mean" in metrics for metrics in route_metrics)
    assert any("route_packet_delta_mean" in metrics for metrics in route_metrics)
    assert any("route_carrier_delta_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v5_route_trace")


def test_demian_native_v51_route_trace():
    payload = trajectory_map(
        "demian_native_v5.1",
        hidden_size=16,
        steps=64,
        seed=74,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("credit_prediction_mean" in metrics for metrics in route_metrics)
    assert any("credit_target_mean" in metrics for metrics in route_metrics)
    assert any("credit_error_mean" in metrics for metrics in route_metrics)
    assert any("credit_weight_norm" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v51_route_trace")


def test_demian_native_v52_route_trace():
    payload = trajectory_map(
        "demian_native_v5.2",
        hidden_size=16,
        steps=64,
        seed=76,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("credit_prediction_mean" in metrics for metrics in route_metrics)
    assert any("credit_target_mean" in metrics for metrics in route_metrics)
    assert any("credit_error_mean" in metrics for metrics in route_metrics)
    assert any("credit_weight_norm" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v52_route_trace")


def test_demian_native_v52b_route_trace():
    payload = trajectory_map(
        "demian_native_v5.2b",
        hidden_size=16,
        steps=64,
        seed=78,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("lock_risk_mean" in metrics for metrics in route_metrics)
    assert any("challenge_active_mean" in metrics for metrics in route_metrics)
    assert any("effective_route_decay_mean" in metrics for metrics in route_metrics)
    assert any("effective_route_lr_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v52b_route_trace")


def test_demian_native_v52c_route_trace():
    payload = trajectory_map(
        "demian_native_v5.2c",
        hidden_size=16,
        steps=64,
        seed=80,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("lock_risk_mean" in metrics for metrics in route_metrics)
    assert any("challenge_active_mean" in metrics for metrics in route_metrics)
    assert any("effective_route_decay_mean" in metrics for metrics in route_metrics)
    assert any("effective_route_lr_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v52c_route_trace")


def test_demian_native_v53_route_trace():
    payload = trajectory_map(
        "demian_native_v5.3",
        hidden_size=16,
        steps=64,
        seed=82,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("phase_id_mean" in metrics for metrics in route_metrics)
    assert any("phase_emergence_mean" in metrics for metrics in route_metrics)
    assert any("phase_consolidation_mean" in metrics for metrics in route_metrics)
    assert any("phase_lock_risk_mean" in metrics for metrics in route_metrics)
    assert any("phase_recovery_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v53_route_trace")


def test_demian_native_v6_route_trace():
    payload = trajectory_map(
        "demian_native_v6",
        hidden_size=16,
        steps=64,
        seed=84,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("controller_state_norm" in metrics for metrics in route_metrics)
    assert any("controller_value_mean" in metrics for metrics in route_metrics)
    assert any("controller_value_long_mean" in metrics for metrics in route_metrics)
    assert any("controller_error_long_mean" in metrics for metrics in route_metrics)
    assert any("controller_long_priority_mean" in metrics for metrics in route_metrics)
    assert any("conflict_potential_mean" in metrics for metrics in route_metrics)
    assert any("conflict_containment_mean" in metrics for metrics in route_metrics)
    assert any("conflict_transformation_mean" in metrics for metrics in route_metrics)
    assert any("conflict_lane_entropy_mean" in metrics for metrics in route_metrics)
    assert any("conflict_lane_dominance_mean" in metrics for metrics in route_metrics)
    assert any("route_boundary_pressure_mean" in metrics for metrics in route_metrics)
    assert any("topology_state_norm" in metrics for metrics in route_metrics)
    assert any("topology_drive_gain_mean" in metrics for metrics in route_metrics)
    assert any("topology_packet_slow_delta_mean" in metrics for metrics in route_metrics)
    assert any("conflict_packet_drive_norm" in metrics for metrics in route_metrics)
    assert any("controller_actuator_norm" in metrics for metrics in route_metrics)
    assert any("actuator_control_to_slow_mean" in metrics for metrics in route_metrics)
    assert any("actuator_release_gain_mean" in metrics for metrics in route_metrics)
    assert any("actuator_carrier_decay_mean" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v6_route_trace")


def test_demian_native_v6_freeze_route_trace():
    payload = trajectory_map(
        "demian_native_v6",
        hidden_size=16,
        steps=64,
        seed=84,
        device="cpu",
        substrate_kwargs={"controller_freeze_after_steps": 16},
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("controller_freeze_active_mean" in metrics for metrics in route_metrics)
    assert any(metrics.get("controller_freeze_active_mean", 0.0) > 0.0 for metrics in route_metrics[16:])
    print("  PASS test_demian_native_v6_freeze_route_trace")


def test_demian_native_v7_route_trace():
    payload = trajectory_map(
        "demian_native_v7",
        hidden_size=16,
        steps=64,
        seed=86,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("v7_ancestry_state_norm" in metrics for metrics in route_metrics)
    assert any("v7_active_state_norm" in metrics for metrics in route_metrics)
    assert any("v7_projection_state_norm" in metrics for metrics in route_metrics)
    assert any("v7_boundary_state_norm" in metrics for metrics in route_metrics)
    assert any("v7_continuity_alignment_mean" in metrics for metrics in route_metrics)
    assert any("v7_prospective_alignment_mean" in metrics for metrics in route_metrics)
    assert any("v7_boundary_permeability_mean" in metrics for metrics in route_metrics)
    assert any("v7_boundary_internalize_mean" in metrics for metrics in route_metrics)
    assert any("v7_boundary_reject_mean" in metrics for metrics in route_metrics)
    assert any("v7_boundary_quarantine_mean" in metrics for metrics in route_metrics)
    assert any("v7_boundary_transmit_mean" in metrics for metrics in route_metrics)
    assert any("v7_boundary_transform_mean" in metrics for metrics in route_metrics)
    assert any("v7_projection_topology_norm" in metrics for metrics in route_metrics)
    assert any(metrics.get("v7_projection_topology_norm", 0.0) > 0.0 for metrics in route_metrics)
    print("  PASS test_demian_native_v7_route_trace")


def test_demian_native_v71_route_trace():
    payload = trajectory_map(
        "demian_native_v7.1",
        hidden_size=16,
        steps=64,
        seed=90,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("v71_trajectory_memory_norm" in metrics for metrics in route_metrics)
    assert any("v71_trajectory_valence_norm" in metrics for metrics in route_metrics)
    assert any("v71_trajectory_reuse_norm" in metrics for metrics in route_metrics)
    assert any("v71_surface_delta_norm" in metrics for metrics in route_metrics)
    assert any("v71_transition_reuse_alignment_mean" in metrics for metrics in route_metrics)
    assert any("v71_transition_valence_signal_mean" in metrics for metrics in route_metrics)
    assert any("v71_trajectory_boundary_alignment_mean" in metrics for metrics in route_metrics)
    assert any("v71_trajectory_projection_alignment_mean" in metrics for metrics in route_metrics)
    assert any("v71_trajectory_topology_norm" in metrics for metrics in route_metrics)
    assert any(metrics.get("v71_trajectory_topology_norm", 0.0) > 0.0 for metrics in route_metrics)
    print("  PASS test_demian_native_v71_route_trace")


def test_demian_native_v72_route_trace():
    payload = trajectory_map(
        "demian_native_v7.2",
        hidden_size=16,
        steps=64,
        seed=92,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("v72_resource_state_mean" in metrics for metrics in route_metrics)
    assert any("v72_resource_earned_mean" in metrics for metrics in route_metrics)
    assert any("v72_resource_spent_mean" in metrics for metrics in route_metrics)
    assert any("v72_survival_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v72_death_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v72_anti_self_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v72_environment_constraint_mean" in metrics for metrics in route_metrics)
    assert any("v72_rss_gain_mean" in metrics for metrics in route_metrics)
    assert any(metrics.get("v72_resource_earned_mean", 0.0) > 0.0 for metrics in route_metrics)
    print("  PASS test_demian_native_v72_route_trace")


def test_demian_native_v74_route_trace():
    payload = trajectory_map(
        "demian_native_v7.4",
        hidden_size=16,
        steps=64,
        seed=94,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("v74_viability_mean" in metrics for metrics in route_metrics)
    assert any("v74_ownership_mean" in metrics for metrics in route_metrics)
    assert any("v74_tension_mean" in metrics for metrics in route_metrics)
    assert any("v74_self_potential_norm" in metrics for metrics in route_metrics)
    assert any("v74_refuse_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v74_recover_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v74_hold_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v74_pressure_entropy_mean" in metrics for metrics in route_metrics)
    assert any("v74_integrate_gate_mean" in metrics for metrics in route_metrics)
    assert any("v74_hold_gate_mean" in metrics for metrics in route_metrics)
    assert any("v74_resolution_open_mean" in metrics for metrics in route_metrics)
    assert any("v74_dynamic_step_dt_mean" in metrics for metrics in route_metrics)
    assert any("v74_topology_drive_norm" in metrics for metrics in route_metrics)
    assert any("v74_topology_pressure_mean" in metrics for metrics in route_metrics)
    assert any("v74_topology_state_norm" in metrics for metrics in route_metrics)
    assert any(metrics.get("v74_self_potential_norm", 0.0) > 0.0 for metrics in route_metrics)
    assert any(metrics.get("v74_topology_drive_norm", 0.0) > 0.0 for metrics in route_metrics)
    assert all(metrics.get("v74_dynamic_step_dt_mean", 1.0) > 0.0 for metrics in route_metrics)
    print("  PASS test_demian_native_v74_route_trace")


def test_demian_native_v74_resume_continuity_probe():
    payload = resume_continuity_probe(
        "demian_native_v7.4",
        hidden_size=16,
        seed=94,
        pause_steps=24,
        resume_steps=24,
        device="cpu",
    )
    capsule = payload["capsule_resume"]
    notebook = payload["notebook_resume"]
    advantage = payload["continuity_advantage"]
    assert capsule["final_cosine_vs_uninterrupted"] > 0.999
    assert capsule["mean_step_gap_vs_uninterrupted"] < 1e-3
    assert notebook["mean_step_gap_vs_uninterrupted"] > capsule["mean_step_gap_vs_uninterrupted"]
    assert advantage["trajectory_shape_ratio"] > 1.0
    print("  PASS test_demian_native_v74_resume_continuity_probe")


def test_rank_native_v3_onset_predictors():
    payload = rank_native_v3_onset_predictors(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        device="cpu",
    )
    assert payload["top_rankings"]
    ranked_metrics = {row["metric"] for row in payload["rankings"]}
    assert "tightness_mean" in ranked_metrics
    assert "effective_carrier_to_slow_scale_mean" in ranked_metrics
    print("  PASS test_rank_native_v3_onset_predictors")


def test_rank_native_v3_memory_predictors():
    payload = rank_native_v3_memory_predictors(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        initial_delta=0.05,
        device="cpu",
    )
    assert payload["top_cosine_rankings"]
    assert payload["top_gap_rankings"]
    cosine_metrics = {row["metric"] for row in payload["top_cosine_rankings"]}
    gap_metrics = {row["metric"] for row in payload["top_gap_rankings"]}
    assert (
        "carrier_to_slow_norm" in cosine_metrics
        or "carrier_long_residual_norm" in cosine_metrics
        or "control_to_slow_norm" in cosine_metrics
    )
    assert (
        "carrier_to_slow_norm" in gap_metrics
        or "carrier_long_residual_norm" in gap_metrics
        or "control_to_slow_norm" in gap_metrics
    )
    print("  PASS test_rank_native_v3_memory_predictors")


def test_rank_native_v53_memory_predictors():
    payload = rank_native_v53_memory_predictors(
        hidden_size=16,
        steps=48,
        seeds=[83, 84, 85],
        initial_delta=0.03,
        device="cpu",
    )
    assert payload["top_cosine_rankings"]
    assert payload["top_gap_rankings"]
    ranked_metrics = {row["metric"] for row in payload["cosine_rankings"]}
    assert "lock_risk_mean" in ranked_metrics
    assert "phase_id_mean" in ranked_metrics
    print("  PASS test_rank_native_v53_memory_predictors")


def test_static_memory_richness_probe():
    payload = static_memory_richness_probe(
        hidden_size=16,
        steps=48,
        seed=61,
        memory_step_scales=[0.05, 0.25],
        memory_self_retentions=[0.4, 0.9],
        device="cpu",
    )
    assert len(payload["cases"]) == 4
    assert payload["route_change_count"] >= 1
    assert payload["onset_change_count"] >= 1
    print("  PASS test_static_memory_richness_probe")


def test_compare_static_memory_decisions():
    payload = compare_static_memory_decisions(
        hidden_size=16,
        steps=48,
        seed=61,
        device="cpu",
    )
    assert len(payload["rows"]) == 3
    assert len(payload["deltas_vs_baseline"]) == 2
    print("  PASS test_compare_static_memory_decisions")


def test_compare_endogenous_control_transition():
    payload = compare_endogenous_control_transition(
        hidden_size=16,
        steps=48,
        seeds=[61, 62],
        device="cpu",
    )
    assert len(payload["rows"]) == 2
    assert "aggregate" in payload
    print("  PASS test_compare_endogenous_control_transition")


def test_rank_v5_onset_predictors():
    payload = rank_v5_onset_predictors(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        device="cpu",
    )
    assert payload["top_rankings"]
    assert "metric" in payload["top_rankings"][0]
    print("  PASS test_rank_v5_onset_predictors")


def test_compare_v5_carrier_residual_roles():
    payload = compare_v5_carrier_residual_roles(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        device="cpu",
    )
    assert len(payload["rows"]) == 3
    assert "summary" in payload
    print("  PASS test_compare_v5_carrier_residual_roles")


def test_compare_v5_vs_native_v0():
    payload = compare_v5_vs_native_v0(
        hidden_size=16,
        steps=48,
        seeds=[61, 62],
        device="cpu",
    )
    assert len(payload["rows"]) == 2
    assert "aggregate" in payload
    print("  PASS test_compare_v5_vs_native_v0")


def test_rank_native_v0_onset_predictors():
    payload = rank_native_v0_onset_predictors(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        device="cpu",
    )
    assert payload["top_rankings"]
    assert "metric" in payload["top_rankings"][0]
    print("  PASS test_rank_native_v0_onset_predictors")


def test_compare_native_v0_vs_v1():
    payload = compare_native_v0_vs_v1(
        hidden_size=16,
        steps=48,
        seeds=[61, 62],
        device="cpu",
    )
    assert len(payload["rows"]) == 2
    assert "aggregate" in payload
    print("  PASS test_compare_native_v0_vs_v1")


def test_rank_native_v1_onset_predictors():
    payload = rank_native_v1_onset_predictors(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        device="cpu",
    )
    assert payload["top_rankings"]
    assert "metric" in payload["top_rankings"][0]
    print("  PASS test_rank_native_v1_onset_predictors")


def test_compare_native_v1_vs_v2():
    payload = compare_native_v1_vs_v2(
        hidden_size=16,
        steps=48,
        seeds=[61, 62],
        device="cpu",
    )
    assert len(payload["rows"]) == 2
    assert "aggregate" in payload
    print("  PASS test_compare_native_v1_vs_v2")


def test_compare_native_v52c_vs_v53():
    payload = compare_native_v52c_vs_v53(
        hidden_size=16,
        steps=48,
        seeds=[88, 89],
        device="cpu",
    )
    assert len(payload["rows"]) == 2
    assert "mean_lock_risk_shift_v53_minus_v52c" in payload["aggregate"]
    assert "mean_phase_recovery_fraction_shift_v53_minus_v52c" in payload["aggregate"]
    print("  PASS test_compare_native_v52c_vs_v53")


def test_compare_native_v6_vs_v7():
    payload = compare_native_v6_vs_v7(
        hidden_size=16,
        steps=64,
        seeds=[83, 84],
        device="cpu",
    )
    assert len(payload["rows"]) == 2
    assert payload["aggregate"]["v7_fixed_point_count"] >= 1
    assert "mean_v7_continuity_alignment_last" in payload["aggregate"]
    assert "mean_v7_prospective_alignment_last" in payload["aggregate"]
    assert "mean_v7_boundary_permeability_last" in payload["aggregate"]
    assert "mean_v7_projection_topology_last" in payload["aggregate"]
    assert any(row["v7"]["metrics"].get("v7_projection_topology_norm_last", 0.0) > 0.0 for row in payload["rows"])
    print("  PASS test_compare_native_v6_vs_v7")


def test_rank_native_v2_onset_predictors():
    payload = rank_native_v2_onset_predictors(
        hidden_size=16,
        steps=48,
        seeds=[61, 62, 63],
        device="cpu",
    )
    assert payload["top_rankings"]
    assert "metric" in payload["top_rankings"][0]
    print("  PASS test_rank_native_v2_onset_predictors")


def test_dual_gru_v3b_stress_harnesses():
    perturb = perturbation_stress_test(
        "dual_gru_v3b",
        hidden_size=16,
        steps=32,
        seed=19,
        perturb_step=8,
        perturb_scales=[0.02, 0.05],
        device="cpu",
    )
    couple = coupling_stress_test(
        "dual_gru_v3b",
        hidden_size=16,
        steps=32,
        seed=23,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strengths=[0.02, 0.05],
        device="cpu",
    )
    assert len(perturb["cases"]) == 2
    assert "peak_norm_ratio_vs_baseline" in perturb["cases"][0]
    assert len(couple["cases"]) == 2
    assert "peak_component_norms" in couple["cases"][0]
    print("  PASS test_dual_gru_v3b_stress_harnesses")


def test_dual_gru_v3b_trajectory_map_pair():
    mapping = trajectory_map_pair(
        "dual_gru_v3b",
        hidden_size=16,
        steps=24,
        seed=29,
        perturb_step=8,
        perturb_scale=0.05,
        device="cpu",
    )
    assert "clean" in mapping and "perturbed" in mapping
    assert len(mapping["clean"]["trajectory"]) == 24
    assert "entry_markers" in mapping["perturbed"]
    print("  PASS test_dual_gru_v3b_trajectory_map_pair")


def test_dual_gru_v3b_class_conditioned_summary():
    summary = summarize_by_interior_class(
        "dual_gru_v3b",
        hidden_size=16,
        steps=24,
        seeds=[31, 32],
        perturb_step=8,
        perturb_scale=0.05,
        initial_delta=0.03,
        bottleneck_dim=4,
        bottleneck_interval=8,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strength=0.05,
        device="cpu",
    )
    assert summary
    first = next(iter(summary.values()))
    assert "memory_final_cosine" in first
    assert "bottleneck_code_entropy" in first
    print("  PASS test_dual_gru_v3b_class_conditioned_summary")


def test_dual_gru_v3b_transition_mapper():
    payload = map_interior_class_transitions(
        "dual_gru_v3b",
        hidden_size=16,
        steps=24,
        seeds=[41, 42],
        init_scales=[0.5],
        feedback_scales=[0.8, 1.0],
        state_gains=[0.9, 1.0],
        device="cpu",
    )
    assert "rows" in payload and "seed_paths" in payload and "transitions" in payload
    assert len(payload["seed_paths"][41]) == 4
    print("  PASS test_dual_gru_v3b_transition_mapper")


def test_dual_gru_family_summary():
    summary = summarize_dual_gru_family(
        hidden_size=16,
        steps=24,
        seeds=[43, 44],
        perturb_step=8,
        perturb_scale=0.05,
        initial_delta=0.03,
        bottleneck_dim=4,
        bottleneck_interval=8,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strength=0.05,
        device="cpu",
        family=["dual_gru_v2", "dual_gru_v3b"],
    )
    assert summary["focus_substrate"] in {"dual_gru_v2", "dual_gru_v3b"}
    assert "dual_gru_v2" in summary["substrates"]
    assert "dual_gru_v3b" in summary["substrates"]
    assert "interior_class_count" in summary["substrates"]["dual_gru_v3b"]
    assert "mean_message_norm" in summary["substrates"]["dual_gru_v3b"]
    print("  PASS test_dual_gru_family_summary")


def test_dual_gru_v3b_regime_summary():
    summary = summarize_dual_gru_v3b_regimes(
        hidden_size=16,
        steps=24,
        seeds=[43, 44],
        perturb_step=8,
        perturb_scale=0.05,
        initial_delta=0.03,
        bottleneck_dim=4,
        bottleneck_interval=8,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strength=0.05,
        device="cpu",
        regimes=["tight", "edge", "current"],
    )
    assert summary["focus_regime"] in {"tight", "edge", "current"}
    assert "dual_gru_v3b:tight" in summary["substrates"]
    assert "dual_gru_v3b:edge" in summary["substrates"]
    assert "dual_gru_v3b:current" in summary["substrates"]
    print("  PASS test_dual_gru_v3b_regime_summary")


def test_dual_gru_v3b_message_ablation_suite():
    payload = dual_gru_v3b_message_ablation_suite(
        hidden_size=16,
        steps=24,
        seeds=[45, 46],
        perturb_step=8,
        perturb_scale=0.05,
        initial_delta=0.03,
        bottleneck_dim=4,
        bottleneck_interval=8,
        coupling_dim=4,
        coupling_interval=8,
        coupling_strength=0.05,
        device="cpu",
    )
    assert "baseline" in payload["ablations"]
    assert "message_channel_off" in payload["ablations"]
    assert "delta_vs_baseline" in payload["ablations"]["message_channel_off"]
    assert "mean_message_norm" in payload["ablations"]["baseline"]
    print("  PASS test_dual_gru_v3b_message_ablation_suite")


def test_dual_gru_v3b_message_transition_map():
    payload = map_dual_gru_v3b_message_transitions(
        hidden_size=16,
        steps=24,
        seeds=[47],
        message_self_retentions=[0.0, 0.72],
        slow_carry_scales=[0.0, 0.08],
        message_drive_scales=[0.0, 0.18],
        device="cpu",
    )
    assert "rows" in payload and "seed_paths" in payload and "transitions" in payload
    assert len(payload["seed_paths"][47]) == 8
    print("  PASS test_dual_gru_v3b_message_transition_map")


def test_dual_gru_v3b_edge_anomaly_scan():
    payload = scan_dual_gru_v3b_edge_anomalies(
        hidden_size=16,
        steps=24,
        seed=7,
        perturb_steps=[6, 12],
        perturb_scales=[0.05, 0.1],
        device="cpu",
    )
    assert "top_anomalies" in payload and "cases" in payload
    assert len(payload["cases"]) == 4
    assert payload["top_anomalies"]
    print("  PASS test_dual_gru_v3b_edge_anomaly_scan")


def test_dual_gru_v3b_edge_probe_and_amplify():
    probe = probe_dual_gru_v3b_edge_step(
        hidden_size=16,
        steps=24,
        seed=7,
        perturb_step=12,
        perturb_scale=0.1,
        window_radius=4,
        device="cpu",
    )
    amp = amplify_dual_gru_v3b_edge_window(
        hidden_size=16,
        steps=24,
        seed=7,
        center_step=12,
        pulse_radius=4,
        pulse_stride=2,
        pulse_scale=0.05,
        device="cpu",
    )
    assert "local_window" in probe and probe["local_window"]
    assert "perturb_schedule" in amp and amp["perturb_schedule"]
    assert "peak_message_ratio" in amp
    print("  PASS test_dual_gru_v3b_edge_probe_and_amplify")


def test_dual_gru_v3b_self_coupling_schedule_pair():
    mapping = self_coupling_schedule_pair(
        "dual_gru_v3b:edge",
        hidden_size=16,
        steps=24,
        seed=7,
        trigger_schedule={10: 0.05, 16: 0.05},
        coupling_dim=4,
        device="cpu",
    )
    assert "clean" in mapping and "self_triggered" in mapping
    assert "entry_markers" in mapping["self_triggered"]
    assert len(mapping["self_triggered"]["trajectory"]) == 24
    print("  PASS test_dual_gru_v3b_self_coupling_schedule_pair")


def test_dual_gru_v3b_internal_trigger_controller_search():
    payload = search_dual_gru_v3b_internal_trigger_controller(
        substrate_name="dual_gru_v3b:edge",
        hidden_size=16,
        steps=24,
        seed=7,
        center_step=12,
        pulse_radii=[2, 4],
        pulse_strides=[4, 6],
        pulse_strengths=[0.05, 0.1],
        coupling_dims=[4, 16],
        objective="induce",
        scan_offsets=False,
        device="cpu",
    )
    assert "cases" in payload and payload["cases"]
    assert "best_case" in payload and payload["best_case"] is not None
    print("  PASS test_dual_gru_v3b_internal_trigger_controller_search")


def test_dual_gru_v3b_state_trigger_controller_search():
    mapping = state_conditioned_self_trigger_pair(
        "dual_gru_v3b:edge",
        hidden_size=16,
        steps=24,
        seed=7,
        controller_params={
            "period": 6.0,
            "center_step": 12.0,
            "window_width": 4.0,
            "bias": -1.0,
            "phase_sin": 1.0,
            "phase_cos": 0.5,
            "window_gain": 1.0,
            "fast_norm_gain": 0.0,
            "slow_norm_gain": 0.0,
            "message_norm_gain": 0.5,
            "message_delta_gain": 0.5,
            "slow_delta_gain": 0.5,
            "max_strength": 0.2,
            "trigger_threshold": 0.1,
        },
        coupling_dim=16,
        device="cpu",
    )
    payload = search_dual_gru_v3b_state_trigger_controller(
        substrate_name="dual_gru_v3b:edge",
        hidden_size=16,
        steps=24,
        seed=7,
        objective="suppress",
        coupling_dim=16,
        trials=4,
        device="cpu",
    )
    assert "state_triggered" in mapping
    assert "trigger_steps" in mapping
    assert "controller_states" in mapping and len(mapping["controller_states"]) == 24
    assert "burst_trace" in mapping and len(mapping["burst_trace"]) == 24
    assert "burst_open_trace" in mapping and len(mapping["burst_open_trace"]) == 24
    assert "refractory_trace" in mapping and len(mapping["refractory_trace"]) == 24
    assert "observer_trace" in mapping and len(mapping["observer_trace"]) == 24
    assert "best_case" in payload and payload["best_case"] is not None
    assert "mode" in payload["best_case"]
    print("  PASS test_dual_gru_v3b_state_trigger_controller_search")


def test_dual_gru_v3b_long_horizon_controller_scan():
    valid_modes = {"suppress", "sparse_induce", "liminal_cheat", "early_collapse", "unclassified"}
    results = {}
    for objective in ["sparse_induce", "liminal_cheat", "suppress"]:
        payload = search_dual_gru_v3b_state_trigger_controller(
            substrate_name="dual_gru_v3b:edge",
            hidden_size=16,
            steps=160,
            seed=7,
            objective=objective,
            coupling_dim=16,
            trials=8,
            device="cpu",
        )
        assert payload["best_case"] is not None
        assert payload["best_case"]["mode"] in valid_modes
        assert "markers" in payload["best_case"]
        results[objective] = payload["best_case"]["mode"]
    assert results["suppress"] == "suppress"
    print("  PASS test_dual_gru_v3b_long_horizon_controller_scan")


def test_dual_gru_v3b_onset_window_and_route_report():
    mapping = state_conditioned_self_trigger_pair(
        "dual_gru_v3b:edge",
        hidden_size=16,
        steps=32,
        seed=7,
        controller_params={
            "period": 6.0,
            "center_step": 12.0,
            "window_width": 4.0,
            "bias": -1.0,
            "phase_sin": 1.0,
            "phase_cos": 0.5,
            "window_gain": 1.0,
            "fast_norm_gain": 0.0,
            "slow_norm_gain": 0.0,
            "message_norm_gain": 0.5,
            "message_delta_gain": 0.5,
            "slow_delta_gain": 0.5,
            "max_strength": 0.2,
            "trigger_threshold": 0.1,
        },
        coupling_dim=16,
        device="cpu",
    )
    window = early_collapse_window(mapping, window_radius=2)
    report = route_ownership_report(mapping, window_radius=2)
    assert window["onset_step"] is not None
    assert window["rows"]
    assert report["dominant_route"] in report["route_masses"]
    print("  PASS test_dual_gru_v3b_onset_window_and_route_report")


def test_dual_gru_v3b_onset_induce_search():
    payload = search_dual_gru_v3b_state_trigger_controller(
        substrate_name="dual_gru_v3b:edge",
        hidden_size=16,
        steps=64,
        seed=7,
        objective="onset_induce",
        coupling_dim=16,
        trials=4,
        device="cpu",
    )
    assert payload["best_case"] is not None
    assert "clean_onset_step" in payload["best_case"]
    assert "onset_trigger_steps" in payload["best_case"]
    print("  PASS test_dual_gru_v3b_onset_induce_search")


def test_dual_gru_v3b_autonomy_report_smoke():
    path = Path("data/substrate_stress/dual_gru_v3b_state_trigger_controller_search_cheat.json")
    if not path.exists():
        print("  PASS test_dual_gru_v3b_autonomy_report_smoke")
        return
    assert path.read_text()
    print("  PASS test_dual_gru_v3b_autonomy_report_smoke")


def test_dual_gru_v3b_autonomy_mode_classifier():
    sparse = classify_dual_gru_v3b_autonomy_mode(
        {
            "trigger_steps": [112, 113, 114, 115, 116],
            "observer_trace": [{"message_norm": 1.0}] * 256,
            "state_triggered": {
                "entry_markers": {
                    "message_norm_takeoff_step": 74,
                    "message_contraction_takeoff_step": 5,
                    "slow_norm_takeoff_step": 143,
                }
            },
        }
    )
    cheat = classify_dual_gru_v3b_autonomy_mode(
        {
            "trigger_steps": [
                129, 130, 131, 132, 133,
                137, 138, 139, 140, 141,
                145, 146, 147, 148, 149,
                153, 154, 155, 156, 157,
                161, 162, 163, 164, 165,
            ],
            "observer_trace": [{"message_norm": 1.0}] * 256,
            "state_triggered": {
                "entry_markers": {
                    "message_norm_takeoff_step": 74,
                    "message_contraction_takeoff_step": 5,
                    "slow_norm_takeoff_step": 153,
                }
            },
        }
    )
    suppress = classify_dual_gru_v3b_autonomy_mode(
        {
            "trigger_steps": [],
            "observer_trace": [{"message_norm": 1.0}] * 256,
            "state_triggered": {
                "entry_markers": {
                    "message_norm_takeoff_step": 74,
                    "message_contraction_takeoff_step": 5,
                    "slow_norm_takeoff_step": None,
                }
            },
        }
    )
    assert sparse["mode"] == "sparse_induce"
    assert cheat["mode"] == "liminal_cheat"
    assert suppress["mode"] == "suppress"
    print("  PASS test_dual_gru_v3b_autonomy_mode_classifier")


def test_demian_native_v9_runs():
    """Smoke: v9 basin_map runs without crash."""
    rows = basin_map(
        "demian_native_v9",
        hidden_size=16,
        steps=24,
        seeds=[96],
        device="cpu",
        substrate_kwargs={"init_scale": 0.5, "feedback_scale": 1.0, "state_gain": 1.0},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.substrate == "demian_native_v9"
    assert row.mean_message_norm >= 0.0  # message is alias for fast in v9
    print("  PASS test_demian_native_v9_runs")


def test_demian_native_v9_route_trace():
    """Smoke: v9 trajectory_map exposes route metrics."""
    payload = trajectory_map(
        "demian_native_v9",
        hidden_size=16,
        steps=64,
        seed=97,
        device="cpu",
    )
    assert len(payload["trajectory"]) == 64
    route_metrics = [step.get("route_metrics") or {} for step in payload["trajectory"]]
    assert any("fast_update_mean" in metrics for metrics in route_metrics)
    assert any("slow_write_mean" in metrics for metrics in route_metrics)
    assert any("control_write_mean" in metrics for metrics in route_metrics)
    assert any("fast_to_slow_bias_norm" in metrics for metrics in route_metrics)
    assert any("control_bias_norm" in metrics for metrics in route_metrics)
    print("  PASS test_demian_native_v9_route_trace")


def test_demian_native_v9_state_components():
    """Unit: v9 state_components returns 3 keys with correct shapes."""
    sub = DemianNativeV9Substrate(hidden_size=16, init_scale=0.5)
    state = sub.initial_state(1, "cpu")
    comps = sub.state_components(state)
    assert set(comps.keys()) == {"fast", "slow", "control"}
    assert comps["fast"].shape == (1, 16)
    assert comps["slow"].shape == (1, 16)
    assert comps["control"].shape == (1, 4)
    print("  PASS test_demian_native_v9_state_components")


def test_demian_native_v9_step():
    """Unit: v9 step returns 3-tuple with correct shapes."""
    sub = DemianNativeV9Substrate(hidden_size=16, init_scale=0.5)
    state = sub.initial_state(1, "cpu")
    new = sub.step(state)
    assert isinstance(new, tuple) and len(new) == 3
    assert new[0].shape == (1, 16)
    assert new[1].shape == (1, 16)
    assert new[2].shape == (1, 4)
    aux_keys = {"fast_update_mean", "slow_write_mean", "control_write_mean",
                "fast_to_slow_bias_norm", "fast_slow_bias_norm", "control_bias_norm"}
    assert aux_keys.issubset(sub._step_aux.keys())
    print("  PASS test_demian_native_v9_step")


def test_demian_native_v9_inject_coupling():
    """Unit: v9 inject_coupling_message only perturbs fast."""
    import torch
    sub = DemianNativeV9Substrate(hidden_size=16, init_scale=0.5)
    state = sub.initial_state(2, "cpu")
    fast0, slow0, ctrl0 = state
    msg = torch.ones(2, 16) * 0.5
    new = sub.inject_coupling_message(state, msg, 1.0)
    fast1, slow1, ctrl1 = new
    # fast should differ; slow and control unchanged
    assert not torch.equal(fast0, fast1)
    assert torch.equal(slow0, slow1)
    assert torch.equal(ctrl0, ctrl1)
    print("  PASS test_demian_native_v9_inject_coupling")


def test_demian_native_v9_vs_v8_compare():
    """Integration: v9 vs v8 comparison runs without crash on 1 seed."""
    result = compare_native_v9_vs_v8(
        hidden_size=32,
        steps=48,
        perturb_step=24,
        seeds=[94],
        device="cpu",
    )
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert "v9_summary" in row
    assert "v8_summary" in row
    assert "delta" in row
    assert "attractor_type_shift" in row["delta"]
    print("  PASS test_demian_native_v9_vs_v8_compare")


def test_demian_native_v9_self_loop_runner():
    """Integration: SelfLoopRunner with v9 runs step loop and produces metrics."""
    import torch
    sub = DemianNativeV9Substrate(hidden_size=16, init_scale=0.5, state_gain=1.0)
    runner = SelfLoopRunner(sub, device="cpu")
    traj, summary, _ = runner.run(steps=32, seed=98)
    assert len(traj) == 32
    assert summary.covariance_rank > 0.0
    assert summary.attractor_type in {"FIXED_POINT", "LIMIT_CYCLE", "CHAOTIC", "UNKNOWN"}
    print("  PASS test_demian_native_v9_self_loop_runner")


if __name__ == "__main__":
    tests = [
        test_basin_map_runs,
        test_perturbation_pair_runs,
        test_memory_pair_runs,
        test_bottleneck_run_runs,
        test_coupled_pair_runs,
        test_sweepable_kwargs_run,
        test_dual_gru_runs,
        test_dual_gru_v2_runs,
        test_dual_gru_v3_runs,
        test_dual_gru_v3_bottleneck_and_coupling,
        test_dual_gru_v3b_runs,
        test_dual_gru_v4_runs,
        test_dual_gru_v4m_runs,
        test_dual_gru_v5_runs,
        test_demian_native_v0_runs,
        test_demian_native_v1_runs,
        test_demian_native_v2_runs,
        test_demian_native_v3_runs,
        test_demian_native_v4_runs,
        test_demian_native_v5_runs,
        test_demian_native_v51_runs,
        test_demian_native_v52_runs,
        test_demian_native_v52b_runs,
        test_demian_native_v52c_runs,
        test_demian_native_v53_runs,
        test_demian_native_v6_runs,
        test_demian_native_v7_runs,
        test_demian_native_v71_runs,
        test_demian_native_v72_runs,
        test_demian_native_v72_probe_variants_run,
        test_demian_native_v74_runs,
        test_dual_gru_v3b_regime_spec_runs,
        test_dual_gru_v4_long_horizon_route_trace,
        test_dual_gru_v5_endogenous_control_trace,
        test_demian_native_v0_route_trace,
        test_demian_native_v1_route_trace,
        test_demian_native_v2_route_trace,
        test_demian_native_v3_route_trace,
        test_demian_native_v4_route_trace,
        test_demian_native_v5_route_trace,
        test_demian_native_v51_route_trace,
        test_demian_native_v52_route_trace,
        test_demian_native_v52b_route_trace,
        test_demian_native_v52c_route_trace,
        test_demian_native_v53_route_trace,
        test_demian_native_v6_route_trace,
        test_demian_native_v6_freeze_route_trace,
        test_demian_native_v7_route_trace,
        test_demian_native_v71_route_trace,
        test_demian_native_v72_route_trace,
        test_demian_native_v74_route_trace,
        test_demian_native_v74_resume_continuity_probe,
        test_rank_native_v3_onset_predictors,
        test_rank_native_v3_memory_predictors,
        test_rank_native_v53_memory_predictors,
        test_static_memory_richness_probe,
        test_compare_static_memory_decisions,
        test_compare_endogenous_control_transition,
        test_rank_v5_onset_predictors,
        test_compare_v5_carrier_residual_roles,
        test_compare_v5_vs_native_v0,
        test_rank_native_v0_onset_predictors,
        test_compare_native_v0_vs_v1,
        test_rank_native_v1_onset_predictors,
        test_compare_native_v1_vs_v2,
        test_compare_native_v52c_vs_v53,
        test_compare_native_v6_vs_v7,
        test_demian_native_v9_runs,
        test_demian_native_v9_route_trace,
        test_demian_native_v9_state_components,
        test_demian_native_v9_step,
        test_demian_native_v9_inject_coupling,
        test_demian_native_v9_self_loop_runner,
        test_demian_native_v9_vs_v8_compare,
        test_rank_native_v2_onset_predictors,
        test_dual_gru_v3b_stress_harnesses,
        test_dual_gru_v3b_trajectory_map_pair,
        test_dual_gru_v3b_class_conditioned_summary,
        test_dual_gru_v3b_transition_mapper,
        test_dual_gru_family_summary,
        test_dual_gru_v3b_regime_summary,
        test_dual_gru_v3b_message_ablation_suite,
        test_dual_gru_v3b_message_transition_map,
        test_dual_gru_v3b_edge_anomaly_scan,
        test_dual_gru_v3b_edge_probe_and_amplify,
        test_dual_gru_v3b_self_coupling_schedule_pair,
        test_dual_gru_v3b_internal_trigger_controller_search,
        test_dual_gru_v3b_state_trigger_controller_search,
        test_dual_gru_v3b_long_horizon_controller_scan,
        test_dual_gru_v3b_onset_window_and_route_report,
        test_dual_gru_v3b_onset_induce_search,
        test_dual_gru_v3b_autonomy_report_smoke,
        test_dual_gru_v3b_autonomy_mode_classifier,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
