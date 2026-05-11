"""Tests for the v9 capsule-continuity probe."""

from development.probe_v9_capsule_continuity import (
    aggregate_sweep,
    default_v9_five_channel,
    parse_int_list,
    parse_window_list,
    run_capsule_probe,
)
from development.substrates.legacy import DemianNativeV9Substrate


def test_capsule_probe_canonical_v9_surface_replay_fails():
    payload = run_capsule_probe(
        "demian_native_v9",
        lambda hidden_size: DemianNativeV9Substrate(hidden_size),
        ("fast", "slow", "control"),
        hidden_size=8,
        seed=94,
        pause_steps=8,
        resume_steps=8,
        device_name="cpu",
    )

    arms = payload["arms"]
    assert arms["full_capsule"]["final_cosine_vs_uninterrupted"] > 0.999
    assert arms["surface_only"]["mean_step_gap_vs_uninterrupted"] > arms["full_capsule"]["mean_step_gap_vs_uninterrupted"]
    assert set(arms) >= {"fast_only", "slow_only", "control_only"}


def test_capsule_probe_v9_five_channel_component_arms():
    payload = run_capsule_probe(
        "v9_five_channel",
        default_v9_five_channel,
        ("fast", "slow", "control", "message", "carrier"),
        hidden_size=8,
        seed=94,
        pause_steps=8,
        resume_steps=8,
        device_name="cpu",
    )

    arms = payload["arms"]
    assert arms["full_capsule"]["final_cosine_vs_uninterrupted"] > 0.999
    assert arms["surface_only"]["mean_step_gap_vs_uninterrupted"] > arms["full_capsule"]["mean_step_gap_vs_uninterrupted"]
    assert set(arms) >= {"message_only", "carrier_only"}


def test_capsule_sweep_helpers_parse_and_aggregate():
    assert parse_int_list("94, 95,96") == [94, 95, 96]
    assert parse_window_list("8,12:16") == [(8, 8), (12, 16)]

    runs = [
        run_capsule_probe(
            "demian_native_v9",
            lambda hidden_size: DemianNativeV9Substrate(hidden_size),
            ("fast", "slow", "control"),
            hidden_size=8,
            seed=seed,
            pause_steps=8,
            resume_steps=8,
            device_name="cpu",
        )
        for seed in (94, 95)
    ]

    aggregate = aggregate_sweep(runs)["demian_native_v9"]
    assert aggregate["n_runs"] == 2
    assert aggregate["all_full_capsules_exact_or_near_exact"] is True
    assert aggregate["all_surface_only_worse_than_full_capsule"] is True
