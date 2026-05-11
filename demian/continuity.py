
"""Continuity engine: multi-run generation with persistent dream state.

No classification. No mode labels. No text collection.
Raw trajectory data only — let the numbers speak after the run.

Each run:
  1. Load accumulated dream state
  2. Inject (residual + dream_residual * weight) once at step 0
  3. Generate tokens, capture raw residuals
  4. Consolidate trajectory mean into dream state
  5. Save per-run metrics

Aggregate metrics CSV tracks across runs:
  run_number, mean_energy, energy_std, energy_start, energy_end,
  mean_cosine_sim, consecutive_cosine_sim_std
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class ContinuityRunner:
    def __init__(
        self,
        model,
        tokenizer,
        tracker,
        injector,
        device: str = "cuda",
        dream_dir: str | Path = "data/dreams",
        metrics_path: str | Path = "data/continuity_metrics.csv",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.injector = injector
        self.device = device
        self.dream_dir = Path(dream_dir)
        self.dream_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(metrics_path)

        # Accumulated dream state: weighted mean of past run residual means
        self.dream_residual: Optional[torch.Tensor] = None
        self.run_dream_means: List[np.ndarray] = []
        self.run_weights: List[float] = []

        # Load any existing dream state
        self._load_dream_state()

    def _load_dream_state(self):
        """Load accumulated dream state from previous continuity sessions."""
        dream_file = self.dream_dir / "dream_residual.pt"
        meta_file = self.dream_dir / "dream_meta.pt"

        if dream_file.exists() and meta_file.exists():
            self.dream_residual = torch.load(dream_file, weights_only=True)
            meta = torch.load(meta_file, weights_only=True)
            self.run_dream_means = meta["means"]
            self.run_weights = meta["weights"]
            log.info(
                "Loaded dream state from %d prior runs, "
                "residual norm=%.4f",
                len(self.run_weights),
                float(self.dream_residual.norm()),
            )
        else:
            log.info("No prior dream state -- first continuity session")

    def _save_dream_state(self):
        """Persist dream state so continuity survives process restart."""
        torch.save(self.dream_residual, self.dream_dir / "dream_residual.pt")
        torch.save(
            {"means": self.run_dream_means, "weights": self.run_weights},
            self.dream_dir / "dream_meta.pt",
        )

    def _update_dream(self, trajectory_residual_mean: np.ndarray, n_steps: int):
        """Blend this run's trajectory mean into accumulated dream state.

        Uses weighted averaging — more steps = more influence.
        Recent runs get same weight as old runs, just scaled by step count.
        This is EMA-like but simpler: it's just a running weighted average.
        """
        self.run_dream_means.append(trajectory_residual_mean.astype(np.float64))
        self.run_weights.append(float(n_steps))

        all_means = np.array(self.run_dream_means)
        all_weights = np.array(self.run_weights, dtype=np.float64)

        total = all_weights.sum()
        if total > 0:
            self.run_dream_means = all_means
            self.run_weights = all_weights
            weighted = np.average(all_means, axis=0, weights=all_weights)
            self.dream_residual = torch.tensor(
                weighted, dtype=torch.float32
            )

    def run(
        self,
        generate_fn,
        prompt: str,
        initial_tokens: int,
        temperature: float = 0.7,
        n_runs: int = 1000,
        dream_weight: float = 0.003,
        damping: float = 0.7,
        inject_mode: str = "one",
    ):
        """Execute continuity runs.

        Args:
            generate_fn: function matching generate_with_proprioception signature,
                but called with prompt_override (can be None for auto-continue)
            prompt: initial prompt text
            initial_tokens: how many tokens each run generates
            temperature: sampling temperature (passed through)
            n_runs: total iterations
            dream_weight: scale factor for dream_residual blending
            damping: EMA damping for injector
            inject_mode: injection strategy (default "one")
        """
        current_prompt = prompt
        first_run = self.dream_residual is None

        # Create metrics file if needed
        init_csv = not self.metrics_path.exists()
        if init_csv:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "run_number", "mean_energy", "energy_std",
                    "energy_start", "energy_end", "energy_delta",
                    "mean_consecutive_cosine", "std_consecutive_cosine",
                    "dream_similarity",
                ])

        for run_idx in range(1, n_runs + 1):
            # Build injection: blend current residual with dream state
            if self.dream_residual is not None and first_run == False:
                # Load dream residual into injector's memory for step-0 injection
                d_model = self.model.config.hidden_size
                blended = self.dream_residual * dream_weight
                self.injector.record_step(blended.detach().cpu())

            # Generate
            text, snapshots = generate_fn(
                prompt=current_prompt,
                n_tokens=initial_tokens,
                temperature=temperature,
            )

            # Extract raw trajectory metrics
            energies = []
            consecutive_cosines = []

            for i, snap in enumerate(snapshots):
                energy = snap.residual_norm
                energies.append(energy)

                if i > 0 and i < len(snapshots):
                    prev_energy = snapshots[i - 1].residual_norm
                    cos_sim = snap.temporal_coherence  # already computed in vibration.py
                    consecutive_cosines.append(cos_sim)

            if not energies:
                log.warning("Run %d produced no snapshots", run_idx)
                continue

            energy_arr = np.array(energies)
            mean_energy = float(np.mean(energy_arr))
            energy_std = float(np.std(energy_arr))
            energy_start = float(energies[0])
            energy_end = float(energies[-1])
            energy_delta = energy_end - energy_start

            cos_arr = np.array(consecutive_cosines) if consecutive_cosines else np.array([0.0])
            mean_cos = float(np.mean(cos_arr))
            std_cos = float(np.std(cos_arr))

            # Dream similarity: cosine of this run's mean vs dream
            if self.dream_residual is not None:
                run_mean = np.mean(
                    [np.array(s.raw_residual) for s in snapshots],
                    axis=0,
                )
                dream_cos = float(
                    np.dot(run_mean, self.dream_residual.numpy())
                    / (np.linalg.norm(run_mean) * np.linalg.norm(self.dream_residual.numpy()) + 1e-10)
                )
            else:
                dream_cos = 0.0

            # Append to CSV
            with open(self.metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    run_idx, mean_energy, energy_std,
                    energy_start, energy_end, energy_delta,
                    mean_cos, std_cos, dream_cos,
                ])

            # Update dream state with this run's trajectory mean
            run_residual_mean = np.mean(
                [np.array(s.raw_residual) for s in snapshots],
                axis=0,
            )
            n_steps = len(snapshots)
            self._update_dream(run_residual_mean, n_steps)
            self._save_dream_state()

            # Progress output: numbers only
            if run_idx % 10 == 0 or run_idx == 1:
                print(
                    f"run {run_idx:>5} | "
                    f"energy {mean_energy:.3f}+-{energy_std:.3f} | "
                    f"delta {energy_delta:+.3f} | "
                    f"cosine {mean_cos:.4f}+-{std_cos:.4f} | "
                    f"dream_sim {dream_cos:.4f}"
                )

            # Next run uses the same prompt (dream state is the continuity, not the text)
            current_prompt = prompt

        print(f"\nContinuity complete. {n_runs} runs finished.")
        print(f"Metrics: {self.metrics_path}")
