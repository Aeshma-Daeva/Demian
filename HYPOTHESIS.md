# Hypothesis Status: Gate-State vs Ordinary Recurrence

Date: 2026-05-16

## Current Read

The latest completed truth campaign does not support the strong name "Gate-State Causal Propagation" as a distinct mechanism.

The better current interpretation is:

> The system shows structured recurrent history propagation, but the available evidence does not yet distinguish it cleanly from ordinary recurrent-state history or random/default five-channel profiles.

This does not mean "nothing is happening." It means the earlier diagnostic gate was too permissive to justify the stronger mechanism claim.

## 2026-05-16 Update: 27 More Track B Candidates Generated

The missing 27 Track B evolution runs have now been generated successfully.

Artifacts:

- `/media/xenith/Games/demian_track_b_27_20260516_parallel/`
- `/media/xenith/Games/demian_track_b_27_20260516_parallel/manifest_summary.json`
- `/media/xenith/Games/demian_track_b_27_20260516_parallel/manifest_summary.csv`
- `data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json`

Validation:

- 27 / 27 new replication jobs exited with status 0.
- 27 / 27 new runs wrote `archive.json`.
- The combined 30-candidate summary has 30 candidate paths and 0 missing paths.
- Rows 1-3 are the original held-out-confirmed candidates.
- Rows 4-30 are archive-best training-selected candidates and are not yet held-out-confirmed strict-profile positives.

Meaning:

The generation step succeeded. It gives the truth campaign the missing candidate pool it originally needed.

It does not yet prove the mechanism. The new 27 candidates were selected from their evolution archives by saved training `rank_score`, so they are candidate positives, not final positives. The honest status is:

> We now have enough Track B candidates to run the intended 30-candidate truth campaign, but the mechanism claim is still pending held-out validation.

Technical correction:

The quick manifest initially reported 27 / 27 `message`/`carrier` top-two training-channel rows. That was not a valid channel-dominance result. The selected archive rows have zero saved per-channel causal-divergence values, so the apparent order came from a tie/order artifact in the first manifest summarizer. The manifest and combined 30-candidate summary now mark channel dominance for rows 4-30 as unresolved until held-out diagnostics compute it.

What remains true:

- 27 / 27 additional evolution runs completed.
- 27 / 27 selected archive-best candidates are valid candidate files.
- Selection was by saved `rank_score`, not by held-out strict profile.
- The generated pool is now large enough for the intended 30-candidate truth campaign.

The sharper hypothesis is still the right next test:

> Candidates that pass the locked strict profile should show message/carrier dominance in multi-timepoint surgery. Candidates that fail should show reduced, absent, or time-shifted message/carrier dominance.

This is a stronger and cleaner claim than the original broad gate-state language.

## Technical Detail: 27-Run Generation Step

Run shape:

```text
replications: 27, numbered 4-30
population: 16
generations: 20
seeds: 2026051304-2026051330
eval seeds: 94,95
steps: 128
perturb step: 64
perturb scales: 0.35,0.7
perturb channels: fast
rank mode: native_emergence
native objective: gate_state
causal mode: gate_state_propagation
reproduction mode: native
default seed: disabled
elitism: disabled
random injection overlay: disabled
device: cpu
parallelism: 3 concurrent replications, 4 workers each
```

Selection rule:

For each new replication, choose the row in `archive.json` with maximum saved `rank_score`. This is a training/evolution archive selection rule. It is not the held-out strict-profile rule.

The `gate_state` native objective used by the evolution rank is:

```text
morphology
+ selected_causal_divergence_raw
+ 0.3 * release_geometric_event
+ 0.3 * phase_transition_score
```

where morphology is:

```text
internal_richness
+ 1.2 * channel_separation
+ 0.7 * mathematical_curiosity
+ geometric_coherence
```

Observed archive-best ranges across the 27 new selected rows:

```text
rank_score:              min 3.1112, mean 3.2907, max 3.6556
release_duty_cycle:      min 0.0625, mean 0.2249, max 0.6875
internal_richness:       min 0.9091, mean 0.9851, max 1.0000
channel_separation:      min 0.6140, mean 0.6734, max 0.6942
release_geometric_event: min 0.0083, mean 0.4602, max 1.8968
phase_transition_score:  min 0.0209, mean 0.3771, max 1.0403
```

Important limitation:

```text
saved per-channel causal divergences in the selected archive rows: all zero
gain_zero_clean_fraction in the selected archive rows: all zero
```

Therefore the new 27 are valid Track B archive-best candidates, but their message/carrier dominance must be measured by the next diagnostic campaign, not inferred from the archive manifest.

## 2026-05-16 Full 30-Candidate Gen016-Locked Truth Campaign

The full campaign was run with the strict-profile calibration locked to the first Track B row only:

```text
replication summary: data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json
output: /media/xenith/Games/gate_state_truth_campaign_20260516_30_gen016locked/
track_b_count: 30
calibration_track_b_count: 1
random controls: 1000
calibration random controls: 300
default/default-jitter controls: 100
calibration default/default-jitter controls: 30
surgery pause steps: 32,64,96
surgery splits: calibration,test
```

Before the full run, a gen016-only preflight confirmed that the locked thresholds were still exactly the original gen016 thresholds:

```json
{
  "min_gain_zero_divergence": 0.2417293204225006,
  "min_message_carrier_dominance": 0.44469043517608386,
  "min_route_divergence": 0.2417293204225006
}
```

The campaign code now has a hard guard: if no calibration Track B candidate passes the broad mechanism gate, it raises an error instead of silently producing infinite thresholds.

Top-line full-campaign result:

```text
decision: demote_to_ordinary_or_unselected_recurrence
held-out Track B strict pass rate: 0.0
held-out control strict pass rate: 0.025974025974025976
max plain baseline recovery gap: 0.0039061478739458254
```

Held-out test breakdown:

```text
track_b_positive:
  mechanism gate: 13/29 = 0.4482758620689655
  strict profile: 0/29 = 0.0
  mean message_carrier_dominance: 0.17313303172163025

random_unselected:
  mechanism gate: 440/700 = 0.6285714285714286
  strict profile: 20/700 = 0.02857142857142857
  mean message_carrier_dominance: 0.24731316107216858

default_jittered:
  mechanism gate: 59/70 = 0.8428571428571429
  strict profile: 0/70 = 0.0
  mean message_carrier_dominance: 0.3695976521845639
```

Meaning:

The gen016-locked strict profile did not generalize to the 29 held-out Track B candidates. Worse, held-out random controls had a nonzero strict pass rate while Track B had zero. Under the current locked thresholds, this argues against the strong Gate-State Causal Propagation claim and against the idea that this Track B archive-best population is enriched for the strict gen016 profile.

The result does not prove that no structured channel anatomy exists. It says the strict gen016 profile is not a stable out-of-sample Track B phenotype under this campaign. The next analysis question is narrower: inspect the multi-timepoint surgery artifacts to see whether any subset of the 13 broad-gate Track B passers still shows message/carrier anatomy despite failing strict thresholding.

## What The Campaign Found

The campaign used all Track B positives available locally:

- Track B positives: 3 total
- Calibration Track B: 1
- Held-out Track B: 2
- Random unselected controls: 1000
- Default/default-jitter controls: 100
- Total candidates characterized: 1103

The broad mechanism gate passed both held-out Track B candidates:

- Held-out Track B broad mechanism pass rate: 2 / 2

But the same broad gate also passed many controls:

- Held-out random controls: 440 / 700
- Held-out default-jitter controls: 59 / 70

So the broad gate is not rare or selective. It mostly detects that some recurrent/internal history effect exists.

## Locked Strict Test

The calibration Track B candidate set these locked strict thresholds:

```json
{
  "min_route_divergence": 0.2417293204225006,
  "min_gain_zero_divergence": 0.2417293204225006,
  "min_message_carrier_dominance": 0.44469043517608386
}
```

Held-out strict-profile result:

- Held-out Track B: 0 / 2
- Held-out random controls: 20 / 700
- Held-out all controls: 20 / 770

The held-out Track B candidates still passed the broad mechanism gate, but they did not preserve the strong message/carrier dominance profile from calibration.

## State Surgery

State surgery on held-out candidate `gen019_candidate008` did not favor the strongest message/carrier story:

- Message/carrier swap gap: 0.7265
- Slow/control swap gap: 0.8123

Slow/control swaps moved future dynamics more than message/carrier swaps in this run. That is evidence against claiming message/carrier as the privileged causal carrier without more proof.

## Improved Multi-Timepoint Surgery Result On The Original Three

The first state-surgery result was underdesigned because it used one candidate and emphasized one pause/final-gap view.

The improved multi-timepoint surgery was run on all three original Track B candidates, including calibration candidate `gen016_candidate012`.

Artifact:

`data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/`

Result:

- `gen016_candidate012`, the calibration candidate that passed the strict profile, showed message/carrier dominance at all 12 pause-window comparisons.
- `gen017_candidate000` showed message/carrier dominance in 11 / 12 comparisons.
- `gen019_candidate008` was mixed: message/carrier dominated most comparisons, but slow/control dominated pause 64 full/mid/late windows.

Meaning:

The improved surgery changes the interpretation. The original one-point surgery was not enough to dismiss the channel anatomy. The better read is now:

> There may be a real message/carrier anatomy, but the current evidence is still underpowered and candidate-specific until the 30-candidate campaign tests it out of sample.

## Meaning

The experiment was useful because it found a weakness in the earlier test.

The earlier gate should be treated as a screening diagnostic, not as proof of a distinct mechanism. It accepts many random/default-like systems, so it cannot support statistical rarity language or a strong named mechanism claim by itself.

Current claim discipline:

- Do not strengthen the paper.
- Do not claim statistical rarity from the broad gate.
- Do not claim a distinct gate-state mechanism yet.
- Rename/demote the working hypothesis to "structured recurrent history propagation" unless a stronger campaign reverses this.

## Important Caveat

The planned campaign wanted 30 independent Track B positives, but the first truth-campaign run had only 3 locally available candidates. That result was therefore not the final universal answer.

The missing 27 candidates now exist, but they still need held-out truth-campaign validation. The honest conclusion is:

> The strong Gate-State Causal Propagation claim is not established. The channel-anatomy hypothesis is alive, but promotion requires the 30-candidate held-out campaign to beat controls and baselines under locked thresholds.

## Next Better Test

Before revisiting the strong claim:

1. Use `data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json` as the 30-candidate Track B input.
2. Freeze thresholds before held-out testing.
3. Rerun the campaign with the intended 30 Track B candidates, 1000 random controls, and 100 default/jitter controls.
4. Require held-out Track B to beat controls on strict profile, message/carrier history probes, and multi-timepoint state surgery.
5. Split results by strict-profile pass/fail status and test whether passers show stronger message/carrier surgery dominance than failures.

Critical threshold-lock requirement:

The full 30-candidate campaign must keep the strict-profile thresholds locked from `gen016_candidate012`, not recalibrate them on the new 27 candidates. In the current campaign entrypoint, that means overriding the default calibration split:

```text
--calibration-track-b-count 1
```

The code default is `--calibration-track-b-count 10`; using that default would include new candidates in calibration and break the held-out logic for the strict-profile claim. Calibration random/default controls do not set the strict thresholds because threshold calibration filters to `family == track_b_positive` and `mechanism_gate_pass`, but keeping Track B calibration at 1 is mandatory for the gen016-locked test.

## Anatomy Reframe Test

The first campaign's state surgery compared only one pause point and mostly read the aggregate/final continuation gap. That can miss a real anatomical pattern where:

- message/carrier dominates early trajectory steering,
- slow/control dominates later trajectory stabilization,
- the original "Gate-State Causal Propagation" name is still too strong, but the five-channel anatomy is not arbitrary.

The campaign code now supports multi-timepoint state surgery through `--surgery-pause-steps`. For each pause point it records:

- full continuation gap,
- early-window gap,
- mid-window gap,
- late-window gap,
- message/carrier swap vs slow/control swap dominance.

This is the right test for the reframe:

> If message/carrier dominates early windows while slow/control dominates late windows, the channel anatomy is partly vindicated even if the original gate-state mechanism name remains demoted.

Small smoke artifact for the new surgery path:

`/tmp/demian_truth_campaign_multisurgery_smoke/`

Artifact directory from this run:

`data/diagnostics/gate_state_truth_campaign_20260515/`
