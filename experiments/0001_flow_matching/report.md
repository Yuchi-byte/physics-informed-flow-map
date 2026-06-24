# 0001 — Flow matching on various datasets

Status: closed

## Hypothesis

Pure flow matching (mfm FM loss, t_cond=0) trains a velocity field that samples
the target distribution: 2D Gaussians (energy-distance gate) and MNIST (FM-loss
gate), through one generic loop with a swappable dataset registry.

## Setup

`run.py [variant]` — `FlowMatchingConfig` (dataset, steps, lr, model knobs)
drives: dataset registry → `build_model` (MLP for vectors, DiT for images) →
generic FM training (`flow_matching.train`) → ODE sampling
(`flow_matching.sample`). Datasets: `gaussians`, `mnist`. Stack: mfm
(interpolant/loss/sampler).

## Results

Run directory: `runs/0001_flow_matching/2026-06-24T204300Z`

- `energy_distance`: 0.009576 (gate: 0.5 — well within margin)
- `final_loss`: 8.677 (down from ~18.3 at step 0; ~2.1× reduction)
- `verdict`: pass

`samples.png` shows 8 distinct, well-separated blobs arranged on a ring with no
merged or missing modes. The fit is visually clean.

Loss decreased from ~18.3 (step 0) to ~8.7 (step 1999), confirming meaningful
learning. Energy distance of 0.0096 is 52× below the 0.5 gate, so no gate
recalibration was required.

## Decision

Status: closed

Adopted — 8 modes recovered, energy distance 0.0096 < 0.5 gate, verdict pass.
