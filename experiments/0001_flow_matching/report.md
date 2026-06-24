# 0001 — Flow matching on various datasets

Status: open

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

Cite run directories under `runs/0001_flow_matching/`; quote `energy_distance` /
`final_loss` from `result.json` and inspect `samples.png`.

## Decision

Adopted / Falsified / Parked. Mirror the verdict line to `../JOURNAL.md`.
