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

## Update 2026-06-25 — epoch loop + hyperparameter tuning

Re-ran on the epoch-based loop (`training.n_epochs`) + hierarchical configs, tuning
each variant on an RTX 5090 for *good* samples, not just a passing gate.

**Gaussians** (`runs/0001_flow_matching/2026-06-25T00-45-58Z`):
- Real-vs-real energy-distance noise floor (n=2048) measured at **~0.0050**.
- Sweep: 5 epochs → E=0.066; 30 epochs → 0.033; **100 epochs / 200 sampler steps → E=0.0104** (~2× floor).
- Samples: 8 tight, evenly-spaced, well-separated modes; none merged/missing/spurious.
- Locked into the `gaussians` variant: `training.n_epochs=100`, `sampling.sampler_steps=200`.

**MNIST** (`runs/0001_flow_matching/2026-06-25T00-53-06Z`, wandb run `dtcrah9l`):
- Baseline (DiT 128/4, 50 epochs, 50 sampler steps): final loss 115; digits readable but ~half malformed.
- **Tuned (DiT hidden=256/depth=6/heads=8, 100 epochs, 200 sampler steps): final loss ~107; clean recognizable digits across all 10 classes.**
- The quality jump came mainly from model capacity + sampler steps, not extra epochs.
- Locked into the `mnist` variant (`conf/model/dit.yaml` enlarged; `n_epochs=100`, `sampler_steps=200`).

**Next lever (not done):** EMA of the velocity weights (mfm has the machinery; our
minimal `train()` does not use it) is the standard trick that would most improve
MNIST sharpness further — worth a proper spec→plan→review pass rather than an
unattended change.

## Update 2026-06-25 — OpenFWI variant (velocity-map prior)

Added an unconditional flow-matching prior over OpenFWI **FlatVel_A** velocity maps
(`experiment=openfwi`, DiT 256/6/8 at 64×64, EMA on, 100 epochs). Run
`runs/0001_flow_matching/2026-06-25T14-18-57Z` (≈22 min on an RTX 5090, ~13.5 s/epoch).

- Samples reproduce the FlatVel_A structure cleanly: flat horizontal layers with
  velocity increasing with depth (dark→bright), matching real maps.
- **Energy distance (best EMA checkpoint, epoch 39, 200 sampler steps) = 0.127**
  vs held-out reals — ≈1.4× the measured real-vs-real floor (0.090).
- **Sampler steps:** 200 beats 500 (0.127 vs 0.197) — the Euler ODE oversolves past
  ~200 steps; 200 is the sweet spot. `sampler_steps=200` retained.
- **Held-out `val_loss` bottomed at epoch 39** (best-checkpoint fired at 19 and 39,
  not 59/79) — the new validation signal flags convergence/mild overfit after ~40
  epochs; EMA + best-checkpoint capture the epoch-39 weights regardless.
- The default `openfwi` variant is already the tuned config; no changes needed.

**Next levers (not done):** more families (CurveVel/Fault) for a richer prior;
conditional `p(v|d)` with PIDM-style physics residuals (needs the seismic data + wave
forward operator) — the core research goal.
