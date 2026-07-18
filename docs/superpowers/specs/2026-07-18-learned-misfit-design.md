# Learned data-space misfit `J` — design

**Date:** 2026-07-18 · **Framework:** new `experiments/0006_learned_misfit` + package module
`physics/learned_misfit.py` · **Consumes:** `0004_inversion`

Source idea: `docs/research/misfit_training.md`. This spec is the expanded, buildable version
of that note.

---

## 1. Motivation

Every guidance path in `0004` steers the FWI update with the gradient of a **data-space**
misfit through the wave operator: `∇_v misfit(F(v), d_obs)`. The two misfits we have —
pointwise L2 and the Peng et al. OT potential (`physics/misfit.py`) — are hand-designed
proxies for "how wrong is this velocity map." They are imperfect: inversions with OT and L2
converge to *different* velocity maps, and OT usually wins, for both the flow-matching and
diffusion priors. The misfit choice materially changes the answer, so it is worth *learning*
a better one.

The quantity we actually care about is the **model-space** error `‖v_true − v‖` — but at
inference `v_true` is unknown, so it cannot be computed. The trick: learn a scalar `J(d1, d2)`
that reads that model-space error off the *data* alone. If `J(d_obs, F(v)) ≈ MSE(v_true, v)`,
then its gradient points down the true model-space error surface — a surface with far fewer
cycle-skipping local minima than the raw waveform L2 — while only ever touching data.

## 2. What `J` is

`J` takes two seismic records and returns a scalar reporting how different their velocity
models are:

- **Signature:** `J(d1, d2) → scalar`, with `d1, d2` of shape `(n_sources=5, n_receivers=70,
  nt=1001)` (the OpenFWI acquisition; batched as `(B, 5, 70, 1001)`).
- **Contract:** `(d1, v1)` and `(d2, v2)` solve the same acoustic wave equation (same source
  geometry/wavelet). Required properties: `J(d, d) = 0`, `J(d1, d2) = J(d2, d1)`, `J ≥ 0`.
- **Intended use:** one argument is the fixed observation `d_obs` (whose `v_true` is unknown);
  the other is `F(v_t)`, the synthetic data of the current inversion state. `∇_{v_t} J`,
  backpropped through the wave operator, is the steering direction.

**Accepted assumption (from the source note).** `J(d1, d2) = 0 ⇒ v1 = v2`. This *ignores*
FWI non-uniqueness (two distinct maps can share data). Justified empirically: maps with
closer data are observed to be closer in model space. A limitation, not a theorem — recorded
in §9.

## 3. Architecture — Siamese Euclidean embedding (option A)

```
J(d1, d2) = ‖ φ(d1) − φ(d2) ‖²          φ : (B, 5, 70, 1001) → (B, k),  k ≈ 128
```

Chosen over a general two-input net and a difference-feature MLP because **every required
property is structural, not trained**: symmetry, `J(d,d)=0`, `J≥0`, and the triangle
inequality (a genuine metric in latent space → the smooth, fewer-minima landscape that is the
whole goal). It is also the cleanest inference wiring: `φ(d_obs)` is computed once and frozen,
so `J(pred) = ‖z_obs − φ(pred)‖²` matches the existing `MisfitFn` signature exactly.

**Target/label.** `J` regresses the **MSE of the two normalized `[-1,1]` velocity maps**:
`target(v1, v2) = mean((v1 − v2)²)`. MSE *is* a squared-Euclidean distance, so it is
representable by a squared-Euclidean embedding exactly (in the limit φ = "decode-and-flatten");
this is the geometrically matched target for option A. Report RMSE = √J for interpretability;
guidance uses J directly (monotone transforms only rescale the gradient).

**Encoder φ (`Encoder(nn.Module)`).** Pure forward, differentiable (guidance backprops through
it). Sketch:
- Standardize the input by a **frozen** scale computed once from the bank (global RMS of `d`),
  applied identically to `d_obs` and synthetics — the same observation-frozen-normalization
  discipline as `OTMisfit`.
- Conv2d stack over the `(receiver × time)` plane with the 5 sources as input channels,
  progressive stride-2 downsampling, GroupNorm + SiLU.
- Global average pool → MLP head → `k`-vector. No final nonlinearity on the embedding.
- Target size ~1–3 M params (small next to the DiT prior; a wave solve dominates cost anyway).

## 4. Training data — offline pair bank

The single highest-leverage design decision. `J` must be accurate on the maps an inversion
*actually visits* — smooth/blurry early states, plausible-but-wrong mid states — not only on
pairs of clean dataset maps.

**Bank (built once, cached).** Each entry is `(v_i, d_i = simulate(v_i))`, `d_i` computed by
the same `physics.forward.simulate` used in guidance and cached to disk (gitignored, e.g.
`data/learned_misfit/CurveFault_B/bank.pt`). Simulation is the cost; decoupling it from
training (sample cheap index pairs at train time) is the point. `v_i` is drawn from:

| source | regime it covers |
|---|---|
| CurveFault_B **train** maps (real) | large, structured differences |
| Gaussian-blur of real maps, random σ | smooth / early-inversion states |
| convex blends `α v_a + (1−α) v_b` | intermediate states |
| flow-matching prior samples (0001 ckpt) | plausible-but-wrong maps |

**Pairing sampler (train time).** Sample index pairs `(i, j)`; `target = MSE(v_i, v_j)`. Mix
modes so the full distance range is populated: some pairs `(real, perturbation-of-same-real)`
for the small-distance regime, some cross pairs for the large-distance regime. All velocity in
normalized `[-1,1]`.

**Splits.** Hold out a CurveFault_B map subset for validation; **exclude the benchmark targets**
(`curvefault_b_*`, from `data/inversion_bench`) from the bank so the downstream inversion target
`curvefault_b_17` is unseen.

**Risk / fallback (see §9).** The gold-standard v2 distribution is *real inversion-trajectory
intermediates*. The four augmentation sources approximate them. If validation alignment (§6.2)
is weak, harvest actual intermediate states from `0004` trajectory runs into the bank — a
documented enhancement, kept out of the MVP to avoid a train-inversion-train loop.

## 5. Inference wiring into `0004`

- Add `"learned"` to `MISFITS` in `physics/misfit.py`; `make_misfit("learned", d_obs, *,
  ckpt, min_freq_hz, dt)` loads φ from `ckpt`, precomputes and freezes `z_obs = φ(d_obs)`,
  returns `fn(pred) = ‖z_obs − φ(pred)‖²`.
- Thread a `method.misfit_ckpt` field through `InversionConfig`; validate it is set iff
  `misfit == "learned"`.
- Band-limiting: if `obs.min_freq_hz > 0`, φ consumes the same high-passed `d` as L2/OT (φ
  must be trained on the matching pipeline). The clean-track experiment (§7) uses `min_freq=0`.
- No change to any steering method: `learned` is just another `misfit_factory` output.

## 6. Validation

`compute_val_loss`-style diagonal-only checks do not apply; `J` is validated by:

1. **Regression quality.** MSE / R² of `J` vs `target` on held-out bank pairs.
2. **Guidance-relevant alignment (the property that matters).** Spearman rank-correlation
   between `J(d_obs, F(v))` and true `MSE(v_true, v)` over a sweep of `v` around a held-out
   `v_true`. A high rank-correlation is what makes `−∇J` point toward `v_true`.
3. **Landscape diagnostic.** Along `v(α) = (1−α) v_true + α v_other`, plot `J(d_obs, F(v(α)))`
   against L2 and OT — the claim is a smoother, more monotone descent with fewer local minima.
4. **Downstream (the acceptance test).** §7.

A `smoke` variant with a trivial bank + few steps gives the mandatory fast plumbing check.

## 7. Downstream experiment (acceptance)

Flow-matching prior, clean track, single target `curvefault_b_17`, three misfits:

```
uv run python experiments/0004_inversion/run.py \
  prior=flow_matching method=flow_tilt method.misfit=l2 \
  target=curvefault_b_17 steps=400 n_samples=10
# … method.misfit=ot
# … method.misfit=learned method.misfit_ckpt=runs/0006_learned_misfit/<ts>/checkpoints/step_<N>.pt
```

Compare on the OpenFWI metrics (MAE/RMSE/SSIM on `[-1,1]`, expected across posterior samples)
plus the per-step misfit trajectory. **Success = the learned misfit matches or beats OT** on
the recovered-map metrics for this target. (Single-target, prior fixed: an existence check, not
a population claim.)

## 8. Components & boundaries

| unit | responsibility | depends on |
|---|---|---|
| `Encoder` (`physics/learned_misfit.py`) | `d → k`-vector, differentiable | torch |
| `SiameseMisfit` / `make_learned_misfit` | freeze `z_obs`, expose `MisfitFn` | `Encoder`, ckpt |
| bank builder (`0006/bank.py`) | `{real, blur, blend, prior} → (v, d)` cache | `simulate`, openfwi loader, 0001 ckpt |
| pair sampler + target (`0006/pairs.py`) | index pairs + `MSE(v_i,v_j)` | bank |
| train loop (`0006/run.py`) | fit φ, log val R²/alignment, ckpt | harness `Run`, above |
| eval/diagnostics (`0006/eval.py`) | alignment + landscape figures | trained φ |
| 0004 wiring | `"learned"` in `MISFITS`, `misfit_ckpt` config | `make_learned_misfit` |

Each is independently testable: encoder shape/gradient; `J(d,d)==0` & symmetry; bank determinism;
sampler distance coverage; `make_learned_misfit` matching `MisfitFn`.

## 9. Risks & open questions

- **Train/inference distribution gap (primary).** If the augmented bank misses the real
  trajectory-state distribution, guidance is OOD exactly where escaping cycle-skipping matters.
  Mitigation: mode-mixed augmentations now; harvest real `0004` trajectory states if §6.2 is weak.
- **Non-uniqueness (accepted).** `J(d,d)=0 ⇒ v1=v2` ignores genuine FWI ambiguity (§2).
- **Euclidean-embedding capacity.** If option A underfits MSE, the expressive upgrade is a
  difference-feature MLP `h(|φ(d1)−φ(d2)|)`, `h(0)=0` — keeps symmetry & zero-diagonal, loses
  the metric guarantee. Out of scope for the MVP.
- **Simulation cost of the bank.** Bounded by bank size × one forward solve; cached once. Prior
  samples and blurred maps still each need a solve — cap bank size explicitly and log it.
- **Family scope.** Trained on CurveFault_B only; cross-family generalization untested.

## 10. Out of scope

Camp-C physics-in-training; multi-family/geometry-general `J`; learned-metric heads (option B);
joint two-input cross-attention (option D); trajectory-harvested banks (fallback only).
