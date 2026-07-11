# Benchmark Hardening + Disease Probes — Implementation Plan

> **Status: ACTIVE — Tasks 1–3 in progress 2026-07-11.**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two-track hardened benchmark (calibration: matched-σ noise;
robustness: missing lows + operator mismatch + OOD) as one `ObservationConfig` threaded
through 0004, then build the Tier 0.5 probe harness (A–E) and produce
`FWI_problem_exploration/disease_report.md` with the Tier 1A go/no-go call.

**Design:** `docs/superpowers/specs/2026-07-11-benchmark-hardening-design.md`
**Research rationale:** `docs/research/fwi-steering-proposal.md` §1, §1.1, §1.5.

## Global constraints

- Do NOT edit reference packages. Work and commit on `main`; ruff → mypy → pytest must pass.
- Tests hermetic and CPU-sized (16×16 grids, nt≈120 — the `test_forward.py` pattern).
- **Continuity guarantee:** `obs=clean` (all defaults) must reproduce current behavior
  bit-for-bit; every existing config/run stays valid with zero changes.
- The guidance operator is never modified by mismatch knobs (that is the definition of
  operator mismatch); band-limiting reaches guidance only via the misfit/eval wrapper.

---

### Task 1: `physics/filters.py` + band-limited `simulate`

**Files:** new `physics/filters.py`; `physics/forward.py`; `tests/test_filters.py`;
`tests/test_forward.py`.

- [x] `highpass(x, min_freq_hz, dt)` — zero-phase cosine-taper high-pass along the last
      (time) axis via rfft, **zero-padded 2× (linear, not circular, convolution)**; no-op
      for `min_freq_hz <= 0`.
- [x] ~~`simulate(..., min_freq_hz)`~~ — **design revised during implementation** (spec §2):
      band-limiting is data-side on both generation (`observe`) and guidance (misfit
      wrapper) with the one shared function — exact by construction; `simulate` carries no
      band-limit knob.
- [x] Tests: smooth-tone kill/preserve; zero-phase (peak + symmetry); wrap-freedom
      (record-end pulse leaves the head clean); differentiability; default passthrough.

### Task 2: `physics/observation.py` — the observation model

**Files:** new `physics/observation.py`; `tests/test_observation.py`.

- [x] `ObservationConfig` (pydantic): `min_freq_hz`, `noise_frac`, `noise_seed`,
      `grid_scale`, `wavelet_freq_scale` (spec §1).
- [x] `Observation` dataclass: `d_obs`, `sigma: float | None`, `noise_floor: float | None`.
- [x] `observe(v_true, cfg, key)` — generation-side simulate (with mismatch knobs + band
      limit), frozen per-key noise (`crc32(f"{key}|s{noise_seed}")` CPU generator),
      `sigma = noise_frac * rms(d_clean)`, `noise_floor = sigma² * numel`.
- [x] Tests: determinism per key; key/seed sensitivity; σ scaling; χ² floor check
      (`‖η‖²/floor ≈ 1`); clean-config bit-identity with `simulate(v).detach()`.

### Task 3: generation-operator mismatch in `simulate`

**Files:** `physics/forward.py`; `tests/test_forward.py`.

- [x] `simulate(..., grid_scale: int = 1, freq_scale: float = 1.0)`: bilinear-upsample
      velocity to `((n-1)s+1)` nodes (`align_corners=True` — exact 690 m extent), `dx/s`,
      `dt/s`, `nt*s`, source/receiver columns at `coarse_index*s`, decimate output `[::s]`
      back to `nt`. Courant number invariant → stability preserved.
- [x] Tests: shape invariance; `s=2` close-but-not-equal to `s=1` (relative L2 in a sane
      band); `freq_scale` shifts the data spectrum's peak.

### Task 4: thread `ObservationConfig` through 0004

**Files:** `inversion/single_target.py`, `inversion/evaluate.py` (evaluator),
`physics/misfit.py`, `experiments/0004_inversion/{run.py,eval.py,conf/}`,
`tests/test_misfit.py`, `tests/test_inversion_eval.py`.

- [x] `make_misfit(..., min_freq_hz=0.0, dt=1e-3)` — wrapper high-passes *predictions*
      before any base misfit (guidance-side band limit, D1); autograd flows (OT + L2).
- [x] `load_target(...)` accepts `obs_cfg` → returns `Observation`; `invert_and_report`
      band-passes eval predictions when `min_freq_hz > 0` and reports
      `inv/misfit_over_floor` + `inv/sigma_true` when noise is on. Evaluator/`score_target`
      threaded too (`obs_cfg`, in-band misfit, `misfit_over_floor`); classical run.py path
      gets a wrapped `forward_op` under band limit (eval.py FWI modules guarded with a
      clear error instead — thread when Probe B needs the multi-map path).
- [x] Hydra group `conf/obs/`: `clean.yaml` (default), `calib.yaml`, `hard_calib.yaml`,
      `robust_mild.yaml`, `robust.yaml` (spec §1 table).
- [ ] *(deferred to Tier 1B work)* Matched-σ default for `mfm_g` when `obs.noise_frac > 0`
      — needs the target loaded before module construction; do together with the
      calibration-ladder experiments.
- [x] Smoke on RunPod (2026-07-11): `flow_tilt` map 6044 `obs=clean` vs HEAD — MAE
      bit-identical, SSIM/ratio deltas ≤6e-7 (float noise); `obs=calib` reports
      `sigma_true`/`misfit_over_floor`; `obs=hard_calib` runs the banded factory
      (flow_tilt) and the wrapped classical forward end-to-end.

### Task 5: OOD targets (Marmousi2 / Overthrust crops)

**Files:** `inversion/benchmark.py` (+ a small acquisition script); `tests/test_benchmark.py`.

- [ ] Acquire Marmousi2 + Overthrust velocity models (SEG open data; script documents URLs
      + checksums into `data/ood_models/`).
- [ ] Crop generator: 70×70 windows at native-ish resolution (record the resampling),
      value-clipped to [1500, 4500] m/s; ~10 crops/model, TV-stratified like `select_targets`.
- [ ] Manifest **schema v2**: `ood: bool`, `source: str` fields (v1 files still load);
      previews + gallery; `InversionBenchmark.ood_ids` property.
- [ ] Tests: v1/v2 manifest compat; crop stats within normalization range; determinism.

### Task 6: Probe A — landscape grid under missing lows

**Files:** `FWI_problem_exploration/landscape_grid.py` (extends
`cycle_skipping_landscape.py` + `cycle_skipping_barrier.py`).

- [ ] α-scan and trapped→truth barrier scan × `min_freq ∈ {0, 2, 4, 6}` × misfit
      ∈ {L2, OT(k=100), OT(k=0), envelope} on the observation model (`observe()`), map 6044
      + one hard-family target.
- [ ] Outputs: the freq×misfit landscape grid figure; aggregate-minima count/depth +
      barrier-height table; journal entry.

### Task 7: Probes B & C — basin-failure rates (classical vs guided)

**Files:** `experiments/0004_inversion/` sweep driver (reuse `eval.py` machinery) + a
probe summary script; basin-failure classifier in the probe script (converged misfit AND
(MAE > threshold OR low-band misfit elevated — the finding-3 trap screen)).

- [ ] Probe B: `classical_fwi` (smooth start) × 20 core targets × {clean, hard_calib(4 Hz)}.
- [ ] Probe C: `flow_tilt·{l2,ot}` same grid, n=8; per-sample trap screen + MAE bimodality.
- [ ] **Decision-rule call** (proposal §1.5 table) recorded in the journal + proposal
      decision log: Tier 1A go/no-go.

### Task 8: Probes D & E — null-space quantification + sampler-generated atlases

**Files:** `FWI_problem_exploration/{nullspace_atlas.py,skip_atlas.py,disease_report.md}`.

- [ ] Probe D: misfit-vs-MAE scatter across pooled equal-fit models; depth-profile fans;
      per-family ambiguity-radius table (± noise).
- [ ] Probe E1: same-d_obs pool (flow_tilt seeds + fmrg_e/mfm_g overfit configs + classical
      multi-start): gallery, pixelwise std, PCA null-space modes, log p(v)-colored embedding
      (embedding may defer the CNF log-likelihood to Tier 1B rung 1.5 and use misfit color
      first).
- [ ] Probe E2: flow_tilt vs `T_τ·d_obs` (τ = ±67 ms) skip families; difference maps; the
      E1-vs-E2 MAE histogram money figure.
- [ ] `disease_report.md` assembling all probes; linked from the proposal.

### Task 9: post-hardening closeouts (runs, not code)

- [ ] L2-vs-OT repeat on `hard_calib` + `robust_mild` (the §2.1c before/after leg).
- [ ] mfm_g σ-boundary extension {0.003, 0.01} + multi-map leg (pre-hardening pending item).
- [ ] `t_cond>0` posterior-fidelity check on the esd_teacher checkpoint (gates Tier 1B
      rung 3; independent of this plan's code but scheduled here so it isn't lost).
