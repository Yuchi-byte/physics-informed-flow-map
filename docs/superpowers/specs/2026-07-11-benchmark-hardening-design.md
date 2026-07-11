# Benchmark Hardening (Tier 0) + Disease-Existence Probes (Tier 0.5) — Design

**Status: APPROVED 2026-07-11** (user sign-off on `docs/research/fwi-steering-proposal.md`
§1/§1.1/§1.5 after the two review sessions; this spec turns those sections into components).

**Goal:** Upgrade the 0004 inversion benchmark from "noiseless inverse crime" to two named,
reproducible configurations — a **calibration track** (matched-σ noise; exact posterior
exists) and a **robustness track** (missing lows, operator mismatch, OOD targets; realistic)
— and build the Tier 0.5 probe harness (A–E) that demonstrates and visualizes the two target
pathologies (cycle skipping, non-uniqueness) before any misfit/scheduling method work.

**Non-goals:** any new steering method (Tier 1A/1B — gated on the probes); spectral/softmin
misfits (§2.1 — separate plan once Probe A's harness exists); changes to priors or training;
3D; field data.

**Research context:** `docs/research/fwi-steering-proposal.md` §1 (upgrades, §1.1 detailed
rationale), §1.5 (probes + decision rules). `FWI_problem_exploration/README.md` findings 1–5
are the prior evidence this extends.

---

## 1. The observation model — one config object, two tracks

Today `d_obs = simulate(v_true).detach()` (`inversion/single_target.py:53`) — noiseless,
same operator as guidance, full band. All hardening enters through one new pydantic config,
threaded to every place a target is loaded:

```python
class ObservationConfig(BaseModel):
    # band-limiting (part of F on BOTH sides — calibration-compatible, spec §2)
    min_freq_hz: float = 0.0          # 0 = off; 4.0 = the canonical "missing lows" stressor
    # calibration track (spec §3)
    noise_frac: float = 0.0           # sigma = noise_frac * RMS(clean band-limited d_obs)
    noise_seed: int = 0               # combined with target key -> frozen realization
    # robustness track: generation-operator mismatch (spec §4); guidance side never changes
    grid_scale: int = 1               # generate on a (grid_scale x)-refined grid, decimate back
    wavelet_freq_scale: float = 1.0   # generation wavelet centre-frequency multiplier
```

Track presets ship as a Hydra config group `obs` in `experiments/0004_inversion/conf/`:

| preset | min_freq_hz | noise_frac | grid_scale | wavelet_freq_scale | claim it serves |
|---|---|---|---|---|---|
| `clean` (default) | 0 | 0 | 1 | 1.0 | continuity — byte-identical to today |
| `calib` | 0 | 0.05 | 1 | 1.0 | exact posterior; calibration metrics falsifiable |
| `hard_calib` | 4.0 | 0.05 | 1 | 1.0 | exact but wide/multimodal posterior (SMC-vs-collapse test) |
| `robust_mild` | 4.0 | 0.05 | 2 | 1.0 | + discretization mismatch (inverse crime dead) |
| `robust` | 4.0 | 0.05 | 2 | 0.97 | + source uncertainty |

`observe(v_true, cfg, key) -> Observation(d_obs, sigma, noise_floor)` is the single
constructor. `Observation.sigma` is `None` on noiseless configs (σ has no principled value
there — proposal §1.1b); `noise_floor = sigma**2 * d_obs.numel()` is the expected L2 misfit
at the true model (χ² mean), the yardstick every misfit ratio is reported against.

## 2. Band-limiting: filter the *data* with one shared function on both sides (D1)

*(Revised during implementation 2026-07-11.)* The first design filtered the source wavelet
inside `simulate` and leaned on source linearity ("filtering the wavelet ≡ filtering the
data") for the guidance side. Implementation showed the equivalence is only exact on
infinite records — finite records + FFT filtering leave circular-wrap / crop-edge residues
that a bit-level test rightly rejects. The revised design is simpler and *exact by
construction*: **one function, `physics.filters.highpass`, applied to data on both sides**:

- **Generation side**: `observe()` applies it to the clean simulated data → `d_obs` has no
  lows (noise added after, broadband).
- **Guidance side**: the **misfit wrapper** applies the identical function to *predictions*
  before comparison (`make_misfit(..., min_freq_hz=)`). The misfit interface is already
  threaded through every guidance path, so no sampler changes.
- **Eval side**: `single_target`/`eval` compute the L2 misfit-ratio metric on high-passed
  predictions too, else the metric would carry an unfixable low-band residual.

Both sides now compose the *same* operator `highpass ∘ simulate` — the filter is part of F
identically, so the posterior stays exactly defined (the (a)+(b) "hard calibration"
argument of proposal §1.1), with no linearity assumption needed. `simulate` itself carries
no band-limit knob. Filter details: zero-phase raised-cosine (flat above `1.5·min_freq`,
zero below `0.5·min_freq`), **zero-padded to 2× length** so the filtering is a linear (not
circular) convolution — record-end energy cannot wrap into early times. Caveat recorded: a
finite record cannot be exactly band-limited (crop-edge leakage ~−30 dB on edge-hard
signals; ~−110 dB on smooth ones) — irrelevant for the missing-lows purpose since the same
residue exists on both sides of the misfit.

Noise is added **after** band-limiting and stays white/full-band — the field situation
(signal has no lows; noise is broadband) and keeps the matched likelihood plain white L2.

## 3. Noise: frozen, deterministic, never stored (D2)

`eta = noise_frac * rms(d_clean) * randn(generator=Generator(seed=crc32(f"{key}|s{noise_seed}")))`
with `key` = benchmark target id (or `"val{gidx}"` on the legacy path). Consequences:

- The realization is **fixed per (target, config)** — `d_obs` behaves like a field
  recording: every method, run, and sampler step sees the same noisy data. Never resampled.
- Nothing is written to `data/inversion_bench/` — regeneration is deterministic from
  `(v_true, ObservationConfig, key)`, so the benchmark directory and manifest stay at
  schema v1 and results remain reproducible from config alone.
- σ is *known by construction* and handed to the likelihood (matched σ). It is **not a
  tuned hyperparameter** — the calibration track exists precisely to remove σ from the
  tuning space (proposal §1.1b).

## 4. Operator mismatch: refined-grid generation, guidance untouched (D3)

`grid_scale = s` regenerates `d_obs` on a physically identical but numerically finer
discretization: velocity bilinearly upsampled to `((70-1)s+1)²` nodes (`align_corners=True`
so the physical extent 690 m is exact), `dx/s`, `dt/s` (Courant number unchanged → stable),
`nt*s` samples, sources/receivers at the same physical positions (`coarse_index * s`),
output decimated `[::s]` back to `(5, 70, 1001)`. The mismatch is genuine numerical
dispersion difference — the mild rung of proposal §1.1c's ladder. `wavelet_freq_scale`
perturbs the generation wavelet's centre frequency (moderate rung). The elastic/attenuation
hard rung is deferred (separate spec when needed).

The guidance operator **never** changes — that is the definition of the mismatch.

## 5. Metrics and σ threading (D4)

- `inv/misfit_over_floor = guided L2 misfit / noise_floor` wherever noise is on (≈1 is
  perfect; <1 = fitting noise = overfitting, now *measurable*).
- Methods with an explicit likelihood temperature (`mfm_g.sigma`) get the matched value
  when `obs.noise_frac > 0` unless explicitly overridden (`method.sigma=null` → matched).
  `flow_tilt`/`dps` keep `guidance_strength` (their normalize-grad scaling severs the
  Bayesian link; matched σ matters for reporting, MFM-G, and later SMC).
- Calibration metrics themselves (cov90/CRPS over posterior samples) are Tier 1B work, not
  this spec — but the floor + matched σ make them computable later without re-running.

## 6. Tier 0.5 probes (A–E): scripts, not package code

Probes live in `FWI_problem_exploration/` (A, D, E extend existing scripts there) and
`experiments/0004_inversion/` sweep configs (B, C reuse `eval.py`). They are exploration
artifacts: plotted, journaled, summarized in `FWI_problem_exploration/disease_report.md` —
not shipped package modules. Definitions, metrics, and decision rules: proposal §1.5.
The one shared piece of package code they need is the observation model above (so probes
run on exactly the benchmark configurations the tiers will use) and a **basin-failure
classifier** (converged misfit + bad MAE, or elevated low-band misfit trap screen —
finding 3) small enough to live in the probe scripts.

## 7. Testing strategy

Hermetic, CPU, tiny grids (the `test_forward.py` pattern: 16×16, nt=120, 40 Hz):

1. **Filter**: smooth low tone killed (<1e-9 energy), smooth high tone preserved (1e-6);
   zero-phase (pulse peak position and symmetry preserved).
2. **Wrap-freedom (load-bearing)**: a pulse near the record end leaves the record head at
   <1e-4 relative (linear convolution via padding; a circular filter fails this by ~2
   orders of magnitude). Replaces the original wavelet-vs-data linearity test — D1 revised
   to one-shared-function-both-sides, which needs no equivalence proof.
3. **Noise determinism**: same `(v, cfg, key)` → identical `d_obs` across calls/devices
   (CPU generator); different `key` or `noise_seed` → different realization; `sigma` scales
   with `noise_frac` and equals `noise_frac * rms` exactly.
4. **Floor**: `‖d_obs − d_clean‖² / noise_floor ≈ 1` within χ² tolerance.
5. **Clean passthrough**: `ObservationConfig()` (all defaults) reproduces today's
   `simulate(v).detach()` bit-for-bit — the continuity guarantee for every existing result.
6. **grid_scale**: output shape unchanged; `s=2` vs `s=1` data close (same physics) but not
   equal (different discretization) — relative L2 difference in a sane band, e.g. 1e-4..1e-1.
7. **Misfit wrapper**: `make_misfit("l2", d_obs, min_freq_hz=f)(pred)` ==
   `make_misfit("l2", highpass(d_obs? no — d_obs enters already filtered)…` — precisely:
   wrapper high-passes `pred` only; test equals plain L2 between `highpass(pred)` and the
   (already band-limited) `d_obs`. OT under the wrapper: gradient flows (autograd check).

## 8. Resolved design questions

- **Store noisy d_obs on disk?** No — deterministic regeneration (D2); manifest stays v1.
- **Filter data or wavelet?** Wavelet on generation (exact, part of F), data-side wrapper on
  guidance/eval (equivalent by linearity; avoids touching five samplers).
- **Is σ tuned?** No — matched by construction on the calibration track; a *mis-specified-σ
  ablation* (sweep assumed σ around σ_true) is a later experiment, not a tuning loop.
- **Colored noise?** Deferred. White-after-band-limit is the calibration-track choice (exact
  L2 likelihood); colored noise is a robustness-track refinement with a non-diagonal
  covariance — revisit if a reviewer-facing result needs it.
- **OOD targets in this spec?** Schema/tagging design only (manifest v2 with `ood: true`,
  `source: "marmousi2"`); acquisition + cropping is plan Task 5 and needs a data download —
  kept out of the core observation model, which is target-source-agnostic.
