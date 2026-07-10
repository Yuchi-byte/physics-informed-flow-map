# Steering-Method Proposal — the 0004 inversion program

*Draft 2026-07-09, revised 2026-07-10 (Claude-drafted for curation; edit freely). Companion documents:*
*[`fwi-improvement-ideas.md`](fwi-improvement-ideas.md) (brainstorm landscape) and
[`research-plan.html`](research-plan.html) (thesis: posterior quality, not cost).*

**Scope decision this document encodes:** we stay in Camp A (learned prior + inference-time
physics steering) and position the work **ML-side on synthetic-but-hard benchmarks** —
band-limited, noisy, operator-mismatched, with OOD targets (Marmousi/Overthrust crops) —
rather than chasing field data this cycle. This serves both audiences: hard-synthetic is the
credibility bar for geophysics reviewers and the reproducibility bar for ML reviewers.

---

## 0. Assets already in place (don't re-plan these)

- **Priors**: definitive 0001 (flow matching), 0002 (flow map), 0003 (DDPM) checkpoints
  across the 10 OpenFWI families; quality-vs-NFE curves measured (flow map wins ≤16 NFE).
- **Inversion harness**: 0004 with a validated prior×method compatibility matrix
  (flow_tilt, DPS, RED-DiffEq, MFM-G/GF, FMRG-E, classical/realistic FWI).
- **Benchmark targets**: self-contained TV-stratified 201-target set
  (`data/inversion_bench/`, `inversion.benchmark`), easy→hard per family.
- **Misfit knob**: `method.misfit=l2|ot` (Peng et al. §5.1 potential; `physics/misfit.py`),
  L2-vs-OT smoke + first comparison **still pending on RunPod**.
- **Known findings that shape this proposal** (JOURNAL): posterior *selection* — not prior
  quality — is the measured bottleneck (GT-free pick trails oracle by ~100–150 m/s);
  MFM-G matches flow_tilt's data fit but loses on model error under the noiseless inverse
  crime; σ has no principled value without observation noise; FMRG-E fits data harder but
  goes further off-manifold; the OT potential's long-range shift sensitivity comes from the
  frozen amplitude weighting, not the transport (`test_misfit.py`).

---

## 1. Tier 0 — benchmark hardening (prerequisite, not a paper claim)

Every method claim below is only credible on a benchmark that cannot be gamed by
data-overfitting. Three upgrades to `0004`, all cheap:

| Upgrade | What | Why |
|---|---|---|
| **Band-limited data** | High-pass `d_obs` (and source wavelet) to kill content below ~4 Hz; knob `bench.min_freq_hz` | The community's canonical cycle-skipping stressor ("no lows"); enables the missing-lows claim in Tier 1A |
| **Observation noise + matched σ** | `d_obs = F(v) + η`, `η∼N(0,σ_true²)`; reward σ **set to σ_true** | Makes the Bayesian target well-defined → calibration metrics (cov90, CRPS) become falsifiable; the unblocking experiment for the posterior-quality thesis |
| **Operator mismatch (kill the inverse crime)** | Generate `d_obs` on a finer grid / different dt or wavelet than the guidance operator | Removes the "fit data harder = win" degeneracy that currently flatters aggressive methods |
| **OOD targets** | Marmousi2 / Overthrust 70×70 crops appended to the benchmark manifest | Measures how much steering compensates prior misspecification (research-plan M5) |

**Two-track split (2026-07-10 review).** "Matched σ" and "operator mismatch / OOD" cannot
live in one benchmark configuration: calibration metrics (cov90, CRPS) are falsifiable only
when the guidance operator equals the observation operator, the noise is exactly the assumed
N(0, σ_true²), and targets are in-distribution for the prior. Under mismatch or OOD there is
no "true posterior" to be calibrated against. Tier 0 therefore produces **two named configs**:

- **Calibration track** — same operator, matched white noise (add noise *before* any band
  filter, or carry the filtered-noise covariance in the likelihood), in-distribution targets.
  Tier 1B's claims live here.
- **Robustness track** — band-limited, operator-mismatched, OOD crops. Tier 1A's claims live
  here. Calibration metrics may still be reported, but the expected reading is "everything is
  overconfident" — a robustness statement, not a calibration one.

Also close out the **pending L2-vs-OT run** (flow_tilt + mfm_g on map 6044 and a benchmark
slice, matched solves; retune σ for OT's O(1) scale) — before and after hardening, since
the hypothesis is that the OT edge appears only *after* hardening.

**Effort**: small (filters, noise, a second simulate config, data plumbing).
**Kill criterion**: none — this tier is infrastructure.

---

## 1.5 Tier 0.5 — disease-existence check (gates Tier 1A; run before any misfit work)

**Principle: demonstrate and *visualize* the pathology before building the cure.** We have
been burned once already — the MFM-G steering arc (JOURNAL ~2026-06-28→07-02: guidance-scale,
renorm, σ, and mc-samples sweeps, then an n=32 validation) ended with the diagnosis that
under the noiseless inverse crime **no steering strength significantly helps beyond the
prior**: strong steering exploits the data-fit degeneracy (misfit −10×, MAE 0.284→0.52–0.62),
gentle steering does nothing (SSIM +0.023 ≈ 1 SEM at n=32), MC draw count is irrelevant
(mc 4→20 identical at 5× solves), and at matched data fit MFM-G still loses on model error
(misfit_ratio 0.077 vs flow_tilt's 0.072, MAE 0.195 vs 0.127) — with noiseless data σ has no
principled value, so there is no well-defined posterior to be *better at*. Precisely stated:
the benchmark rewarded guidance (−40% MAE) and even misfit design (the OT wins), but not the
posterior-quality edge that is MFM-G's entire claim. Caveat (part of why §1.5 exists): that
diagnosis is still confounded with the untested `t_cond>0` channel (§6.5 item 1); the
matched-σ track plus the fidelity check disentangle them.

`FWI_problem_exploration/` (single map 6044, full-band data) already gives a partial answer,
and it cuts *against* the naive cycle-skipping story:

- The aggregate L2 landscape along `α·v_true` is unimodal-but-saturating; the textbook skip
  wells exist only at the single-trace level (README finding 1).
- From a cycle-skipped start, multiscale and envelope reach the **same wrong MAE** as plain
  L2 — on this geometry the binding pathology is **non-uniqueness**, not basin geometry
  (finding 4). Convexifying the metric did not help because the null space, not the wells,
  is what bites.
- Trapped models betray themselves in the low-frequency band misfit even when full-band
  misfit is ~0 (finding 3) — a cheap GT-free trap diagnostic, reusable for posterior-sample
  screening (see §6.5 item 3).

What is *not* yet known: all of that was measured with full-band data (lows present) on one
easy FlatVel-A map with classical descent. The Tier 1A stressor is the opposite regime —
**missing lows** — and a guided sampler is not a descent method. Four probes close the gap:

### Probe A — misfit landscapes under missing lows (extend `cycle_skipping_landscape.py`)

- High-pass `d_obs` + wavelet at `min_freq ∈ {0 (control), 2, 4, 6}` Hz; rerun (i) the
  α-scan `v = α·v_true`, α ∈ [0.6, 1.4], and (ii) the trapped→truth straight-line barrier
  scan (`cycle_skipping_barrier.py`).
- Run **every candidate misfit over the same scans**: L2, OT (k=100 and k=0), envelope, the
  spectral variants (§2.1b), softmin-shift (§2.1a). One harness → the misfit-convexity
  figure (misfit vs α / vs barrier coordinate, per frequency band) that becomes Fig. 1 of
  any misfit paper.
- Metrics: number and depth of *aggregate* local minima; barrier height ratio vs full-band;
  width of the global valley.
- Expectation to test: removing lows deepens the per-trace wells and may finally make the
  aggregate landscape multimodal — the regime where misfit design can matter. If it does
  not, Tier 1A is dead on this geometry (see decision rules).

### Probe B — classical basin-failure rate at benchmark scale

- classical·tv (existing 0004 method) from smooth `0.75×smooth(v_true)` starts on the
  20-target core set (`*_08` + `*_09`, two per family), full-band vs `min_freq=4` Hz.
- Metrics: **basin-failure rate** = fraction with converged data misfit (< threshold) AND
  (MAE > threshold OR elevated low-band misfit per the finding-3 trap screen); final MAE
  distribution per family.
- Cheap: no prior, no sampler; ~40 classical runs.

### Probe C — does the guided sampler inherit the disease?

- flow_tilt·{l2, ot} on the same 20 targets, full-band vs `min_freq=4`, n=8 samples each,
  method defaults otherwise (800 solves).
- Metrics: per-sample trap screen (low-band misfit), MAE-histogram bimodality across the 8
  samples (cycle skipping shows as multimodal failure, the §2 metric), basin-failure rate
  side-by-side with Probe B's classical rate.
- Visuals: gallery of trapped vs untrapped posterior samples per target; far-offset trace
  overlays (the one-cycle-apart picture from `cycle_skipping_landscape.py` fig 2).

### Probe D — non-uniqueness, concretized and quantified

- Per core target: pool equal-data-fit models (classical multi-start as in
  `classical_fwi_nonuniqueness.py` + guided posterior samples) and plot the **data-misfit vs
  model-MAE scatter** — the vertical spread at fixed misfit *is* the null space, made
  visible.
- Depth-resolved ambiguity: laterally averaged velocity-profile fan across the pool
  (finding-5 style) → *where* the data stops constraining the model, per family.
- Repeat under matched-σ noise (calibration track): noise should *widen* the honest
  ambiguity — the very quantity Tier 1B's posterior is supposed to report.
- Output: a per-family **"ambiguity radius at fixed relative misfit"** table — the
  non-uniqueness baseline every Tier 1B claim is measured against.

### Probe E — disease atlases via the samplers themselves (the visualization layer)

Use our own generation/inversion machinery to *sample* the pathological sets and look at
them — the samplers are level-set explorers, not just the methods under test. (Probe D
quantifies; Probe E is what the quantities look like.)

- **E1 — the same-d_obs family (null-space atlas).** For a fixed target, pool maps that fit
  its d_obs to equal misfit (misfit_ratio ≤ threshold) from deliberately *diverse*
  generators: flow_tilt·{l2, ot} from n ≥ 32 noise seeds (the on-manifold slice of the
  solution set); the JOURNAL-documented off-manifold data-fitters used as **features, not
  bugs** (fmrg_e at g ≥ 1, mfm_g at small σ / renorm=T — the misfit-crushers); and classical
  multi-start (`classical_fwi_nonuniqueness.py`). Visuals:
  - gallery sorted by MAE-to-truth (equal data fit across the whole row — the disease in
    one image);
  - pixelwise std / quantile maps → *where* the null space lives spatially (expect the deep
    half, per finding 5);
  - top PCA eigenmaps of the pool = empirical null-space modes;
  - pairwise-MAE MDS embedding colored by (i) generator and (ii) prior log-likelihood
    log p(v) — a visual preview of rung 1.5: off-manifold members should sit at low log p(v).
  - Headline quantity: ambiguity radius **with** the prior (flow_tilt subset) vs **without**
    (full pool) — the prior's uniqueness contribution, measured and pictured.
- **E2 — the cycle-skipped family.** Generate *prior-plausible* maps whose predicted data
  is one cycle off: run flow_tilt with the target's d_obs replaced by `T_τ·d_obs`,
  τ = ±one dominant period (≈ 67 ms at 15 Hz; also per-shot-sign variants), plus the
  classical trapped models from `cycle_skipping_escape.py` for the off-manifold version.
  Visuals: skip-family gallery vs truth; difference maps (*where* the velocity absorbs the
  shift — shallow vs deep); far-offset trace overlays (the one-cycle-apart picture).
  This is also the **diagnostic twin of the §2.1a shifted-copies idea**: it empirically
  answers "are the preimages of slightly-shifted data slightly-perturbed models?" — if yes
  at small τ, the softmin nuisance potential is cheap tolerance; if the preimages differ
  wildly, shift tolerance is expensive in model space and the schedule must retire it early.
- **The money figure**: MAE-to-truth histograms of E1 (null space) vs E2 (one full skip) on
  shared axes — the two diseases' sizes in model space, directly comparable. Finding 1
  predicts E2 ≈ 20% background-velocity error; finding 2 predicts E1 spans ~500 m/s. If
  E1 ≳ E2, non-uniqueness is confirmed as the dominant disease on this geometry; the gap
  between them is the honest budget split between Tier 1A and Tier 1B.

Cost note: n=32 flow_tilt draws ≈ 25.6k solves per target/config — run E on 2–3 targets
(one easy, one hard family), not the full core set.

### Decision rules

| Probe B/C outcome (band-limited) | Reading | Consequence |
|---|---|---|
| classical fails AND guided fails | disease exists, prior doesn't cure it | Tier 1A proceeds as planned |
| classical fails, guided mostly clean | **the prior dissolves cycle skipping at this scale** | headline reportable result on its own; retarget misfit work at operator-mismatch robustness, not basins |
| neither fails | geometry too easy for the claim | harden the geometry (lower Ricker centre frequency, deeper/larger crops, longer record), rerun Probe A; do **not** start Tier 1A |

**Deliverable**: `FWI_problem_exploration/disease_report.md` + figures (landscape grid
freq×misfit, trap gallery, misfit-vs-MAE scatters, depth-profile fans), linked from here.
**Effort**: small-medium (~2–4 pod-days, dominated by Probe C).

---

## 2. Tier 1A — objective scheduling over generative time (cycle skipping)

**One story, three instances**: generation is coarse→fine, so make the *objective*
coarse→fine along sampler time t. Subsumes the "staged/annealed strategy" cross-cutting
angle in `fwi-improvement-ideas.md` §2.

**Gate (2026-07-10): run §1.5 first.** `FWI_problem_exploration` finding 4 says the binding
pathology on the current geometry is non-uniqueness, not cycle skipping — Tier 1A's premise
must be demonstrated in the missing-lows regime before this machinery is built.

1. **Frequency annealing** (Bunks et al. × guidance): guide early steps against low-passed
   `d_obs`, ramp the cutoff with t. Precomputed filter bank; drops into the misfit
   interface as a t-aware `MisfitFn`.
2. **Misfit annealing**: kinematic-robust misfit early (envelope, or the OT potential),
   L2 late — the "OT as early-stage convexifier" reading, which our unit-test finding
   suggests is the honest role for the Peng potential.
3. **AWI-driven guidance controller**: per step, compute the AWI Wiener matching filter
   between predicted and observed traces; its concentration at zero lag is a *direct
   measurement of kinematic error*. Use it to gate guidance strength / select the stage —
   replacing Peng's generic TV(x̂₀) proxy with a physics-native one. AWI here is the
   *controller*, not the misfit.

**Hypothesis**: on the band-limited benchmark, scheduled objectives beat every constant
objective (L2, OT alike) at matched solves; the prior alone does not close the gap.
**Experiments**: flow_tilt + mfm_g on FlatFault/CurveFault + OOD slice; constant-L2,
constant-OT, each annealing variant, AWI-gated versions; guidance-misfit trajectories
logged per step.
**Metrics**: MAE/RMSE/SSIM, misfit ratio, basin-failure rate (fraction of samples with
MAE > threshold — cycle skipping shows up as multimodal failure, not mean error).
**Kill criterion**: if constant-L2 with the hardened benchmark already matches the best
schedule on FlatFault *and* OOD targets, the prior absorbs the misfit-design problem —
itself a reportable negative result; pivot effort to Tier 1B.
**Effort**: medium (filter bank + t-aware misfit interface + AWI filter, all CPU-side math).

### 2.1 Misfit-design review (2026-07-10) — the tolerance–uniqueness tension, and three candidate misfits

**Principle to design against.** Every shift-tolerant misfit (OT, AWI, envelope, spectral,
smoothing) works by adding *invariance* — declaring more predicted datasets (hence more
velocity models) acceptable. That widens basins (fights cycle skipping) precisely by
**enlarging the effective null space** (worsens non-uniqueness). Our two target problems
pull the misfit in opposite directions. The resolution is scheduling — tolerant early to
find the basin, tight late to resolve within it (this tier) — plus a prior/posterior
treatment of what tolerance leaves behind (Tier 1B). No misfit choice can address
non-uniqueness itself; `FWI_problem_exploration` finding 4 is the local proof.

**(a) Shifted-copies misfit (proposed 2026-07-09) — corrected form.** Proposal: build
synthetic time-shifted copies `T_τk·d_obs` and *sum* the (L2 or OT) misfits against all of
them, so models whose slight perturbations explain shifted data are preferred. As written it
collapses: `Σ_k ‖F(v) − T_k d‖² = K·‖F(v) − d̄‖² + const`, and `d̄` (the shift-average) is
`d_obs` convolved with the shift distribution — i.e. **sum-of-L2 against shifted copies ≡ a
single L2 against low-pass-filtered data** (Gaussian shifts of spread s ≡ spectral weight
`exp(−ω²s²/2)`). That is Bunks frequency continuation, already instance 1. The OT-sum
version changes ~nothing (W2 is ~linear in small shifts). The *intent* — prefer models whose
neighbourhood explains slightly-shifted data, a flat-minimum/stability criterion — survives
as a **mixture (marginalized) likelihood** over the shift as a nuisance variable:

```
p(d|v) = Σ_k w_k · exp(−‖F(v) − T_τk d_obs‖² / 2σ²)   →   misfit = −logsumexp  (a softmin)
```

Softmin ≠ sum: it fits the data *up to the best small shift* rather than fitting smeared
data. Caveats: a whole-gather shift is a 1-parameter nuisance (time-datum / origin error);
real cycle skipping is per-trace and per-event, per-trace shift dictionaries explode
combinatorially, and the principled per-trace machinery is AWI / graph-space OT. The honest
novelty claim is therefore **not** shift-robustness (classical FWI solved that several times
over: extension methods, AWI, cross-correlation traveltime, DTW) but **nuisance
marginalization inside generative guidance** — no diffusion-FWI paper models
datum/wavelet/gain error in the potential; it extends naturally to source-wavelet and
amplitude-gain nuisances and is the principled response to the Tier 0 operator-mismatch
track. Cost: K misfit evaluations per step, zero extra wave solves.

**(b) Spectral L2 (proposed 2026-07-10) — critique, and what survives.** Four variants with
very different verdicts:

1. **Complex-spectrum L2** — by Parseval, *identical* to time-domain L2 (up to
   normalization). Zero new content; keep only as the unit test (`spectral·complex ≡ l2`
   to float tolerance).
2. **Amplitude-spectrum L2**, `‖ |D̂_pred| − |D̂_obs| ‖²` per trace — exactly invariant to
   per-trace time shift (a shift is a pure phase ramp). That is the wanted property, but it
   is *total* invariance: it discards **all traveltime information — FWI's primary velocity
   signal** — enlarging the null space by the entire ~350-dim group of independent per-trace
   shifts. On our direct-wave-dominated geometry the amplitude spectrum is mostly the
   wavelet and only weakly v-sensitive; and finding 4 says independent per-trace
   wrong-cycle fitting is exactly how the artifact models arise — this misfit cannot even
   *see* that failure mode. **Disqualified as a standalone misfit; admissible only as the
   early-t end of an annealing schedule.** Numerics: `d|z|/dz` is non-smooth at spectral
   zeros — use `|z|²` or ε-smoothed magnitude; log-magnitude doubles as a whitening
   preconditioner (cf. the OT frozen-weighting finding).
3. **Frequency-weighted complex L2**, `Σ_ω W(ω)·|D̂_pred − D̂_obs|²` — by the convolution
   theorem, ≡ time-domain L2 against filtered data. Useful, and *the clean implementation*
   of frequency annealing (instance 1): precompute `rfft(d_obs)` once, anneal `W(ω, t)`.
   Unifies (a)'s sum form, time-smoothing, and band-limiting as one knob.
4. **Multi-resolution STFT (spectrogram) L2** — the defensible middle ground: amplitude-STFT
   with a bank of window lengths is *locally* shift-invariant (within a window) while
   keeping kinematics at window scale; window length is a continuous tolerance dial from
   variant 2 (window = nt) down to pointwise L2 (window → 1) — i.e. a **schedulable misfit
   family**, tailor-made for instance 2. Prior art: multi-resolution STFT losses are
   standard in neural vocoders (Yamamoto et al. 2020); wavelet-scattering metrics are
   provably stable to small deformations (Mallat 2012). Neither appears in FWI guidance;
   the closest classical relative is the envelope misfit.
5. **f–k (2-D frequency–wavenumber) magnitude misfit** *(clarified 2026-07-10 — this, not
   variant 2, is the user's intended construction)*: 2-D FFT per shot gather over
   (receiver, time); compare `|D̂_pred(ω,k)|` vs the frozen `|D̂_obs(ω,k)|` with L2 or OT.
   Materially better than variant 2: in 2-D only *global* time/space shifts are phase
   ramps, and **kinematics live in the magnitude** — a linear event of slowness p maps
   onto the line k = p·ω, so event slopes / apparent velocities / moveout are |f–k|
   content. Keeps the primary velocity signal, gains invariance to exactly the right
   nuisances (datum/origin errors), and is structurally cycle-skip-immune (no phase → no
   oscillatory mismatch). What it loses: absolute event times (t₀) are phase → weak
   sensitivity to depth-shifted interfaces at fixed velocity — a specific, testable
   null-space enlargement (probe: misfit-vs-depth-shift curve `v(z) → v(z−Δ)`, alongside
   Probe A's α-scan). **OT-on-|f–k| is the standout combination**: magnitudes are
   nonnegative by construction, so Peng's shift-to-positive hack — the exact ingredient
   our unit tests showed neuters the transport — is unnecessary, and kinematic error
   becomes smooth transport in the (ω,k) plane (slope rotation) instead of an oscillatory
   L2 residual. Practicals: taper the short 70-trace arrays (leakage); expect direct-wave
   energy to dominate → frozen spectral amplitude weighting; full 2-D Sinkhorn on
   ~70×501 grids per shot per step is heavy — start with sliced-W2 (cheap,
   differentiable) or plain/log L2 on |f–k|. Classical relatives to differentiate from:
   Radon/τ-p misfits, semblance/velocity-spectrum objectives, slope tomography, and
   Shin & Cha's Laplace/Laplace–Fourier-domain FWI (the canonical "spectral misfit vs
   cycle skipping" line). f–k-magnitude OT inside generative guidance: no prior art known,
   **needs a targeted lit check before any novelty claim**.

Bayesian caveat (applies to OT and everything above except plain/weighted L2): a non-L2
potential defines a *Gibbs/generalized posterior*, not exact Bayes under a noise model — the
calibration track's matched-σ story strictly applies to L2 only; report calibration for
robust misfits as generalized-posterior calibration.

**Implementation (misfit interface).** Extend `physics/misfit.py::make_misfit` with
`spectral` (params: `mode = complex|amp|logamp|stft`, `freq_weight`, `stft_windows`, `eps`)
and `softmin_shift` (params: shift list in samples, temperature = σ²). Frozen observation
transforms precomputed once (mirroring `OTMisfit`'s frozen-observation pattern), predicted
transform per call — `torch.fft.rfft` / `torch.stft` are differentiable and trivial next to
a wave solve (nt=1001 → 501 bins). Tests in the `tests/test_misfit.py` style: (i) Parseval
identity (variant 1 ≡ l2); (ii) amp-mode *exactly flat* vs per-trace shift — its selling
point and its indictment: also assert it cannot distinguish the trapped model of
`cycle_skipping_barrier.py` from the truth; (iii) misfit-vs-α and misfit-vs-shift curves via
the Probe-A harness (§1.5). Note each misfit's O(·) next to the σ knob — `normalize_grad=T`
absorbed OT's scale for the Tweedie methods but not for RED-DiffEq (JOURNAL 2026-07-09);
verify per variant.

**(c) OT decomposition ablation — the fastest paper-grade result in hand.** The 10/10
flow_tilt·OT win was measured under the inverse crime — *no cycle skipping present to fix* —
and the unit tests locate the long-range shift signal in the observation-frozen amplitude
weighting, not the transport. Ablate the Peng potential into {weighting only,
shift-normalize + W2 only, full} ± a designed whitening preconditioner (log-spectral or
trace-energy normalization), on both Tier 0 tracks. Either outcome is reportable: "it's the
preconditioner, not the transport" (likely, per current evidence) or "transport matters once
the lows are missing" (the cycle-skipping reading). This is the mechanism ablation Peng et
al.'s paper lacks, and it decides which ingredient the annealing schedule should anneal.

**Misfit bake-off protocol** (once §1.5 Probes B/C confirm the disease): L2 /
weighted-spectral / STFT-annealed / OT / OT-decomposed / softmin-shift / AWI / envelope
under **one** annealing framework, matched per-method tuning budget on a disjoint dev slice,
scored on basin-failure rate (cycle skipping) *and* ambiguity radius + calibration (the
uniqueness cost of tolerance) — the tension made measurable, which *is* the paper story.

---

## 3. Tier 1B — posterior calibration ladder (the thesis chapter)

The matched-σ benchmark (Tier 0) makes "which sampler is closest to the true posterior"
answerable. Climb in three rungs, cheap→exact:

1. **Held-out-shot selection** (near-free, do first): guide on shots {0,2,4}, select/score
   posterior samples on held-out shots {1,3} — honest selection that can't overfit the
   guidance data. Directly attacks the measured selection bottleneck. Also compare
   pixelwise-median/consensus ensembling.
2. **Particle repulsion** (cheap, biased): SVGD/particle-guidance kernel repulsion between
   the n parallel samples during guidance — fights posterior collapse at zero extra wave
   solves. Read out on cov90/CRPS.
3. **Twisted SMC** (asymptotically exact): N particles, existing guided dynamics as the
   twisted proposal, importance weights + resampling correct the bias exactly as N grows.
   Flow-map angle that makes it novel rather than a port: intermediate potentials
   Φ_t(x_t) evaluated with genuine one-step posterior draws (one network eval + one wave
   solve per particle) instead of the Tweedie plug-in used by diffusion-SMC papers.
   Doubles as the gold standard to *measure* the bias of DPS/flow_tilt/MFM-G.

**Hypothesis**: calibration (|cov90−0.9|, CRPS) improves monotonically up the ladder;
MFM-G's advantage over Tweedie finally appears under matched σ; SMC-vs-cheap gaps
quantify each method's bias.
**Kill criterion (per rung)**: repulsion — no cov90/CRPS improvement at n≤8 on matched-σ
targets; SMC — effective sample size collapses at affordable N (≤32) even with flow-map
twisting.
**Effort**: rung 1 small; rung 2 small-medium; rung 3 large (the headline method work).

---

## 4. Tier 2 — the compute chapter (solves budget)

Orthogonal, multiplies with everything above. Matches ideas-doc bet #2 and lever D.

1. **Stochastic source encoding / shot subsampling**: one random shot (or random-polarity
   supershot) per guidance step; the sampler's stochasticity averages crosstalk over
   ~200 steps. Includes the ideas-doc analysis question: how encoding crosstalk interacts
   with the OT normalization (crosstalk shifts the density background — our weighting
   finding says this matters).
2. **Multi-fidelity forward solves**: coarse grid / shorter record early in sampler time,
   full fidelity in the final quarter. Same coarse→fine structural argument as Tier 1A.
3. **FNO-early hybrid** (stretch within tier): FNO adjoint for early-step gradients, true
   solver late — the surrogate's gradient inaccuracy is *scheduled away* rather than fixed.

**Hypothesis**: ≥5× solve reduction at ≤5% quality loss (MAE/SSIM) on the hardened
benchmark; encoding and multi-fidelity compose.
**Kill criterion**: quality loss >15% at 5× reduction after tuning → report the trade-off
curve anyway (it's a result either way).
**Effort**: encoding small; multi-fidelity medium; FNO medium-large (training a surrogate).
**Novelty caveat (see §7)**: supergather encoding inside diffusion-FWI is already
published (arXiv:2512.12797) — this tier is an ablation/interaction chapter (single-shot
vs supergather per step, crosstalk × OT normalization, × multi-fidelity), not a
standalone claim.

---

## 5. Tier 3 — stretch / second-paper candidates

| Idea | One line | Why deferred |
|---|---|---|
| **Graph-space OT as guidance potential** | The principled OT (transport in (t,a) plane, no positivity hack) — fixes the exact weakness our unit tests exposed in the Peng potential | Assignment problem per trace per step; needs Sinkhorn/auction engineering |
| **ROM objective as early-stage potential** | Provably closer-to-convex objectives exactly where rough x̂₀ needs them | Deep math, 3D/elastic maturity unclear (ideas-doc bet #3 risk) |
| **WRI-style warm starts along the trajectory** | Treat the reverse trajectory as a continuation path for the physics solve | Speculative; solver-internals work |
| **Amortized posterior distillation** | Distill expensive (eventually SMC-grade) guided inversions into a one-step conditional flow map — "physics-verified teacher, Camp-B-speed student" | Needs Tier 1B to exist first; then it's the natural capstone |

---

## 6. Sequencing (what actually runs next on RunPod)

1. Pending L2-vs-OT smoke + comparison (closes the current work item).
2. Tier 0 hardening (band-limit, matched-σ noise, operator mismatch, OOD crops).
3. Tier 1B rung 1 (held-out-shot selection) — near-free, immediately improves all reporting.
4. Tier 1A scheduling sweep and Tier 1B rung 2 (repulsion) in parallel — both cheap.
5. Decide the paper spine on those results: scheduling story (1A-led) vs calibration story
   (1B-led); start SMC (rung 3) and the compute chapter (Tier 2) behind it.

**Paper-shaped outcomes**: (a) "Objective scheduling over generative time for FWI"
(Tier 1A + Tier 0, with OT/L2/AWI as instances); (b) "Calibrated posterior FWI via
flow-map-twisted SMC" (Tier 1B + Tier 0); (c) compute chapter strengthens either.
Negative results at kill criteria are reportable within (a)/(b).
**Amendment 2026-07-10:** paper (d) is now on the table and is the fastest — the misfit
mechanism/decomposition story (§2.1c + §1.5 Probe A figures + the L2-vs-OT hardened
comparison); and the sequencing in §6.6 supersedes the list above.

---

## 6.5 Strategic review notes (2026-07-10) — critiques logged

From the 2026-07-09/10 review sessions; each item is absorbed into a tier above or tracked
in §6.6.

1. **The flow map had quietly become optional.** Tier 1A is prior-agnostic; the flow map
   survives only in Tier 1B rung 3 (twisting potentials) — and its premise (a good
   `t_cond>0` one-step posterior) is the same unvalidated channel the JOURNAL's 2026-07-02
   PICKUP ranked as hypothesis #1 for MFM-G's underperformance. The cheap distributional
   check (one-step posterior draws vs multi-step conditional rollouts from the same `x_t`;
   no wave solves) is a **gating prerequisite** for rung 3's novelty claim. If it fails,
   the honest paper is "calibrated posterior FWI" with item 3(ii) below as the flow-native
   ingredient.
2. **Tier 0 split into calibration vs robustness tracks** (absorbed into §1): matched-σ
   calibration claims and operator-mismatch/OOD robustness claims are mutually exclusive
   per configuration.
3. **Selection ideas beyond held-out shots** (the measured bottleneck is *selection*):
   (i) held-out shots are highly correlated observations sharing the null space — they may
   not discriminate off-manifold overfitting; also try a held-out *frequency band*.
   (ii) **Exact-likelihood selection**: the flow prior is a CNF — score samples by
   `log p(v) − misfit/2σ²` with exact `log p(v)` via divergence integration (Hutchinson
   trace along the ODE; ~1 prior-cost per candidate, zero wave solves). DDPMs only get an
   ELBO — flow-native; goes in as **Tier 1B rung 1.5**.
   (iii) The **low-frequency trap screen** (`FWI_problem_exploration` finding 3: trapped
   models keep an elevated low-band misfit) as a third GT-free selection feature.
   (iv) **MFM-Search** (per-step best-of-N, `euler_sampler_tree_search` already in the mfm
   repo) was JOURNAL-flagged as untried and selection-targeted, then dropped from the
   draft — revived as the step between rungs 2 and 3 (it is the weights-only limit of the
   SMC).
4. **Make SMC produce a practical deliverable**: use SMC reference posteriors on a small
   target set to fit a cheap post-hoc recalibration of flow_tilt's posterior (temperature/σ
   rescaling or conformal intervals) — ship bias measurements *and* a recalibrated cheap
   sampler, not just a gold standard.
5. **Fair-comparison hygiene**: per-method budgeted tuning protocol on a disjoint dev slice
   (else "scheduled beats constant" is contestable), and paired statistics across the
   benchmark (per-target win rates, Wilcoxon) — the JOURNAL is full of single-seed
   within-noise deltas.
6. **Missing baseline class**: a Camp-B amortized model (InversionNet, the OpenFWI
   standard) on both Tier 0 tracks — the first question any ML reviewer asks a
   calibration paper.
7. **Cut FNO-early** (Tier 2 item 3): training a surrogate is a project in itself, sits in
   a deprioritized tier, and relies on the surrogate's known weak point (gradient
   accuracy).
8. **Scale credibility**: 70×70 crops will read as toy to GJI/Geophysics reviewers
   regardless of hardening (a 70×70 Marmousi crop loses what makes Marmousi meaningful).
   Either commit to ML venues or budget one native-resolution Marmousi demo.
9. **Two problems, two chapters — the portfolio split**: cycle skipping is a
   misfit-geometry problem (Tier 1A, *if* §1.5 shows it exists here); non-uniqueness is a
   null-space problem that **no misfit can fix** (Tier 1B + prior + extra modalities).
   Shift-tolerant misfits trade uniqueness for convexity (§2.1), so 1A and 1B are
   complements, not substitutes.

## 6.6 TODO (2026-07-10)

**Gating / this week**
- [ ] `t_cond>0` posterior-fidelity check (gates 1B rung 3 and the flow-map story; no wave solves)
- [ ] §1.5 Probe A — landscape grid under missing lows, all candidate misfits (extends `cycle_skipping_landscape.py`)
- [ ] §1.5 Probe B — classical basin-failure rate, 20 core targets, ± band-limit
- [ ] §1.5 Probe C — guided-sampler inheritance → make the Tier 1A go/no-go call
- [ ] §1.5 Probe D — misfit-vs-MAE scatters + ambiguity-radius table → `disease_report.md`
- [ ] §1.5 Probe E1 — null-space atlas (same-d_obs family via flow_tilt seeds + off-manifold fitters + classical multi-start; galleries, std maps, PCA modes, log p(v)-colored embedding)
- [ ] §1.5 Probe E2 — cycle-skipped family (flow_tilt vs `T_τ·d_obs`) + the E1-vs-E2 histogram money figure
- [ ] Close the pending L2-vs-OT run (pre-hardening leg)

**Benchmark / infrastructure**
- [ ] Tier 0 calibration track (matched-σ noise; noise before filtering, or filtered covariance in the likelihood)
- [ ] Tier 0 robustness track (band-limit knob `bench.min_freq_hz`, operator mismatch, OOD crops)
- [ ] Tuning protocol + paired statistics in the eval harness (§6.5 item 5)
- [ ] InversionNet baseline on both tracks

**Misfit design (gated on §1.5 outcome, except the ablation)**
- [ ] OT decomposition ablation (§2.1c) — runnable now on the current benchmark, rerun on both tracks
- [ ] `misfit=spectral` (complex/amp/logamp/stft/**fk_amp/fk_ot**) + `misfit=softmin_shift` + unit tests (Parseval identity, shift-flatness, trapped-model blindness; f–k: depth-shift blindness curve)
- [ ] Lit check: f–k/Radon-domain misfits, spectral-magnitude OT, Shin & Cha Laplace-domain FWI — before any §2.1b-variant-5 novelty claim
- [ ] Misfit bake-off under one annealing framework (§2.1 protocol) — only if Probes B/C show basin failures
- [ ] AWI as misfit *and* controller (upgraded from controller-only)

**Posterior / selection (Tier 1B)**
- [ ] Rung 1: held-out-shot selection (+ held-out frequency-band variant)
- [ ] Rung 1.5: exact-likelihood selection (CNF `log p(v)`) + low-freq trap screen as selection features
- [ ] Read arXiv:2601.23262 **before** rung 2; then particle repulsion / MFM-Search
- [ ] Rung 3 SMC only after the `t_cond` check passes; include the distilled-recalibration deliverable (§6.5 item 4)

Decide the paper spine after the §1.5 decision call + the OT ablation — not before.

---

## 7. Novelty check — literature sweep (2026-07-09)

Findings that *change* our claims are bolded. Verify abstracts before citing in a paper.

**Diffusion-prior FWI is now a crowded, fast-moving lane** (as the ideas doc predicted):
Taufik & Alkhalifah's diffusion posterior sampling for FWI (arXiv:2512.12797), DiffusionInv
(arXiv:2505.03138, fine-tunes the diffusion model with FWI guidance), reconstruction-guided
diffusion for Bayesian FWI (GJI 2026, ggag066), decoupled latent optimization
(arXiv:2606.14139), joint velocity–slope diffusion priors (arXiv:2607.04982), Peng et al.
(arXiv:2603.16393), and an NVIDIA PhysicsNeMo reference example. Consequence: *prior +
plain guidance* is table stakes; our contribution must be the steering method / benchmark
rigor, which is what Tiers 1A/1B are.

**Tier 2 source encoding: the headline is partially taken.**
**arXiv:2512.12797 already couples diffusion posterior sampling with simultaneous-source
supergathers** (solve count reduced ~proportionally to supergather size, Langevin
refinement in clean space). Random-shot minibatch FWI is also classical
(e.g. J. Earth Syst. Sci. 2021). Remaining gaps: per-step *random single-shot* guidance vs
supergathers inside the reverse process; the crosstalk × OT-normalization interaction
(ideas-doc bet #2 — still unclaimed); composition with multi-fidelity. Reposition Tier 2
as an ablation/interaction chapter, not a standalone claim.

**Tier 1A frequency annealing: the generic mechanism exists in image restoration.**
**Frequency-Guided Posterior Sampling (arXiv:2411.15295) applies a time-varying low-pass
filter to measurements during reverse diffusion**, matching the coarse-to-fine spectral
emergence of diffusion sampling. Scheduled/piecewise guidance also exists generically
(SPIN, arXiv:2603.07860; piecewise guidance, arXiv:2507.18654). No FWI instantiation
found: the cycle-skipping/missing-lows analysis, the Bunks-continuation connection, and
the misfit-annealing + AWI-controller instances appear open. Position 1A as "objective
scheduling for wave-equation guidance" citing 2411.15295 as the imaging precedent, with
the *controller* (physics-native AWI trigger) as the novel core rather than the filter
schedule alone.

**Tier 1A AWI controller: open.** Matching filters meet ML as *misfit surrogates*
(DLM-FWI, GJI 2026; SEG 2025 dynamic matching filtering; DL-AWI) — none use the filter as
a guidance-strength/stage controller inside a generative sampler.

**Tier 1B: strongest surviving gap.** SMC×diffusion exists for generic (mostly linear)
inverse problems — MCGDiff (Feynman-Kac formulation), Twisted Diffusion Sampler, latent
SMC (arXiv:2502.05908), decoupled diffusion SMC (arXiv:2502.06379) — but no nonlinear
PDE-constrained/FWI application found, and none use flow-map one-step posteriors as
twisting potentials. SVGD exists in FWI as classical UQ (annealed SVGD,
arXiv:2410.13249) but not as repulsion between guided sampler trajectories. The matched-σ
calibration benchmark + bias measurement of DPS/MFM-G against an SMC gold standard
appears unclaimed.

**Tier 3 ROM: line is alive and healthy** (SIAM Review 2024 "When data-driven ROM meets
FWI"; SIAM J. Imaging Sci. 2025 first-order hyperbolic systems / multiparameter acoustic;
convexity-vs-FWI-misfit results explicit). No generative-prior combination found — the
gap stands, effort estimate unchanged (deep math).

**OT misfits: mature in classical FWI** (Engquist & Froese; Métivier graph-space,
industrially proven per Viridien/CGG First Break 2021; shallow-seismic Geophysics 2024) —
confirming that OT-*inside-generative-guidance* comparisons, not new OT variants, are
where our misfit work should point.

**Spectral / deformation-stable misfits (2026-07-10 additions).** Attribute misfits are
classical: envelope & instantaneous phase (Bozdağ, Trampert & Tromp 2011),
cross-correlation traveltime (Luo & Schuster 1991), DTW (Ma & Hale 2013);
fitting-up-to-shift with a shift penalty is the extension-method line (Symes;
Biondi & Almomin's TFWI) and AWI. Multi-resolution STFT losses are standard in audio
synthesis (Yamamoto et al. 2020) and scattering transforms give provable deformation
stability (Mallat 2012) — neither found in FWI guidance (§2.1b variant 4 gap).
**Nuisance-marginalized likelihoods inside generative guidance** (datum / wavelet / gain
nuisances in the potential): no prior art found in the diffusion-FWI lane — the surviving
novel kernel of the shifted-copies idea (§2.1a).

## 8. References

Grouped by the tier they support. Links captured in the 2026-07-09 sweep; entries marked
*(verify)* are recalled from memory — confirm the ID before citing in a manuscript.

### Diffusion/generative priors for FWI (context; Tiers 0–1 positioning)

- Peng, Jiang, Ma & Yan 2026 — *Robust Physics-Guided Diffusion for Full-Waveform
  Inversion* — [arXiv:2603.16393](https://arxiv.org/abs/2603.16393). Our OT-potential
  reference point (`method.misfit=ot`); PDF in `docs/papers/`.
- Taufik & Alkhalifah 2025 — *Diffusion Model-Based Posterior Sampling in FWI* —
  [arXiv:2512.12797](https://arxiv.org/abs/2512.12797). **Already does supergather source
  encoding inside diffusion FWI** — the Tier 2 repositioning driver.
- *DiffusionInv: Prior-enhanced Bayesian FWI using diffusion models* —
  [arXiv:2505.03138](https://arxiv.org/abs/2505.03138) (fine-tunes the prior with FWI guidance).
- *Accelerating Bayesian FWI using reconstruction-guided diffusion sampling* — GJI 2026 —
  [ggag066](https://academic.oup.com/gji/article/245/2/ggag066/8487641). Closest prior art
  to Tier 1B's Bayesian framing.
- *Decoupled Latent Optimization of Diffusion Models for FWI* —
  [arXiv:2606.14139](https://arxiv.org/abs/2606.14139).
- *Joint Velocity–Slope Diffusion Prior for Structurally Constrained Velocity Model
  Building* — [arXiv:2607.04982](https://arxiv.org/abs/2607.04982).
- Wang & Alkhalifah — *A prior regularized FWI using generative diffusion models* —
  [arXiv:2306.12776](https://arxiv.org/abs/2306.12776) (the early entry in the lane).
- *Improving FWI in the Large Model Era* — [arXiv:2603.00377](https://arxiv.org/html/2603.00377v2).
- NVIDIA PhysicsNeMo diffusion-FWI reference example —
  [docs](https://docs.nvidia.com/physicsnemo/26.03/physicsnemo/examples/geophysics/diffusion_fwi/README.html).
- RED-DiffEq — arXiv:2509.21659 *(verify)*; DiffusionVel — arXiv:2410.21776 *(verify)* —
  both already cited in `research-plan.html`.
- Potaptchik et al. 2026 — *Meta Flow Maps enable scalable reward alignment* —
  arXiv:2601.14430; PDF in `docs/papers/`. The steering estimators we build on.
- Huang et al. 2026 (ICML) — FMRG — see `inversion/fmrg.py` header *(no link on file)*.

### Misfit design / OT (Tiers 0, 1A, 3)

- Engquist & Froese 2014 — *Application of the Wasserstein metric to seismic signals* —
  Commun. Math. Sci. 12(5). The W2-convexity-in-shift result (for nonnegative signals).
- Yang, Engquist et al. 2018 — *Application of OT and the quadratic Wasserstein metric to
  FWI* — Geophysics — [geo2016-0663.1](https://library.seg.org/doi/abs/10.1190/geo2016-0663.1).
  The trace-wise shift-normalize construction Peng et al. inherit.
- Métivier, Brossier et al. 2016 — *An optimal transport approach for seismic tomography* /
  OT misfit — GJI 205(1). KR-norm line; critique of linear shift normalization.
- Métivier et al. 2018–2019 — graph-space OT — GJI / Geophysics *(verify exact refs)*;
  industrial validation: Messud et al. 2021 — *OT FWI: from theory to industrial
  applications* — [First Break / Viridien PDF](https://www.viridiengroup.com/sites/default/files/2021-12/First%20Break%20December%202021%20CGG%20Messud%20et%20al%20Final%20published.pdf).
- *Optimal transport-based FWI for shallow seismic data* — Geophysics 2024 —
  [geo2024-0165.1](https://library.seg.org/doi/10.1190/geo2024-0165.1).
- Warner & Guasch 2016 — *Adaptive waveform inversion: theory* — Geophysics 81(6). The
  Wiener matching filter behind the Tier 1A controller.
- Sun & Alkhalifah — matching-filter OT (MSOT) *(verify ref)*.
- Bunks et al. 1995 — *Multiscale seismic waveform inversion* — Geophysics 60(5). The
  frequency-continuation classic behind Tier 1A (and what §2.1a's sum form collapses to).
- Bozdağ, Trampert & Tromp 2011 — *Misfit functions for FWI based on instantaneous phase
  and envelope measurements* — GJI 185(2). The attribute-misfit classics (§2.1b).
- Luo & Schuster 1991 — *Wave-equation traveltime inversion* — Geophysics 56(5). The
  original fit-up-to-shift (cross-correlation) misfit.
- Ma & Hale 2013 — *Wave-equation reflection traveltime inversion with dynamic warping and
  FWI* — Geophysics 78(6). DTW misfit.
- Symes 2008 — *Migration velocity analysis and waveform inversion* — Geophys. Prospecting
  56(6) *(verify)*; Biondi & Almomin 2014 — *Simultaneous inversion of full data bandwidth
  by tomographic FWI* — Geophysics 79(3) *(verify)*. Extension methods / time-shift-extended
  FWI — §2.1a's nearest classical relatives.
- Yamamoto et al. 2020 — *Parallel WaveGAN* — ICASSP. Multi-resolution STFT loss — the
  audio precedent for §2.1b variant 4.
- Mallat 2012 — *Group invariant scattering* — Comm. Pure Appl. Math. 65(10).
  Deformation-stable spectral metrics (§2.1b).
- *DLM-FWI: deep learning matching filtering for FWI* — GJI 2026 —
  [ggag019](https://academic.oup.com/gji/article/245/1/ggag019/8427320); also SEG IMAGE25
  [dynamic matching filtering](https://onepetro.org/SEGAM/proceedings-abstract/IMAGE25/IMAGE25/SEG-2025-4315365/798949)
  and [DL-AWI](https://www.mdpi.com/2076-3263/16/2/65). Matching filters as misfit
  surrogates — none as a guidance controller (the Tier 1A gap).

### Guidance scheduling precedents (Tier 1A)

- *Frequency-Guided Posterior Sampling for Diffusion-Based Image Restoration* —
  [arXiv:2411.15295](https://arxiv.org/html/2411.15295v2). **The imaging precedent for
  frequency annealing** — cite and differentiate (FWI, cycle skipping, missing lows).
- SPIN — *Sparse Scheduled Diffusion Guidance for Inverse Problems* —
  [arXiv:2603.07860](https://arxiv.org/abs/2603.07860).
- *Posterior sampling with piecewise guidance* — [arXiv:2507.18654](https://arxiv.org/html/2507.18654).

### SMC / particle methods / UQ (Tier 1B)

- Wu, Trippe, Naesseth, Blei & Cunningham 2023 (NeurIPS) — *Practical and Asymptotically
  Exact Conditional Sampling in Diffusion Models* (Twisted Diffusion Sampler) —
  [arXiv:2306.17775](https://arxiv.org/abs/2306.17775),
  [code](https://github.com/blt2114/twisted_diffusion_sampler).
- Cardoso et al. — MCGDiff, *Monte Carlo guided Diffusion for Bayesian linear inverse
  problems* — [OpenReview](https://openreview.net/forum?id=nHESwXvxWK) / arXiv:2308.07983 *(verify)*.
- *Inverse Problem Sampling in Latent Space Using SMC* —
  [arXiv:2502.05908](https://arxiv.org/abs/2502.05908).
- *Decoupled Diffusion SMC for linear-Gaussian inverse problems* —
  [arXiv:2502.06379](https://arxiv.org/html/2502.06379v1).
- *Annealed SVGD for improved uncertainty estimation in FWI* —
  [arXiv:2410.13249](https://arxiv.org/pdf/2410.13249). SVGD in classical FWI — the
  Tier 1B rung-2 differentiation point.
- Liu & Wang 2016 (NeurIPS) — *Stein Variational Gradient Descent* — arXiv:1608.04471.
- Corso, Xu, De Bortoli, Barzilay & Jaakkola 2023 — *Particle Guidance: non-I.I.D. Diverse
  Sampling with Diffusion Models* — [arXiv:2310.13102](https://arxiv.org/abs/2310.13102),
  [code](https://github.com/gcorso/particle-guidance).
- **⚠ *Particle-Guided Diffusion Models for PDEs* —
  [arXiv:2601.23262](https://arxiv.org/abs/2601.23262) (2026). Found during reference
  verification — potentially close prior art for Tier 1B rung 2 (repulsion between guided
  trajectories in a PDE setting). READ BEFORE claiming rung-2 novelty.** Related:
  *Marginal-Preserving Particle Guidance* — [arXiv:2605.06553](https://arxiv.org/html/2605.06553v1);
  *Particle Denoising Diffusion Sampler* — [arXiv:2402.06320](https://arxiv.org/html/2402.06320v2).

### Compute: encoding, relaxation, ROM, surrogates (Tiers 2–3)

- Krebs et al. 2009 — *Fast full-wavefield seismic inversion using encoded sources* —
  Geophysics 74(6). The supershot/random-polarity classic.
- *FWI with random shot selection using adaptive gradient descent* —
  [arXiv:2005.09899](https://arxiv.org/pdf/2005.09899) /
  [J. Earth Syst. Sci. 2021](https://link.springer.com/article/10.1007/s12040-021-01679-y).
- van Leeuwen & Herrmann 2013 — Wavefield Reconstruction Inversion — GJI *(verify)*;
  IR-WRI: Aghamiry et al. — [arXiv:1809.00891](https://arxiv.org/pdf/1809.00891).
- Borcea, Garnier, Mamonov & Zimmerling 2024 — *When data-driven ROM meets FWI* —
  [SIAM Review 66](https://epubs.siam.org/doi/10.1137/23M1552826); 2025 first-order
  hyperbolic / multiparameter acoustic — [SIAM J. Imaging Sci.](https://epubs.siam.org/doi/10.1137/24M1699784)
  / [arXiv:2505.08937](https://arxiv.org/pdf/2505.08937); the original —
  [arXiv:2202.01824](https://arxiv.org/abs/2202.01824). Convexity-vs-L2 results explicit.
- Li et al. 2020 — Fourier Neural Operator — arXiv:2010.08895 *(verify)* (Tier 2 FNO-early).
- Richardson — Deepwave (the differentiable solver everything here runs on) —
  [github](https://github.com/ar4/deepwave).

## 9. Decision log

- 2026-07-09 — Document drafted from the two brainstorm sessions (Claude + user ideas doc).
  Positioning fixed: Camp A, ML-side, synthetic-but-hard. Tier 1 = scheduling + calibration.
- 2026-07-09 — Lit sweep (§7): source-encoding headline partially taken (2512.12797),
  frequency-annealing mechanism has an imaging precedent (2411.15295) → Tier 1A's novel
  core shifts to the AWI controller + FWI/missing-lows analysis; Tier 1B (flow-map-twisted
  SMC + matched-σ calibration) confirmed as the strongest clean gap.
- 2026-07-10 — Review session absorbed. Tier 0 split into calibration/robustness tracks
  (§1); §1.5 disease-existence check added as the Tier 1A gate — `FWI_problem_exploration`
  finding 4 (on the current geometry the binding pathology is non-uniqueness, not cycle
  skipping; missing-lows regime is the open question) makes this non-optional. §2.1
  misfit-design review added: the tolerance–uniqueness tension named as the design
  principle; shifted-copies idea shown to collapse to frequency continuation in sum form,
  corrected to softmin nuisance marginalization (novel kernel: nuisance-marginalized
  guidance); spectral L2 triaged (complex ≡ L2 by Parseval; amplitude-only disqualified
  standalone — total kinematic blindness; frequency-weighted = the clean frequency-annealing
  implementation; multi-res STFT = the schedulable middle ground, audio precedent, open in
  FWI guidance); OT decomposition ablation named the fastest paper-grade result. §6.5/§6.6
  strategic critiques + TODO logged: `t_cond>0` fidelity check gates rung 3;
  exact-likelihood (CNF) selection added as rung 1.5; low-freq trap screen as a selection
  feature; MFM-Search revived; SMC distilled-recalibration deliverable; tuning-protocol +
  paired-statistics requirement; InversionNet baseline; FNO-early cut; scale-credibility
  caveat.
- 2026-07-10 (later) — Spectral misfit clarified as the 2-D f–k construction (variant 5,
  §2.1b): magnitude keeps kinematics (event slopes), invariance shrinks to global shifts,
  structurally cycle-skip-immune; known cost is t₀/depth-registration blindness (phase).
  OT-on-|f–k| flagged as the standout candidate — nonnegativity for free removes the Peng
  positivity hack that the unit tests showed neuters the transport. Added to Probe A +
  TODO with a pre-claim lit check (Radon/τ-p, semblance, slope tomography, Shin & Cha
  Laplace-domain).
- 2026-07-10 (later) — §1.5 opening claim challenged and corrected: the MFM-G arc was ~5
  days not "weeks", and the benchmark *did* reward guidance (−40% MAE) and misfit design —
  what it couldn't reward is the posterior-quality edge beyond Tweedie (no principled σ →
  no well-defined posterior target), and that diagnosis remains confounded with the
  untested t_cond>0 channel. Evidence now cited inline (the five JOURNAL entries + the
  barrier/nonuniqueness studies). Probe E added: disease atlases generated *by the samplers
  themselves* — E1 same-d_obs null-space atlas (flow_tilt seeds + off-manifold fitters as
  features + classical multi-start; galleries, pixelwise std, PCA null-space modes,
  log p(v)-colored embedding), E2 cycle-skipped family via flow_tilt against T_τ·d_obs
  (the diagnostic twin of §2.1a), and the E1-vs-E2 MAE histogram as the disease-report
  money figure.
