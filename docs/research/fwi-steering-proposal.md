# Steering-Method Proposal — the 0004 inversion program

*Draft 2026-07-09 (Claude-drafted for curation; edit freely). Companion documents:*
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

Also close out the **pending L2-vs-OT run** (flow_tilt + mfm_g on map 6044 and a benchmark
slice, matched solves; retune σ for OT's O(1) scale) — before and after hardening, since
the hypothesis is that the OT edge appears only *after* hardening.

**Effort**: small (filters, noise, a second simulate config, data plumbing).
**Kill criterion**: none — this tier is infrastructure.

---

## 2. Tier 1A — objective scheduling over generative time (cycle skipping)

**One story, three instances**: generation is coarse→fine, so make the *objective*
coarse→fine along sampler time t. Subsumes the "staged/annealed strategy" cross-cutting
angle in `fwi-improvement-ideas.md` §2.

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

## 8. Decision log

- 2026-07-09 — Document drafted from the two brainstorm sessions (Claude + user ideas doc).
  Positioning fixed: Camp A, ML-side, synthetic-but-hard. Tier 1 = scheduling + calibration.
- 2026-07-09 — Lit sweep (§7): source-encoding headline partially taken (2512.12797),
  frequency-annealing mechanism has an imaging precedent (2411.15295) → Tier 1A's novel
  core shifts to the AWI controller + FWI/missing-lows analysis; Tier 1B (flow-map-twisted
  SMC + matched-σ calibration) confirmed as the strongest clean gap.
