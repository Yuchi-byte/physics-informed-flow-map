# FWI Improvement — Brainstorm & Method Comparison

Working document for finding a promising, publishable direction to improve full waveform inversion (FWI).
Reference point: Peng et al., scaled-normalised optimal transport (OT) misfit.

**Target problems:** cycle skipping · computation time · null space / non-uniqueness · uncertainty.

**Framing:** FWI has four levers — the **misfit**, the **search space**, the **prior**, and the **solver**.
Standalone contributions on any single lever are crowded; the publishable gaps are mostly *combinations of two levers* with honest field-data benchmarks.

---

## 1. Candidate directions

### A. Misfit functions (cycle skipping)

| Method | Key refs | Idea | Status / gap |
|---|---|---|---|
| Scaled-normalised OT | Peng et al. | Normalise signed seismic data into distributions for W2 | Our reference point |
| Graph-space OT | Métivier, Brossier et al. | Lift traces to (t, amplitude) point clouds; avoids positivity hack | Industrially proven (Viridien) — natural baseline to beat |
| Adaptive Waveform Inversion (AWI) | Warner & Guasch | Wiener matching filter, penalise deviation from delta | Commercial (S-Cube); AWI-vs-OT under realistic noise is under-studied |
| Matching-filter OT (MSOT) | Sun & Alkhalifah | Apply OT to the matching filter, not the data | Sidesteps OT positivity issues more naturally |
| Envelope / instantaneous phase / DTW | Bozdağ; Ma & Hale | Kinematics-focused attributes | Mostly useful as stage-1 objectives in multiscale strategies |
| Spectral L2 |  | Misfit in Fourier space | time-shift that causes cycle skipping might not affect the frequency spectrum |

> Assessment: yet another standalone misfit is a hard sell in 2026. Misfit **+ something else** is where papers are.

### B. Extended / relaxed formulations (search space)

- **Wavefield Reconstruction Inversion (WRI)** — van Leeuwen & Herrmann; penalty/ADMM variants (IR-WRI, Aghamiry et al.). Relax the wave-equation constraint → convexifies the problem. **Gap:** efficient *time-domain* 3D versions.
- **Source-extension methods** — Symes; Huang. Same spirit, extend the source instead of the wavefield.
- **Reflection FWI (RWI)** — separates tomographic vs. migration updates; deep macromodel building where diving waves don't reach.

### C. Machine learning (hottest publishable territory)

- **Diffusion-model priors as plug-and-play regularisers** inside classical FWI iterations. Most active direction 2025–26; composes with *any* misfit, including scaled-normalised OT. **Gap:** convincing field-data demonstrations.
- **Low-frequency extrapolation** (Sun & Demanet; Ovcharenko et al.) — synthesise the missing lows that cause cycle skipping, then standard multiscale FWI. **Gap:** out-of-distribution robustness on field data.
- **Network reparameterisation / deep image prior / implicit neural representations (SIREN)** — invert network weights instead of grid values; implicit smoothness aids convexity. Cheap to combine.
- **Neural-operator surrogates (FNO family)** for the forward model — the ML answer to compute time. **Weak point:** gradient accuracy. Hybrid (surrogate early, full solver late) is publishable.
- **PINN-based FWI** — caution: scales poorly to realistic 3D; reviewers know it.

### D. Computation time (orthogonal → combines with everything)

- **Source encoding / stochastic shot minibatching** — supershots with random encoding; crosstalk noise is the cost. **OT + encoding is under-explored** (transport plan × crosstalk interaction).
- **Reduced-order models (ROM)** — Druskin, Borcea et al. Data-driven ROM objectives are *provably closer to convex* AND cheap. Deep math, uncrowded, directly competes with OT on the cycle-skipping claim.
- Engineering enablers (not papers by themselves): checkpointing vs. random boundaries, GPU differentiable frameworks (Deepwave, JUDI, jwave).

### E. Uncertainty quantification (reviewer differentiator)

- Bayesian FWI via SVGD, variational inference, randomised MAP. "How wrong is your model?" is increasingly demanded. UQ + robust misfit = differentiator.

---

## 2. Shortlisted combination bets

| # | Combination | Novelty | Risk | Compute story | One-line pitch |
|---|---|---|---|---|---|
| 1 | Diffusion prior + transport-based misfit | High | Medium (field data) | Prior is cheap at inference | Prior fixes null space & missing lows; OT fixes kinematic errors — nobody has shown both on field data |
| 2 | Scaled-normalised OT + stochastic source encoding | Medium-high | Medium (crosstalk × normalisation) | Direct speedup | "Make Peng et al. fast" — analyse how normalisation interacts with encoding crosstalk |
| 3 | ROM objective vs./+ OT | High (theory angle) | Medium-high (3D maturity) | ROM is cheap | Less crowded, provable convexity claims, strong-theory paper |
| 4 | UQ (SVGD / rand-MAP) on top of robust misfit | Medium | Low | Expensive — needs #2-style acceleration | Uncertainty-aware cycle-skip-robust FWI |
| 5 | Learned low-freq extrapolation + OT safety net | Medium | Medium (OOD data) | Cheap add-on | OT as robustness insurance for imperfect extrapolated lows |

**Cross-cutting angle:** cycle-skip robustness and compute cost trade off (robust misfits & extended methods cost more per iteration). A *staged/annealed* strategy — cheap robust objective early, least-squares late — with explicit trade-off management is itself a clear paper story.

---

## 3. Evaluation criteria (for comparing candidates)

- Cycle-skipping robustness (poor starting model, no lows below ~4 Hz)
- Wall-clock / cost per iteration and to convergence
- Field-data readiness (noise, elastic effects, acquisition gaps)
- Theoretical grounding (convexity claims, convergence guarantees)
- Novelty vs. 2023–2026 literature (needs a sweep — see open questions)
- Implementation lift (can we prototype in Deepwave/JUDI quickly?)

## 4. Open questions / next actions

- [ ] Literature sweep (2023–2026) on diffusion-prior FWI — what exactly has been published, which misfits used, any field data?
- [ ] Same for ROM-based FWI (Borcea/Druskin line) — 3D status? elastic?
- [ ] Does anyone combine OT misfits with source encoding? Check SEG/EAGE 2024–2026 abstracts.
- [ ] Benchmark setup: Marmousi2 + BP 2004 salt with truncated lows, then one field dataset.
- [ ] Which forward-modelling framework to standardise on for prototypes?

## 5. Decision log

*(record choices + reasons as we narrow down)*

- 2026-07-06 — Document created; initial landscape mapped from discussion.
