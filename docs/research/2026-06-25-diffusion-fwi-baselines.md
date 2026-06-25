# Diffusion / flow models for FWI — baselines + PhysicsNeMo assessment

**Date:** 2026-06-25. Research notes (literature + tooling scan) for positioning this
project's contribution and choosing baselines. Compiled from a web literature review.

## TL;DR

- Our differentiator is **camp C**: PIDM-style **physics-residual losses in *training*** on a
  **flow-matching** velocity-map prior. Almost all FWI generative work puts physics at
  *inference* (camp A) or uses no physics (camp B). That gap is the defensible novelty.
- The prerequisite for any conditional / posterior-sampling FWI is a differentiable
  **wave-equation forward operator** — standard tool is **Deepwave** (PyTorch). We have
  only velocity maps on disk, not seismic `d`; `d` must be downloaded or synthesized via
  Deepwave.
- **NVIDIA PhysicsNeMo: do not adopt.** Score/EDM-diffusion only (no flow matching). Mine
  its `examples/geophysics/diffusion_fwi` as a *reference recipe* (DPS + Deepwave); consider
  Deepwave itself. It won't meaningfully reduce our code.

## The trigger paper — RED-DiffEq

"Regularization by denoising diffusion models for solving inverse PDE problems with
application to full waveform inversion." Shan, Zhu, **Lin (OpenFWI senior author)**, **Lu Lu
(DeepONet)**. *Communications Physics* 2026. arXiv: 2509.21659.

- **Method:** unconditional diffusion prior trained on OpenFWI velocity models, used **at
  inference** as a Regularization-by-Denoising (RED) term in the FWI objective:
  `min_v  data_misfit(F(v), d)  +  λ · denoiser_residual(v)`. Diffusion model **not
  retrained** — a plug-in prior. RED-style (denoiser residual), a cousin of DPS.
- **Category A** (prior + inference-time guidance).
- Trained on OpenFWI; **generalizes zero-shot to Marmousi and Overthrust** (headline).
- Relation to us: same goal, different axis — they do diffusion + inference regularization;
  we propose flow-matching prior + PIDM physics-in-training.

## Taxonomy of diffusion-for-FWI

**A. Unconditional prior + inference-time likelihood guidance** (dominant in geophysics/UQ)
- Train `p(v)` offline; steer reverse sampling / optimization with the wave-equation data
  misfit (DPS gradient) or a denoiser regularizer (RED).
- Papers: RED-DiffEq (2509.21659); DPS-FWI / "A prior regularized FWI using generative
  diffusion models" (2306.12776, IEEE TGRS 2023); Robust Physics-Guided Diffusion —
  Wasserstein-2 guidance (2603.16393, 2026); Diffusion Posterior Sampling in FWI
  (2512.12797); DiffusionInv — Bayesian fine-tune (2505.03138); SLIM-group Bayesian/UQ
  work — WISER (2405.10327), ML velocity + UQ (2411.06651).
- Strength: principled posterior / UQ, no paired data. Weakness: a wave-equation solve at
  every step (expensive); DPS approximation can break for strongly nonlinear operators.

**B. Conditional / amortized (seismic → velocity in one pass)** (dominant on OpenFWI)
- Wave equation only generates training data; inference is one forward pass.
- Papers: DiffusionVel (2410.21776) — best published OpenFWI conditional-diffusion numbers;
  Conditional Rectified Flow (2603.15354) — **flow matching**, closest to us; I2SB
  Schrödinger bridge (2506.15346) — code + all 10 families; Geological/Well prior CFG
  (2412.06959); Controllable Synthesis (2402.06277).
- Strength: fast, benchmark-friendly. Weakness: amortization gap, needs large paired data,
  OOD-fragile.

**C. Physics residual in *training* (PIDM-style)** (nascent — our angle)
- Add PDE-residual loss to the generative model's training objective.
- Papers: PIDM (2403.14404, ICLR 2025 — the parent method); WaveDiffusion (2410.09002 —
  physics consistency emerges from joint latent diffusion, not an explicit residual).
- Status: very few apply explicit PDE-residual *training* losses to a generative FWI model.

**Where we sit:** flow-map prior (camp B machinery) + PIDM physics-in-training (camp C).

## OpenFWI as a benchmark

- 12 sub-datasets (Vel: FlatVel/CurveVel A/B; Fault: FlatFault/CurveFault A/B; Style A/B;
  CO2). Repo: github.com/lanl/OpenFWI. CC BY-NC-SA 4.0.
- Standard metrics: **MAE, RMSE/MSE, SSIM** on velocity maps. No generative paper surveyed
  reports FID / diversity — reporting calibrated posterior **diversity** would distinguish a
  `p(v|d)` method.
- Reference numbers (from DiffusionVel, the most-cited comparison table):

  | Method | FlatVel-B MAE | FlatVel-B SSIM | CurveVel-B MAE | CurveVel-B SSIM |
  |---|---|---|---|---|
  | Conventional FWI | 0.0846 | 0.7580 | 0.1760 | 0.6019 |
  | InversionNet | 0.0437 | 0.9343 | 0.1695 | 0.6604 |
  | VelocityGAN | 0.0700 | 0.8713 | 0.1651 | 0.6659 |
  | **DiffusionVel** | **0.0232** | **0.9738** | **0.0983** | **0.8331** |

  Deterministic SOTA ceiling (SPAMoE, 2604.07421): FlatVel-A MAE 0.0035 / SSIM 0.9982.

## Recommended baselines for our work

1. **InversionNet + VelocityGAN** — OpenFWI official supervised baselines (LANL repo).
2. **DiffusionVel** (2410.21776) — best conditional-diffusion OpenFWI numbers (no official repo found).
3. **Conditional Rectified Flow (2603.15354) + I2SB (2506.15346)** — closest to flow matching;
   I2SB code: github.com/stankevich-mipt/seismic_inversion_via_I2SB (all 10 families).
4. **RED-DiffEq (2509.21659) + a DPS-prior (2306.12776)** — the camp-A comparison for any
   physics-guided posterior we build.

## Foundational (non-seismic) methods our approach builds on

- **DPS** — Diffusion Posterior Sampling, Chung et al., ICLR 2023 (2209.14687). Code:
  github.com/DPS2022/diffusion-posterior-sampling.
- **Score-based inverse problems** — Song et al., ICLR 2022 (2111.08005). The template FWI
  camp-A papers copy.
- **Score-based SDEs** — Song et al., ICLR 2021 (2011.13456).
- **Flow Matching** — Lipman et al., ICLR 2023. Basis for the rectified-flow / PIS lines.
- **PIDM** — Bastek, Sun, Kochmann, ICLR 2025 (2403.14404). Direct parent of our camp-C angle.

## NVIDIA PhysicsNeMo (formerly Modulus) — assessment

- **What it is:** Apache-2.0 physics-ML toolkit (`pip install nvidia-physicsnemo`, v2.1.1,
  June 2026). Library of composable modules, **not** a framework you migrate into — plug
  pieces into your own PyTorch loop. Python 3.11–3.13 (we're on 3.12, fine).
- **Diffusion module** (`physicsnemo.diffusion`): EDM/score-family schedulers, ODE/SDE
  samplers, EDM preconditioners, and a stable **`DiT`** backbone. **No flow matching / no
  stochastic interpolants / no rectified flow.**
- **PDE utilities** (`physicsnemo.sym.PhysicsInformer`): symbolic-autodiff PDE residuals —
  good for coordinate PINNs, **not** for FWI's forward-simulation residuals.
- **One geoscience example:** `examples/geophysics/diffusion_fwi` (v2.1, May 2026) — DPS-style
  physics-guided posterior sampling, forward modeling via **Deepwave**, on the E-FWI dataset
  (not OpenFWI), score-based (not flow). Best available reference recipe for `p(v|d)`.
- **Verdict:** **skip wholesale adoption** — its diffusion stack is score-based, so using it
  means dropping our flow-map foundation; its DiT overlaps mfm's working backbone; its sym
  tools don't fit FWI. **Cherry-pick** the `diffusion_fwi` example as a *design reference*
  and consider **Deepwave** as the forward solver. Do not add it as a dependency.

## Concrete next steps this implies (not yet scoped)

1. **Forward operator:** add **Deepwave** (acoustic wave eq) to generate seismic `d` from
   velocity maps and to compute data-misfit gradients. Prerequisite for everything below.
2. **Camp-C experiment (our novelty):** PIDM-style physics-residual term added to the
   flow-matching training objective on OpenFWI velocity maps.
3. **Camp-A comparison:** RED/DPS-style inference-time guidance using our flow-map prior +
   Deepwave, benchmarked against RED-DiffEq.
4. Evaluate on OpenFWI with MAE/RMSE/SSIM **and** posterior diversity/calibration.
