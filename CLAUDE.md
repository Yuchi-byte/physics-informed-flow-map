# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This repo is a **uv workspace** (virtual root: tooling only, packages under `packages/`).

```bash
uv sync                        # install all workspace members + dev tools
uv run python <script.py>      # run a script in the workspace venv
```

`uv sync` auto-detects the CUDA driver and picks matching PyTorch wheels
(`[tool.uv] torch-backend = "auto"` in the root `pyproject.toml`), falling back
to CPU when no GPU is present.

Experiments live in `experiments/` (contract: `experiments/README.md`). Each
numbered `NNNN_slug/` directory is a framework; runs land in the git-ignored
`runs/`:
```bash
uv run python experiments/0001_flow_matching/run.py experiment=smoke        # fast plumbing check
uv run python experiments/0001_flow_matching/run.py experiment=mnist n_steps=500
uv run python experiments/new.py "short title"                              # scaffold a new framework
```

For the MFM reference package (run from `packages/mfm-meta-flow-map-main/`):
```bash
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py        # train
torchrun --nnodes=1 --nproc_per_node=1 scripts/sample.py       # sample
python evaluations/evaluator.py evaluations/VIRTUAL_imagenet256_labeled.npz samples.npz  # FID
```

For the PIDM reference package (run from `packages/PhysicsInformedDiffusionModels-main/`):
```bash
python main_toy.py    # toy study (hypersphere constraint)
python main.py        # Darcy flow / topology optimization
python sample.py      # evaluate trained models
```

## Architecture

This project combines two research approaches into a new method:

**`packages/physics-informed-flow-map/`** — the new package under development (`src/physics_informed_flow_map/`). This is where the combined method will be implemented. It currently ships the shared experiment harness (`experiment/`: typed pydantic `Config` + manifest-pinned `Run` lifecycle) consumed by the `experiments/` frameworks.

**`packages/mfm-meta-flow-map-main/`** — [Meta Flow Maps (ICML 2026)](https://arxiv.org/abs/2601.14430) reference implementation. Key concepts:
- *Stochastic Interpolants (SI)*: linear interpolation between noise and data, `x_t = (1-t)*noise + t*x1`. Implemented in `src/mfm/SI/`.
- *Meta Flow Map model*: learns a velocity field `v(s, u, x, t_cond, x_cond)` that maps between arbitrary time points rather than just noise→data. Wraps a DiT backbone via `SIModelWrapper`.
- *Training*: PyTorch Lightning + Hydra config (`conf/`). GLASS distillation from a pretrained DMF checkpoint is the recommended training mode.
- *Sampling*: ODE solver (`ode_sampler_fn`), consistency sampler, kernel sampler — all in `src/mfm/SI/samplers.py`.

**`packages/PhysicsInformedDiffusionModels-main/`** — [Physics-Informed Diffusion Models (ICLR 2025)](https://arxiv.org/abs/2403.14404) reference implementation. Key concepts:
- Adds physics residual loss terms (`c_residual * residual_func(x)`) to standard DDIM training so generated samples satisfy PDE/constraint equations.
- Three constraint types: equality residuals (`c_residual`), inequality constraints (`c_ineq`), optimization objectives (`lambda_opt`).
- `src/denoising_toy_utils.py` contains the core loss computation; `src/residuals_darcy.py` and `src/residuals_mechanics_K.py` implement PDE residuals for the two main benchmarks.

**The research goal** is to apply physics-informed constraint losses (PIDM-style) to the flow map framework (MFM-style), so the new package will likely draw the SI/sampler structure from MFM and the physics residual loss terms from PIDM.

## Research framing — three generative camps for FWI

The spine of the field is three camps for generative FWI:

- **Camp A — learn a prior offline, steer it with the wave equation at inference.**
  DPS, RED-DiffEq. Physics enters only at sampling time.
- **Camp B — learn a conditional map seismic→velocity in one pass.** DiffusionVel,
  conditional rectified flow, I2SB. No explicit PDE residual.
- **Camp C — bake PDE residuals into *training* of the generative model.** PIDM, PBFM.

**Where we are:** this project is **solidly in Camp A** — a learned prior (`experiments/0001`
flow-matching prior, `experiments/0002` flow-map prior) steered by the wave equation at inference
(each experiment's `inversion.py` DPS-style tilting; `experiments/0003` diffusion baselines). We
are developing **improvements within Camp A**; physics stays at inference, not training.
**Camp C is not our target — it is not really applicable to this problem.**

The human-curated research narrative lives in `docs/research/*.html` (`prior-work.html`,
`research-plan.html`) — **read these before working on the method, inference, or positioning.**
`prior-work.html` §5.5 is the authoritative account of how Meta Flow Maps work and the steering
estimators we build on (the stochastic one-step posterior; MFM-G — the gradient-based / IWAE
estimator we use — vs. MFM-GF and DPS); `research-plan.html` holds the current thesis (posterior
quality, not cost). **The user curates these directly and requests regeneration; do not silently
rewrite them.** Keep the camp positioning above consistent with them when it shifts.
AI-generated journals/reports (`experiments/JOURNAL.md`, `experiments/*/report.md`) are
**AI-facing** and are NOT kept under `docs/research/`.

## Environment notes

- Python 3.12, managed with `uv`.
- PyTorch is installed from the `pytorch-cu130` index on Linux (CUDA 13.0).
- The MFM package supports three attention backends selectable via `conf/model/sit_xl_2.yaml`: torch SDPA (default), Flash Attention v2 (A100), Flash Attention v3 (H100/H200).
