# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class of
related runs sharing machinery, not a single run. Prefer adding a *variant* to an
existing framework over scaffolding a new number.

## Layout

```
experiments/
├── README.md                 # this contract
├── JOURNAL.md                # one verdict line per concluded finding
├── new.py                    # scaffold the next framework
└── NNNN_slug/
    ├── run.py                # entry point: @hydra.main + Config subclass
    ├── conf/
    │   ├── config.yaml       # base defaults + hydra block
    │   └── experiment/       # config groups (default, smoke, …)
    ├── <helpers>.py          # framework-specific machinery
    └── report.md             # Hypothesis → Setup → Results → Decision
```

## Anatomy of a framework

- **`run.py`** is a `@hydra.main` entry point. It declares a typed `Config`
  subclass (`physics_informed_flow_map.experiment.Config`), composes a
  `DictConfig` from `conf/`, validates it via `Config.from_dictconfig(cfg)`
  (strict — `extra="forbid"`, a typo'd override is an error), then drives the run.
- **`conf/`** holds the Hydra config: `config.yaml` (base defaults + the `hydra`
  block) and `experiment/*.yaml` config groups (variants, each starting with
  `# @package _global_`). Select a variant with `experiment=<name>`.
- The run lifecycle is owned by the harness: `start_run(experiment, run_dir, config)`
  → `run.log(**metrics)` per step → `run.finish(verdict, **summary)`. Tracking goes
  to Weights & Biases; checkpoints are saved locally.

## Where results land

Tracking (config, metrics, sample images, verdict) goes to **Weights & Biases**.
Local artifacts land in the git-ignored `runs/` at the repo root:

```
runs/<framework>/<UTC-stamp>/        # = hydra.run.dir
├── .hydra/config.yaml               # Hydra's composed-config snapshot
├── checkpoints/step_<N>.pt          # local checkpoints (final always saved)
└── samples*.png                     # eval images (also logged to wandb)
```

The verdict is recorded in the wandb run summary and printed to the console.
wandb captures the git commit + a diff patch natively, so runs stay reproducible.

## Running

```bash
WANDB_MODE=online uv run python experiments/NNNN_slug/run.py [experiment=<variant>] [key=value ...]
```

First runs (or any machine without a wandb login) should use `WANDB_MODE=disabled`
for a no-tracking plumbing check, or `WANDB_MODE=offline` to record locally and
`wandb sync` later; `online` requires `wandb login`.

Examples:

```bash
uv run python experiments/0001_flow_matching/run.py experiment=smoke
uv run python experiments/0001_flow_matching/run.py experiment=mnist eval_every=500 ckpt_every=1000
uv run python experiments/0001_flow_matching/run.py experiment=openfwi
```

## Conventions

- **Verdicts are asserted in code**, never by eye — gate on a threshold and pass
  it to `run.finish`.
- Every framework ships a `smoke` variant with trivial budgets for a fast
  end-to-end plumbing check (no strength claim).
- Record findings in `report.md` (cites run directories) and mirror the one-line
  verdict to `JOURNAL.md`. Don't journal in package docs.

## Unconditional drift
ODE vs SDE for the unconditional drift: Flow-matching / SI training is simulation-free: the loss regresses v_θ against the analytic interpolant velocity at random t — no trajectory is ever integrated during training. ODE-vs-SDE is purely a sampling-time choice over the same learned velocity field.
- ODE: integrate dx/dt = v(t,t,x) — the deterministic probability-flow transport (get_unconditional_drift_ode).
- SDE: Euler–Maruyama with drift 2v − x/t and noise scale σ_t² = 2(1/t − 1) (get_unconditional_drift_sde). For the linear interpolant, the score is recoverable from the velocity, so this SDE shares the ODE's marginals — same distribution, stochastic paths.

## Inversion methods (`0004_inversion`): prior × steering

Every inverter in `0004` is one **prior** (what structure we assume) crossed with one
**steering method** (how the wave equation pulls samples toward `d_obs`). The two axes are
orthogonal in principle — MFM's `get_conditional_drift_fn` already takes the estimator as a
parameter — but the grid is **sparse**: not every combination is valid (see notes).

**Priors:** `none` (no learned prior) · `diffusion` (**DDPM**, not DDIM — a `DDPMScheduler`
+ UNet/DiT denoiser) · `flow_matching` (0001) · `flow_map` (0002, MFM's time-conditional
one-step posterior).

**Steering:** `none` (prior sample, the control) · `direct-opt` (gradient descent on the
velocity itself, no prior) · `tweedie` (single-point posterior mean `x̂₁`) · `iwae`
(importance-weighted, backprops the reward) · `sne` (self-normalised, no backprop) · `RED`
(Regularization-by-Denoising).

| prior ↓ / steering → | none | direct-opt | tweedie | iwae | sne | RED |
|---|---|---|---|---|---|---|
| **none**          | —¹ | `classical_fwi`, `realistic_fwi` | — | — | — | — |
| **diffusion (DDPM)** | `unguided` | n/a | `dps` | — | — | `red_diffeq` |
| **flow_matching** | `unguided` | n/a | `flow_tilt` | — | — | — |
| **flow_map (MFM)** | `unguided` | n/a | `flow_tilt` | `mfm_g` | `mfm_gf` | — |

Method roster (config `name:` → what it is):

| method | prior | steering | one-liner |
|---|---|---|---|
| `unguided`      | any learned | none | prior sample, no physics — the control |
| `classical_fwi` | none | direct-opt | naive least-squares FWI: random start + fixed-step Adam, no reg — exposes non-uniqueness |
| `realistic_fwi` | none | direct-opt | properly-run FWI: smooth start + multiscale freq continuation + Tikhonov + L-BFGS |
| `dps`           | diffusion (DDPM) | tweedie | canonical DPS (Chung 2023); Tweedie `x̂₀ = pred_original_sample`, backprops **through the denoiser** |
| `flow_tilt`     | flow / flow_map | tweedie | DPS-style tilt; Tweedie `x̂₁ = xₜ + (1−t)v`, **detached** forward-only gradient |
| `mfm_g`         | flow_map | iwae | MFM gradient-based (Eq. 22): MC posterior draws, backprop the data log-likelihood |
| `mfm_gf`        | flow_map | sne | MFM gradient-free (Eq. 20): reward-weighted posterior average, no backprop |
| `red_diffeq`    | diffusion (DDPM) | RED | diffusion prior as a Regularization-by-Denoising term in a wave-equation-steered optimisation |

**Notes.**
1. `none` prior + `none` steering produces nothing to invert; the closest artefact is the
   `guidance_strength=0` control of a `direct-opt` method, which just returns its starting model.
2. `direct-opt` only applies to `prior=none` (there's no prior to sample); the `n/a` cells are
   not meaningful, not merely unimplemented.
3. `iwae`/`sne` are **flow_map-only**: they need the one-step posterior sampler
   `v(0,1,ε | t_cond=t, x_cond=xₜ)` to draw genuine `x₁ ~ p(x₁|xₜ)`. A plain `flow_matching`
   or `diffusion` prior can't produce those draws.
4. `tweedie` has two implementations by prior: the **detached** cheap approximation for flow
   priors (`flow_tilt`), and the canonical **through-the-denoiser** backprop for diffusion
   (`dps`). Same concept (`drift_estimator="dps"` in MFM's steering *is* the Tweedie path),
   different fidelity.

## New framework

```bash
uv run python experiments/new.py "short title of the idea"
```

## Model guidance 
In mfm, model guidance using classifier-free guidance (CFG) is turned on. This happens during training, not inference time, therefore it has nothing to do with reward/physics steering. In our experiment, CFG is turned off as openFWI doesn't have the concept of 'classes' like that in ImageNet, so it's irrelevant. 

## off-diagonal distillation
Here I explain how this training is done.
Line 482 in mfm/losses/loss.py is essentially doing the following math. Take the middle equation in equation (11) in the mfm paper and substitute the residual form X_{s,u}(x) = x + (u-s)·v_{s,u}(x). 
  Now look at the code:

  tangents = (1, 0, vss)          # differentiate w.r.t. s (=1), not u (=0), along direction vss in x
  vsu, jvp = torch.func.jvp(vsu_fn, (s, u, Is), tangents)

  The JVP computes:

  jvp = ∂v/∂s · 1  +  ∂v/∂u · 0  +  ∇_x v · vss
      = ∂_s v_{s,u}  +  v_{s,s} · ∇_x v_{s,u}

  So the target becomes:

  distillation_teacher = vss + (u-s) * jvp
                       = v_{s,s} + (u-s) · [∂_s v_{s,u} + v_{s,s} · ∇_x v_{s,u}]

Note that vss is that from the teacher (through extract_posterior_velocity) and the jvp is done through the student. 

## distillation 
Both diagonal (distill_fm_loss) and off-diagonal loss (distillation_loss) uses the teacher's velocity, that is extracted from the extract_posterior_velocity() function. This function IS GLASS reparameterisation. It first computes t_start, the reparameterised time (line 121-125). It then computes x_star, which is the linear sufficient statistic S in equation (25) in the mfm paper and it is the point at which we need to query the teacher. It then finds v_star by querying the teacher 'teacher_model.v(t_star, t_star,x_star, 0, 0s). This is the only place the teacher is called. 
In the next few lines, term 2 is the second term in equation (25), and term 1 is the other two term combined (the xt_cond term is the third term, and Is term is the first term in the equation). Overall, the function returns the conditional teacher velocity even though the teacher we have is unconditional. This conditional teacher velocity then becomes the target velocity used in both diagonal and off diagonal distillation.  


## conditioning -- the 'meta-ness' 
When t_cond is zero, the values of xt_cond is completely irrelevant: it can be set to noise, zeros, or anything, and nothing will be affected. This is designed in the DiT model. Firstly, x_cond_embedder and t_cond_embedder both have initial weights and biases that are zero. So a from-scratch model never learns to use x_cond. To learn x_cond, mfm copies x_embedder into x_cond_embedder inside 'init_from_dmf'. In my experiments, those activation codes are handled in the function activate_x_cond_conditioning(). The preserve_t_cond_0 is off in mfm's code, probably so that the network can learn its own gating schedule rather than hardcoding xt_cond to be zero when t_cond = 0. 