# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class of
related runs sharing machinery, not a single run. Prefer adding a *variant* to an
existing framework over scaffolding a new number.


## Running the scripts 

To invert curvefault_b_17: 

```
uv run python experiments/0004_inversion/run.py \
  prior=flow_map method=flow_tilt \
  method.misfit=ot \
  target=curvefault_b_17 \
  ckpt=/workspace/runs/0002_flow_map/openfwi_mf_2026-07-08T10-23-48Z/checkpoints/step_89_ema.pt steps=5 \
  model.hidden=768 model.depth=12 model.num_heads=12 model.patch_size=4
```
To run inversion on Marmousi using flow matching prior: 
```
uv run python experiments/0004_inversion/run.py \
  prior=flow_matching method=flow_tilt method.misfit=l2 \
  target=marmousi_fault05 steps=200 n_samples=10\
  ckpt=/workspace/runs/0001_flow_matching/openfwi_2026-07-07T11-19-11Z/checkpoints/step_89_ema.pt \
  model.hidden=768 model.depth=12 model.num_heads=12
``` 
To invert Marmousi using diffusion prior: 
```
uv run python experiments/0004_inversion/run.py prior=diffusion method=dps method.misfit=ot prior.denoiser_kind=dit target=marmousi_fault05 steps=200 n_samples=10 ckpt=/workspace/runs/0003_diffusion/openfwi_2026-07-07T23-26-16Z/checkpoints/step_59_ema.pt model.hidden=768 model.depth=12 model.num_heads=12
```
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
## Methodology notes -- training
### Training steps 
Training mimics that done in mfm. mfm starts directly with a teacher FLOW MAP model which is already state-of-the-art. 

There are three losses that are sumed to give the training loss: 
* fm_loss or distill_fm_loss. This trains the diagonal loss (i.e. when s=u). You only choose one out of those two losses because you decide whether you train from data or from a teacher. mfm chooses to train from a SOTA teacher flow map model. We choose to train from data, because our teacher (the output from experiment 0001) isn't necessary better than traning from data.
* distillation_loss. This trains the off diagonal loss (i.e. when s!=u). The lsd, esd_teacher, etc 'distillation_type' is only relevant here. This training has to be done based on a teacher model, which in our case is the flow matching model from experiment 0001. 

Note that the 'meta-ness' of mfm, which is the conditioning component, is embedded in the mfm training itself: 
* For diagonal loss, check that fm_loss is trained conditionally in 0002????????
* For off-diagonal loss, the conditional velocity comes from the unconditional teacher using a clever trick called GLASS distillation. It queries the unconditional teacher at a special point that returns the conditional velocity the student is trying to learn from. 

The training has three phases: firstly, the warmup phase that only trains the diagonal loss. Secondly the anealling phase that now gradually turns on the off-diagonal training: the jump size increases linearly with training steps (training steps are similar to epochs). Thirdly, the off diagonal is now fully turned on. Note that the diagonal loss is always turned on.  

0001 flow matching trains through the mfm framework end-to-end (same DiTMFM model, same get_consistency_loss_fn loss), with the flow-map-specific parts (off-diagonal jumps, conditioning) disabled through parameter settings rather than different code. Therefore it makes sense that 0001 and 0002 have the same amount of model parameters. 

uncond_prob (renamed from mfm's t_cond_rate_0) = 0.1 is the rate at which t_cond is 0 (when 0, the mfm is unconditional). So 10% of the time, t_cond will be set to 0, whilst 90% of the time t_cond will take any value between 0 and 1. For each training sample, a Bernoulli draw with probability 1 - uncond_prob decides whether the conditioning time t_cond is nonzero. This is the mechanism that turns the flow map into a posterior model rather than just a prior. Keeping it 0.1 rather than 0 because the model must stay good at t_cond = 0. Reserving a fraction of batches for the unconditional case keeps the prior mode from degrading while most of the capacity goes to the harder, richer conditional task. It's the same idea as classifier-free-guidance label dropout (mfm uses class_dropout_prob = 0.1 for class labels analogously), just applied to the state-conditioning pathway.


### Validation loss 
Validation loss is exactly the same as training loss, but applied to the validation dataset, which is unseen during training. compute_val_loss() takes a target velocity map and finds the interpolant associate with the x and t. Then x and t are passed to the neural network, and the model's predicted velocity is compared with the target velocity (that from the interpolant) to compute the loss. The validation loss has nothing to do with sampling (i.e. image generation through some ODE/SDE solver). The sampling is done when on_eval is true merely for visualisation purposes. 
Validation loss is only calculated on diagonal velocities, because off-diagonal velcoities have no data-defined ground truth. The teacher velocity will be the closest to the ground truth, from which the student learns. 
The off-diagonal velocities are tested through the few-step sample. sample_few_step() uses the trained s-u velocities to generate the samples when a particular epoch is on_eval. Those visualisations are the images generated using the weights at that particular training epoch. 
For 0002, the validation loss is observed to increase before decreasing. The validation loss is only tracking the diagonal loss. And the initial increase is associated with the flow-map anealing phase. Initially, the anealing is almost turned off and the model behaves like a very good flow matching model. As off-diagonal loss are introduced to the training, the model weights now deviates from the flow matching to train the off-diagonal, resulting in an increase in validation loss. 

### Model guidance 
In mfm, model guidance using classifier-free guidance (CFG) is turned on. This happens during training, not inference time, therefore it has nothing to do with reward/physics steering. In our experiment, CFG is turned off as openFWI doesn't have the concept of 'classes' like that in ImageNet, so it's irrelevant. 

### off-diagonal distillation
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

### distillation 
Both diagonal (distill_fm_loss) and off-diagonal loss (distillation_loss) uses the teacher's velocity, that is extracted from the extract_posterior_velocity() function. This function IS GLASS reparameterisation. It first computes t_start, the reparameterised time (line 121-125). It then computes x_star, which is the linear sufficient statistic S in equation (25) in the mfm paper and it is the point at which we need to query the teacher. It then finds v_star by querying the teacher 'teacher_model.v(t_star, t_star,x_star, 0, 0s). This is the only place the teacher is called. 
In the next few lines, term 2 is the second term in equation (25), and term 1 is the other two term combined (the xt_cond term is the third term, and Is term is the first term in the equation). Overall, the function returns the conditional teacher velocity even though the teacher we have is unconditional. This conditional teacher velocity then becomes the target velocity used in both diagonal and off diagonal distillation.  


### conditioning -- the 'meta-ness' 
When t_cond is zero, the values of xt_cond is completely irrelevant: it can be set to noise, zeros, or anything, and nothing will be affected. This is designed in the DiT model. Firstly, x_cond_embedder and t_cond_embedder both have initial weights and biases that are zero. So a from-scratch model never learns to use x_cond. To learn x_cond, mfm copies x_embedder into x_cond_embedder inside 'init_from_dmf'. In my experiments, those activation codes are handled in the function activate_x_cond_conditioning(). The preserve_t_cond_0 is off in mfm's code, probably so that the network can learn its own gating schedule rather than hardcoding xt_cond to be zero when t_cond = 0. 

## Methodology notes on Inversion 
At every inversion, no matter the method, 4 random samples were drawn and their trajectories calculated. That's why eventually we have 4 different samples in the trajectory visualisation. All 4 final velocity images ARE the inversions of the same d_obs. Having four rather than one means the sampling uncertainty can also be determined. Arbitrarily, we took one of the velocity maps as the 'final' inversion -- in practice, we could have picked the min-misfit sample as the final inversion. Note that the metrics (mae_mean, etc) are averaged across the 4 samples. Those 4 samples have nothing to do with the mc_samples in mfm: for mfm, each sample will have their own mc_samples. 