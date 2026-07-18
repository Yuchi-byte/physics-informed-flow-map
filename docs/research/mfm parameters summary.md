# MFM parameters summary

Configuration parameters used by *Meta Flow Maps enable scalable reward alignment*
(Potaptchik, Saravanan, Mammadov, Prat, Albergo, Teh — ICML 2026), plus the config
values shipped in `packages/mfm-meta-flow-map-main/conf/`.

Sources:
- **Paper**: [arXiv:2601.14430v2](https://arxiv.org/abs/2601.14430) (46 pages incl. appendices);
  local copy at `docs/research/2601.14430v2.pdf`. Page numbers below are PDF/printed pages.
- **Code**: `packages/mfm-meta-flow-map-main/conf/*.yaml` (release configs).

Where the paper and the repo configs disagree, both values are listed.

---

## 1. Experiments performed

| § | Experiment | Task | Page |
|---|---|---|---|
| 7.1.1 / F.1 | **2D Gaussian Mixture Model** | Steering a GMM prior to a linear-inverse-problem posterior | p. 15–16, 37 |
| 7.1.2 / F.2 | **MNIST** | Steering to a multimodal class-mixture posterior | p. 16, 38 |
| 7.2.1 / F.3, F.3.1 | **ImageNet 256×256 — base MFM quality** | Few-step FID + posterior-sampler fidelity; scale/objective ablations | p. 17–18, 39 |
| 7.2.2 / F.3.2 | **ImageNet 256×256 — inference-time steering** | MFM-G / MFM-GF / MFM-Search vs DPS, Best-of-N | p. 18–19, 39–42 |
| 7.2.3 / F.3.4 | **ImageNet 256×256 — reward fine-tuning** | MFM-FT objective (Eq. 43) on HPSv2 | p. 19–20, 44 |

---

## 2. Shared method-level parameters

### 2.1 Interpolant and diffusion schedule

| Parameter | Value | Reference |
|---|---|---|
| Interpolant | Linear, `I_t = α_t I_0 + β_t I_1`, `α_0 = β_1 = 1`, `α_1 = β_0 = 0` | Eq. (4), p. 4 |
| Base distribution `p_0` | Gaussian `N(0, I)` | Prop. C.1, Eq. (74), p. 30 |
| Diffusion coefficient | `σ_t² / 2 = (β̇_t/β_t) α_t² − α̇_t α_t` | Eq. (14), p. 5; Eq. (75), p. 30 |
| Code: interpolant class | `mfm.SI.interpolants.Linear`, `t_max: 1.0` | `conf/SI/linear.yaml` |

### 2.2 Loss / objective options

| Parameter | Value | Reference |
|---|---|---|
| Total objective | `L_MFM = L_diag + L_cons` | Eq. (33), p. 11 |
| Consistency objectives available | Eulerian (self / teacher), Lagrangian (self / teacher), Mean Flow (self), Semigroup (self) | Table 1, p. 10 |
| Adaptive loss weight | `w = 1 / (‖Δ‖₂² + c)^p` | Eq. (73), p. 29 |
| Adaptive `c` | `0.01` (fixed across all ablations) | Table 4 caption, p. 39 |
| Adaptive `p` (diag, cons) | `(1.0, 1.0)` default; `(0.5, 1.0)` best for XL/2 2- and 4-step | §F.3.1, p. 39 |
| Code: `fm_adaptive_loss_p` / `_c` | `0.5` / `0.01` | `conf/config_train.yaml` |
| Code: `distill_adaptive_loss_p` / `_c` | `1.0` / `0.01` | `conf/config_train.yaml` |
| Code: `distillation_type` | `esd_teacher` | `conf/config_train.yaml` |
| Code: `fm_loss_type`, `distill_fm_loss_type`, `distillation_loss_type` | `adaptive` | `conf/config_train.yaml` |
| Code: `distill_teacher_stop_grad` | `true` | `conf/config_train.yaml` |
| Code: `data_fm` / `distill_fm` | `false` / `true` | `conf/config_train.yaml` |

### 2.3 Model guidance (MG)

| Parameter | Value | Reference |
|---|---|---|
| MG target | `v_tgt = İ_t + ω · sg(v_θ(·,y) − v_θ(·,∅))` | Eq. (69)–(70), p. 28 |
| Equivalent CFG scale | `ω_CFG = 1/(1 − ω)` | p. 28 |
| MG scale `ω` (MFM training) | `0.6` | Table 5, p. 46 |
| Guidance interval (MFM training) | `[0.0, 1.0]` | Table 5, p. 46 |
| Guidance interval (DMF flow-map training) | `[0.0, 0.7]` | Table 5, p. 46 |
| Code: `model_guidance` | `true`, `model_guidance_base_prob: 0.0` | `conf/config_train.yaml` |
| Code: `model_guidance_class_ws` | `[2.5]` | `conf/model/sit_xl_2.yaml` |
| Code: `model_guidance_x_cond_ws` | `[]` | `conf/model/sit_xl_2.yaml` |

### 2.4 Architecture conditioning

| Parameter | Value | Reference |
|---|---|---|
| Global conditioning vector | `c = Embed_s(s) + Embed_u(u) + Embed_class(y) + Embed_t(t)` | Eq. (71), p. 29 |
| Input construction | `x_input = PatchEmbed(x_s) + PosEmbed + AdaLN-Zero[PatchEmbed'(x) | t]` | Eq. (72), p. 29 |
| Param overhead (B/2) | 131M → 134M | §F.3, p. 39 |
| Param overhead (XL/2) | 675M → 684M | §F.3, p. 39 |

---

## 3. Experiment 1 — 2D Gaussian Mixture Model (§7.1.1, §F.1, p. 15–16, 37)

### Model / data

| Parameter | Value | Reference |
|---|---|---|
| Prior `p_1` | 3-component 2D GMM, equal weights 1/3 | §F.1, p. 37 |
| Means | `µ₁ = (−3,−3)`, `µ₂ = (0,0)`, `µ₃ = (3,3)` | §F.1, p. 37 |
| Covariances | `Σ₁ = Σ₂ = Σ₃ = 0.5 · I₂ₓ₂` | §F.1, p. 37 |
| Network | small MLP | §F.1, p. 37 |
| Training objective | Semigroup MFM loss (Table 1) | §F.1, p. 37 |

### Reward / target

| Parameter | Value | Reference |
|---|---|---|
| Likelihood | `y = aᵀx + ε`, `ε ~ N(0, σ²)` | §F.1, p. 37 |
| Measurement vector `a` | `[1.2, −0.8]ᵀ` | §F.1, p. 37 |
| Noise `σ` | `0.2` | §F.1, p. 37 |
| Observation `y_obs` | `−1.0` | §F.1, p. 37 |

### Sampling / evaluation

| Parameter | Value | Reference |
|---|---|---|
| ODE solver | Euler, `N = 1000` steps | §F.1, p. 37 |
| SDE solver | Euler–Maruyama, `N = 1000` steps | §F.1, p. 37 |
| MC samples in drift estimator | `N ∈ {1, 2, 4, 8, 16, 32, 64, 128}` | Fig. 4 p. 16; Fig. 11 p. 38 |
| SMC (TDS) particles | `K = 4096` | §F.1, p. 37; Fig. 4 caption p. 16 |
| Seeds for SMC mean | 20 | §F.1, p. 37 |
| Evaluation sample count | 4096 posterior samples | §F.1, p. 37 |
| Metrics | Sliced Wasserstein-2 (S-W₂), MMD (multi-scale RBF kernel, unbiased estimator) | §F.1, p. 37 |
| Baselines | DPS, SMC/TDS | §7.1.1, p. 15 |

---

## 4. Experiment 2 — MNIST (§7.1.2, §F.2, p. 16, 38)

| Parameter | Value | Reference |
|---|---|---|
| Network | UNet-based MFM, **9M parameters** | §F.2, p. 38 |
| Training objective | Semigroup MFM loss (Table 1) | §F.2, p. 38 |
| Reward | `exp(r(x)) = p(c_mix|x) = Σᵢ wᵢ p_θ(yᵢ|x)` | §7.1.2, p. 16; §F.2, p. 38 |
| Reward model | simple CNN classifier | §F.2, p. 38 |
| Mixture weights `w` | `[0, 1, 0, 1, 0, 2, 0, 2, 0, 4]` (normalised) | §F.2, p. 38 |
| ODE / SDE solver steps | Euler / Euler–Maruyama, `N = 500` | §F.2, p. 38 |
| MC samples in drift estimator | `N ∈ {1, 2, 4, 8, 16, 32}` | Fig. 5, p. 16 |
| SMC particles | `K = 64` | §F.2, p. 38; Fig. 5 p. 16 |
| Seeds for SMC mean | 20 | §F.2, p. 38 |
| Evaluation sample count | 4096 steered samples | §F.2, p. 38 |
| Metric | L₂ between empirical class PMF and `w` | §F.2, p. 38 |
| Baselines | DPS, MFM-GF | §7.1.2, p. 16 |

---

## 5. Experiment 3 — ImageNet 256×256 base model (§7.2.1, §F.3, p. 17–18, 39)

### 5.1 Full training configuration (Table 5, p. 46)

| Parameter | MFM-B/2 | MFM-XL/2 |
|---|---|---|
| **Model** | | |
| Resolution | 256×256 | 256×256 |
| Params (M) | 134 | 683 |
| Hidden dim. | 768 | 1152 |
| Heads | 12 | 16 |
| Patch size | 2×2 | 2×2 |
| Sequence length | 256 | 256 |
| Layers | 12 | 28 |
| Encoder depth | 8 | 20 |
| **Optimisation (base flow model)** | | |
| Optimiser | AdamW | AdamW |
| Batch size | 256 | 256 |
| Learning rate | 1e-4 | 1e-4 |
| Adam `(β₁, β₂)` | (0.9, 0.95) | (0.9, 0.95) |
| Adam `ε` | 1e-8 | 1e-8 |
| Adam weight decay | 0.0 | 0.0 |
| EMA decay | 0.9999 | 0.9999 |
| **Flow model training** | | |
| Training iterations | 800K | 4M |
| Epochs | 160 | 800 |
| Class dropout probability | 0.2 | 0.2 |
| Time proposal `µ_FM` | 0.0 | — |
| REPA alignment depth | — | 8 |
| REPA vision encoder | — | DINOv2-B/14 |
| QK-norm | ✗ | ✗ |
| **DMF flow map training** | | |
| Training iterations | — | 400K |
| Epochs | — | 80 |
| Class dropout probability | — | 0.1 |
| Time proposal `µ_FM` | — | 0.0 |
| Time proposal `(µ_MF⁽¹⁾, µ_MF⁽²⁾)` | — | (0.4, −1.2) |
| Model guidance scale `ω` | — | 0.6 |
| Guidance interval | — | [0.0, 0.7] |
| **MFM training** | | |
| Training iterations | 100K | 100K |
| Batch size | 512 | 360 |
| Epochs | 40 | 28 |
| Optimiser | RAdam | RAdam |
| Learning rate | 1e-4 | 1e-4 |
| LR warmup | Linear, first 2000 steps | Linear, first 2000 steps |
| RAdam `(β₁, β₂)` | (0.9, 0.999) | (0.9, 0.999) |
| RAdam `ε` | 1e-8 | 1e-8 |
| RAdam weight decay | 0.0 | 0.0 |
| EMA decay | 0.9999 | 0.9999 |
| Class dropout probability | 0.2 | 0.2 |
| Model guidance scale `ω` | 0.6 | 0.6 |
| Guidance interval | [0.0, 1.0] | [0.0, 1.0] |

Backbone: latent DiT operating in a pretrained VAE latent space (§7.2.1, p. 17).
B/2 is initialised from SiT B/2; XL/2 is initialised from DMF XL/2+ (§F.3, p. 39).

### 5.2 Training objective per variant (§F.3, p. 39)

| Variant | Objective |
|---|---|
| Trained from data | Mean-Flow MFM objective (Table 1) |
| Distillation | Eulerian (Teacher) objective (Table 1), regressing onto GLASS-extracted conditional drift (Eq. 32) |

### 5.3 Ablation grid (Table 4, p. 39)

Configurations evaluated at NFE ∈ {1, 2, 4, 8}, adaptive-loss `c = 0.01` fixed,
`p` given as (diag, cons):

| Model | Config |
|---|---|
| B/2 | Data (1.0, 1.0) |
| XL/2 | Data (1.0, 1.0) |
| XL/2 | Distill (1.0, 1.0) |
| XL/2 | **Distill (0.5, 1.0)** — best, reported as MFM-XL/2 in Table 2 |

### 5.4 Evaluation settings

| Parameter | Value | Reference |
|---|---|---|
| FID images | 50,000 generated + 50,000 reference | Table 2 caption, p. 17 |
| Sampler | K-step refinement sampler (Algorithm 1) | §7.2.1, p. 17 |
| NFE reported | 1, 2, 4 | Table 2, p. 17 |
| CFG notation | `2×` denotes CFG usage in baselines | Table 2 caption, p. 17 |
| Posterior-recovery conditioning set | `N = 50,000` noised images | §7.2.1(A), p. 17 |
| Posterior-recovery conditioning times `t` | 0.0, 0.1, 0.2, 0.5 | Fig. 6, p. 18 |
| Value-function ground truth | SDE rollout, **200 steps**, `N = 200` particles | §7.2.1(B), p. 17 |
| Value-function cheap estimator | `N = 200` particles via MFM or GLASS | §7.2.1(B), p. 17 |
| Value-function conditioning times `t` | 0.1, 0.2, 0.3, 0.5 | Fig. 6, p. 18 |
| Value-function reward | ImageReward, prompt "A high-quality, high-resolution photograph of a tabby cat" | §7.2.1(B), p. 17 |
| Value-function metric | Pearson correlation `r` | Fig. 6 caption, p. 18 |
| NFE swept | 1–16 | Fig. 6, p. 18 |

---

## 6. Experiment 4 — ImageNet inference-time steering (§7.2.2, §F.3.2, p. 18–19, 39–42)

### 6.1 Sampling

| Parameter | Value | Reference |
|---|---|---|
| Dynamics | Probability-flow ODE (Eq. 39), Euler scheme | §F.3.2, p. 39 |
| Discretisation steps `K` | **250** | §F.3.2, p. 39 |
| Steered class | tabby cat (class 281 in code) | §7.2.2, p. 18; `conf/config_sample_steering.yaml` |
| Images generated per configuration | 128 | §7.2.2, p. 18 |
| Metric | average reward over the 128 images | §7.2.2, p. 18 |

### 6.2 Rewards

| Parameter | Value | Reference |
|---|---|---|
| Reward models | ImageReward, PickScore, HPSv2 | §7.2.2, p. 18 |
| Prompt | "A high-quality, high-resolution photograph of a tabby cat." | §7.2.2, p. 18 |
| Reward multiplier `λ` | `{1.0, 2.5, 5.0}`, with `p_reward ∝ p_model exp(λ r_θ)` | §7.2.2, p. 18 |

### 6.3 Methods and their parameters

| Method | Parameters | Reference |
|---|---|---|
| MFM-G (Eq. 22/37) | MC samples `N ∈ {1, 2, 4, 8, 16, 32}` | Figs. 18–20, p. 42 |
| MFM-GF (Eq. 20) | MC samples `N ∈ {1, 2, 4, 8, 16, 32}` | Figs. 18–20, p. 42 |
| MFM-Search (Algorithm 4) | candidates `M ∈ {1, 2, 4, 8, 16, 32, 64}`; `N` posterior samples per candidate | Figs. 7, 14–17, p. 19, 40–41 |
| DPS | uses the same extracted unconditional drift | §F, p. 37 |
| Best-of-N | `N_BoN ∈ [1, 1000]`; pool of **128,000** samples, partitioned into **128** disjoint groups of size `N_BoN` | §F.3.2, p. 39 |

All baselines (DPS, SMC-TDS, Best-of-N) are implemented on the drift extracted
from the MFM itself (§F, p. 37).

### 6.4 Gradient renormalisation (§F.3.6, p. 45)

| Parameter | Value | Reference |
|---|---|---|
| Rescaling rule | `b*_t(x) = b_t(x) + λ ‖b_t(x)‖₂ · ∇V(x)/‖∇V(x)‖₂` | Eq. (136), p. 45 |
| `λ` used | **1** | §F.3.6, p. 45 |
| Alternatives considered | clipping vs rescaling; rescaling used in **all** ImageNet steering experiments | §F.3.6, p. 45 |

### 6.5 NFE accounting (§F.3.5, p. 45)

| Method | NFE formula |
|---|---|
| MFM-GF | `K + 2NK` |
| MFM-G | `K + 4NK` (2× multiplier assumed for the backward pass) |
| MFM-Search | `2MNK` |
| DPS | `4NK` |
| Best-of-N | `K·N_BoN + N_BoN` |

---

## 7. Experiment 5 — ImageNet reward fine-tuning (§7.2.3, §F.3.4, p. 19–20, 44)

| Parameter | Value | Reference |
|---|---|---|
| Objective | MFM-FT (Eq. 43), off-policy | §5.2, p. 13 |
| Reward model | HPSv2 | §7.2.3, p. 19 |
| Prompt template | "A high-quality, high-resolution photograph of a {class}." | §7.2.3, p. 19 |
| Classes | all ImageNet classes | §7.2.3, p. 19 |
| Reward multiplier `λ` | `{10, 25, 50}` | §7.2.3, p. 19; §F.3.4, p. 44 |
| Training iterations | ~10,000 | Fig. 9, p. 19 |
| Evaluation cadence | every 500 training iterations | Fig. 9 caption, p. 19 |
| Evaluation samples | 512 ODE samples | Fig. 9 caption, p. 19 |
| Evaluation reward models | HPSv2, ImageReward, PickScore | Fig. 9, p. 19 |
| Qualitative sampler | class-conditioned ODE, **250 steps** | Fig. 10 caption, p. 20 |

---

## 8. Repository configs (`packages/mfm-meta-flow-map-main/conf/`)

All configs share the Hydra defaults `model: sit_xl_2`, `dataset: imagenet_1k`,
`SI: linear`, and set `use_parametrization: false`, `seed: 42`.

### 8.1 `model/sit_xl_2.yaml`

```
_target_: mfm.models.DiTMFM      learn_loss_weighting: false
input_size: 32                   patch_size: 2
in_channels: 4                   encoder_depth: 20
hidden_size: 1152                depth: 28                num_heads: 16
label_dim: 1000
use_joint_attention: false       preserve_t_cond_0: false
model_guidance_class_ws: [2.5]   model_guidance_x_cond_ws: []
attn_func: "base"                # base | fa2 | fa3
is_zero_data: true               init: dmf
```

### 8.2 `dataset/imagenet_1k.yaml`

```
name: "imagenet"   img_resolution: 256   img_channels: 3
```

### 8.3 `SI/linear.yaml`

```
_target_: mfm.SI.interpolants.Linear   t_max: 1.0
```

### 8.4 `config_train.yaml` — MFM training

| Key | Value |
|---|---|
| `init_from_dmf` / `dmf_path` | `true` / `ckpts/dmf_xl_2_256.pt` |
| `optimizer` | `RAdam` |
| `lr.val` / `lr.warmup_steps` / `lr.min_lr` | `1e-4` / `2000` / `1e-4` |
| `trainer.num_train_steps` | `100000` |
| `trainer.batch_size` | `32` (paper Table 5 reports the global batch: 360 for XL/2, 512 for B/2) |
| `trainer.ema.decay` | `0.9999` |
| `trainer.precision` | `bf16-mixed` |
| `trainer.gradient_clip_val` | `1.5` |
| `trainer.accumulate_grad_batches` | `1` |
| `trainer.class_dropout_prob` | `0.2` |
| `trainer.checkpoint_every_n_steps` | `25000` |
| `trainer.log_every_n_steps` | `25` |
| `trainer.num_warmup_steps`, `anneal_end_step`, `t_cond_warmup_steps`, `t_cond_anneal_end_step` | `0` |
| `trainer.t_cond_0_rate` / `t_cond_power` | `0.1` / `2.0` |
| `loss.*` | see §2.2 |
| `weighting_model` | `LossWeightingNetwork`, `channels: 128`, `clamp_min: −10.0`, `clamp_max: 10.0` |
| `sampling.every_n_steps` | `10000` |
| `sampling.n_unconditional_samples` | `128` |
| `sampling.n_kernel_steps` | `[1, 2, 4]` |
| `sampling.kernel_cfg_scales` / `cfg_scales` | `[2.5]` / `[2.5]` |
| `sampling.n_conditioning_samples` / `n_samples_per_image` | `32` / `4` |
| `sampling.consistency_sampler.steps_to_test` | `[1, 4]` |
| `sampling.consistency_sampler.t_conds` | `[0.00, 0.20, 0.40]` |
| `sampling.max_batch_size` / `vae_batch_size` | `32` / `32` |

### 8.5 `config_sample.yaml` — unconditional FID sampling

| Key | Value |
|---|---|
| `sampler` | `kernel` (alt: `t0`) |
| `kernel_sampler_steps` | `1` (options 1, 2, 4) |
| `cfg_scale` | `2.5` |
| `num_samples` | `50000` |
| `per_proc_batch_size` | `128` |
| `save_png` | `false` |

### 8.6 `config_sample_steering.yaml` — inference-time steering

| Key | Value |
|---|---|
| `sampler` | `ode` |
| `drift_estimator` | `iwae` (MFM-G); alts `dps`, `sne` (MFM-GF) |
| `mc_samples` | `4` |
| `n_steps` | `250` |
| `guidance_scale` | `1.0` |
| `renorm_gradient` / `renorm_scale` | `true` / `1.0` |
| `image_reward.model_name` | `HPSv2` |
| `image_reward._lambda` | `2.0` |
| `image_reward.prompt` | "A high-resolution, high-quality photograph of a tabby cat." |
| `cfg_scale` | `2.5` |
| `class_label` | `281` |
| `num_samples` / `per_proc_batch_size` | `8` / `32` |

### 8.7 `config_steering_search.yaml` — MFM-Search

| Key | Value |
|---|---|
| `n_steps` | `250` |
| `step_power` | `2.5` |
| `particles` (M) | `8` |
| `mc_samples_per_particle` (N) | `4` |
| `resampling_type` | `max_overall` (alt: `max_per_particle`) |
| `transition_type` | `stochastic` |
| `image_reward._lambda` | `1.0` (no effect on greedy search) |
| `image_reward.model_name` / `prompt` | `HPSv2` / tabby-cat prompt |
| `cfg_scale` / `class_label` | `2.5` / `281` |
| `num_samples` / `per_proc_batch_size` | `128` / `64` |

### 8.8 `config_finetune.yaml` — reward fine-tuning

| Key | Value |
|---|---|
| `optimizer` | `RAdam` |
| `lr.val` / `warmup_steps` / `min_lr` | `1e-4` / `250` / `5e-5` |
| `trainer.num_train_steps` | `10000` |
| `trainer.batch_size` | `18` |
| `trainer.ema` | `enabled: true`, `decay: 0.999`, `min_step: 100` |
| `trainer.precision` | `bf16-mixed` |
| `trainer.gradient_clip_val` | `1.5` |
| `trainer.checkpoint_every_n_steps` | `5000` |
| `trainer.log_every_n_steps` | `1` |
| `imagereward_function` | `HPSv2` (alts: `ImageReward`, `PickScore`) |
| `loss.prompt_prefix` | "A high-resolution, high-quality photograph of a " |
| `loss._lambda` | `50.0` |
| `loss.sigma_t_sq` / `sigma_t_sq_type` | `200` / `constant` (alt `schedule`) |
| `filter_for_classes` / `filter_class_indices` | `false` / `[]` |
| `sampling.every_n_steps` / `num_samples` | `500` / `512` |

### 8.9 `config_sample_posterior.yaml` — posterior sampling

| Key | Value |
|---|---|
| `posterior_t` | `0.4` |
| `posterior_sampler_steps` | `1` |
| `posterior_n_samples` | `16` (per image) |
| `cfg_scale` | `2.5` |
| `num_samples` / `per_proc_batch_size` | `16` / `32` |

### 8.10 `config_sample_value.yaml` — value-function estimation

| Key | Value |
|---|---|
| `init_from_dmf` / `save_ground_truth` | `true` / `true` |
| `ddpm.n_steps` / `ddpm.n_samples` | `200` / `200` (high-fidelity SDE reference) |
| `n_samples_mc` | `200` |
| `n_steps_mc` | `1` |
| `n_method_mc` | `consistency` (alts: `glass`, `flows`) |
| `posterior_t` | `0.1` |
| `cfg_scale` / `cls` | `2.5` / `281` |
| `reward_model` / `reward_prompt` | `HPSv2` / tabby-cat prompt |
| `num_samples` / `per_proc_batch_size` | `128` / `32` |
| `trainer.batch_size` | `64` |

### 8.11 Environment (README)

| Item | Value |
|---|---|
| Python | 3.12 |
| Attention backends | Torch SDPA (default) / FlashAttention v2 (A100/Ampere) / FlashAttention v3 (Hopper, CUDA 12.9) — set via `attn_func` |
| Checkpoints | `ckpts/mfm-xl2.pt` (HF `adh1s/mfm`); `ckpts/dmf_xl_2_256.pt` (HF `kyungmnlee/DMF`) |

---

## 9. Notes on paper/code discrepancies

- `config_train.yaml` sets `trainer.batch_size: 32` with `accumulate_grad_batches: 1`; Table 5
  (p. 46) reports global batch sizes of 360 (XL/2) and 512 (B/2), i.e. the released config is
  a per-device setting.
- `config_sample_steering.yaml` sets `image_reward._lambda: 2.0`, which is not one of the
  three λ values swept in the paper (`{1.0, 2.5, 5.0}`, §7.2.2 p. 18).
- The steering prompt differs slightly: paper uses "A **high-quality, high-resolution**
  photograph of a tabby cat." (§7.2.2, p. 18); configs use "A **high-resolution, high-quality**
  photograph of a tabby cat."
- `config_finetune.yaml` uses `loss._lambda: 50.0`, the largest of the paper's `{10, 25, 50}`.
- `loss.sigma_t_sq: 200` (finetune config) has no named counterpart in the paper's Eq. (43).
