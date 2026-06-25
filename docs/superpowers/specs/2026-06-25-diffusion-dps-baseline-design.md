# Diffusion-prior + DPS baseline for 0002 — Design

**Date:** 2026-06-25
**Status:** Approved for planning

## Goal

Add a **diffusion prior + DPS** posterior-sampling baseline for the 0002 FWI experiment, to
compare against our flow-tilting PoC. This is the apples-to-apples **camp A** comparison: the
*same* FlatVel-A velocity data, the *same* Deepwave forward operator and 64→70 bridge, and the
*same* DPS-style likelihood guidance — only the generative prior changes (diffusion vs flow).

This spec covers the **diffusion prior + DPS sampler only**. The classical-FWI control and the
head-to-head quantitative comparison (MAE/SSIM/#solves, side-by-side figure) are a deliberate
follow-up spec.

## Motivation

Our flow-tilting PoC works (data misfit → 0.5%, MAE ~300 m/s). To claim the flow(-map) prior is
competitive/cheaper, we need the diffusion baseline the literature uses (RED-DiffEq, DPS-FWI):
an unconditional diffusion prior over velocity maps + DPS guidance with the wave-equation
likelihood. We **import the diffusion machinery** (`diffusers`) rather than hand-rolling it, and
write only the thin DPS guidance (≈25 lines), following the canonical
[DPS](https://github.com/DPS2022/diffusion-posterior-sampling) algorithm and reusing the
**gradient-normalization** lesson from the flow PoC. The vendored `packages/diffefwi`
(DeepWave-KAUST, MIT — diffusion prior + Deepwave FWI) is a local reference for the pattern; we
do not import it (it is elastic / custom-DDPM / RED-style, not a clean drop-in).

## Architecture

New `baselines/` subpackage in the package, parallel to `flow_matching/` and `physics/`. The
diffusion model + scheduler come from `diffusers`; the denoiser backbone is built through a
small factory so it can be swapped later (e.g. for our DiT).

### A. Diffusion prior — `baselines/diffusion_prior.py`

```python
def build_denoiser(kind: str = "unet", *, sample_size: int = 64, channels: int = 1) -> nn.Module:
    """Construct the denoiser network. kind='unet' -> diffusers UNet2DModel (the default,
    standard, imported backbone). The factory is the seam to plug an alternative backbone
    (e.g. our DiT) for an architecture-controlled comparison later."""
```
- `kind="unet"` returns a `diffusers.UNet2DModel` sized for `(channels, sample_size, sample_size)`
  velocity maps (a modest config, e.g. `block_out_channels=(64,128,256)`, attention at the lower
  resolutions). Any other `kind` raises `NotImplementedError` (seam only — no DiT this spec).
- A standard predict-noise DDPM training loop:
  ```python
  def train_diffusion_prior(denoiser, scheduler, loader, *, n_epochs, lr, device,
                            log=None) -> list[dict[str, float]]:
      # per step: x1 from loader; t ~ U[0, T); noise ~ N(0,I);
      # x_t = scheduler.add_noise(x1, noise, t); pred = denoiser(x_t, t).sample;
      # loss = mse(pred, noise); Adam step; record/log loss.
  ```
  `scheduler` is a `diffusers.DDPMScheduler` (e.g. `num_train_timesteps=1000`). The loader is
  built from the existing `OpenFWIDatasetConfig().build()` (the same FlatVel-A maps, normalised
  to `[-1, 1]`, that trained the flow prior).
- A `Run`-free script saves the trained denoiser `state_dict` to a checkpoint path.

### B. DPS sampler — `baselines/diffusion_dps.py`

```python
def dps_sample(denoiser, scheduler, shape, forward_fn, d_obs, *, n_samples,
               guidance_strength, normalize_grad=True, device) -> Tensor:
    """Canonical DPS over a diffusers reverse process. Returns (n_samples, *shape) at t=0."""
```
Reverse DDPM with per-step likelihood guidance (verified `diffusers` surface):
```
x = randn(n_samples, *shape)
scheduler.set_timesteps(num_steps)            # num_steps from len(scheduler.timesteps)
for t in scheduler.timesteps:
    x = x.detach().requires_grad_(True)
    eps   = denoiser(x, t).sample             # predicted noise
    out   = scheduler.step(eps, t, x)
    x0hat = out.pred_original_sample           # Tweedie x_hat_0
    loss  = ((forward_fn(x0hat) - d_obs) ** 2).sum()
    (grad,) = torch.autograd.grad(loss, x)     # canonical DPS: through denoiser + forward_fn
    if normalize_grad: grad = grad / per_sample_norm(grad)   # the flow-PoC lesson
    x = (out.prev_sample - guidance_strength * grad).detach()
return x
```
Notes: unlike the flow `guided_sample` (which used the cheap detached approximation), this is the
**canonical** DPS that backpropagates through the denoiser — the faithful literature baseline.
`forward_fn` is the same bridge used by the flow PoC (resize 64→70 → denormalize → `simulate`),
so the two methods are scored on identical physics.

### C. Script — `experiments/0002_fwi_tilting/train_diffusion.py`

A plain script (no Hydra): build the denoiser + scheduler, train the prior on FlatVel-A, save a
checkpoint; then a quick `dps_sample` inversion of the **same held-out map** used in the flow PoC
(simulate `d_obs` with `physics.forward.simulate`, run DPS, render `true | v_hat | error` and
print MAE/RMSE + data-misfit reduction) — the qualitative confirmation that the baseline works
end-to-end, mirroring the flow PoC. Quantitative head-to-head is the next spec.

## Testing

Hermetic — no real UNet training, no Deepwave, run under `WANDB_MODE=disabled` if wandb is touched.

- **`tests/test_diffusion_dps.py`** — the core property mirrors `test_tilt`: with a **mock
  denoiser** (a tiny `nn.Module` whose `.forward(x, t)` returns an object with a `.sample` tensor,
  e.g. zeros so `x0hat` still depends differentiably on `x` via the scheduler), a real but tiny
  `diffusers.DDPMScheduler(num_train_timesteps=10)`, and a cheap linear `forward_fn`, assert the
  guided sample's data misfit is **lower than the unguided** (`guidance_strength=0`) one, and the
  output shape is correct.
- **`tests/test_diffusion_prior.py`** — `build_denoiser("unet", sample_size=16, channels=1)`
  returns a module whose forward on `(2, 1, 16, 16)` + a timestep yields `.sample` of shape
  `(2, 1, 16, 16)`; `build_denoiser("dit")` raises `NotImplementedError`. (Small `sample_size`
  keeps the UNet instantiation/forward fast.)

## Dependencies

`diffusers` and `accelerate` are already in the package dependencies — **no new dependencies**.
(SSIM / `torchmetrics` belongs to the deferred comparison spec.)

## Migration / compatibility

- Purely additive: a new `baselines/` subpackage and one experiment script. No existing code or
  interfaces change. `forward_fn` reuses `physics.forward.simulate` and the `to_mps70` bridge from
  the flow PoC.

## Out of scope (deferred)

- Classical-FWI control and the **quantitative head-to-head comparison** (MAE/SSIM/#forward-solves,
  side-by-side figure) — the next spec; `packages/diffefwi/src/diffefwi/regularization.py`
  (TV/Tikhonov/Laplacian) is a reference for the classical control.
- The DiT diffusion backbone (`build_denoiser` seam only — no implementation here).
- Conditional / amortized diffusion (DiffusionVel-style camp B), supervised baselines, EMA of the
  diffusion weights, RED-style alternating inversion, Hydra/wandb promotion of 0002.
