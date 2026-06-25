# Diffusion-prior + DPS baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diffusion-prior + DPS posterior-sampling baseline to the 0002 FWI experiment, as the apples-to-apples camp-A comparison against the existing flow-tilting PoC.

**Architecture:** A new `baselines/` subpackage in `physics_informed_flow_map`, parallel to `flow_matching/` and `physics/`. It imports the diffusion machinery from `diffusers` (`UNet2DModel` denoiser + `DDPMScheduler`) behind a small `build_denoiser` factory, adds a standard predict-noise DDPM training loop, and a thin canonical DPS sampler that backpropagates a wave-equation likelihood gradient through the denoiser. A plain experiment script trains the prior on FlatVel-A and inverts the same held-out map the flow PoC used.

**Tech Stack:** Python 3.12, PyTorch, `diffusers` 0.31.0.dev0 (already a dependency), pytest, `uv`. No new dependencies.

## Global Constraints

- **No new dependencies.** `diffusers` and `accelerate` are already in `packages/physics-informed-flow-map/pyproject.toml`; SSIM/`torchmetrics` belong to the deferred comparison spec.
- **Normalisation convention (verbatim):** velocity maps live in `[-1, 1]`; physical m/s is `1500.0` (`VMIN`) to `4500.0` (`VMAX`), imported from `physics_informed_flow_map.flow_matching.openfwi`.
- **Same data, same physics:** the prior trains on the same FlatVel-A maps via `OpenFWIDatasetConfig`, and DPS inverts the same held-out map through the same Deepwave operator (`physics.forward.simulate`) and the same 64→70 bridge (`to_mps70`) the flow PoC used.
- **DPS is canonical:** the likelihood gradient is taken w.r.t. the noisy state `x` and backpropagates **through the denoiser** (unlike the flow PoC's cheaper detached approximation), reusing the **gradient-normalisation** lesson (`normalize_grad=True`).
- **mypy strict** runs over `packages/physics-informed-flow-map/src` and `tests` (not `experiments/`). Every new file under `src`/`tests` must fully type-check; `diffusers.*` is already in the mypy `ignore_missing_imports` override.
- **ruff** lints/formats everything under `packages/physics-informed-flow-map/` and `experiments/`; the pre-commit `ruff format` hook may reformat files — if it does, `git add -A` and re-commit.
- **Factory seam only:** `build_denoiser` supports `kind="unet"` now; any other `kind` raises `NotImplementedError` (no DiT this plan).

---

## File Structure

- `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py` — subpackage exports (`build_denoiser`, `train_diffusion_prior`, `dps_sample`).
- `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_prior.py` — `build_denoiser` factory + `train_diffusion_prior` loop.
- `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_dps.py` — `dps_sample` canonical DPS sampler.
- `packages/physics-informed-flow-map/tests/test_diffusion_prior.py` — `build_denoiser` shape + `NotImplementedError` seam.
- `packages/physics-informed-flow-map/tests/test_diffusion_dps.py` — guided misfit < unguided (mock denoiser, tiny real scheduler, linear forward).
- `experiments/0002_fwi_tilting/train_diffusion.py` — plain script: train prior on FlatVel-A, save checkpoint, DPS-invert the held-out map, render + print metrics.

---

### Task 1: Diffusion prior — `build_denoiser` + `train_diffusion_prior`

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py`
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_prior.py`
- Test: `packages/physics-informed-flow-map/tests/test_diffusion_prior.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Uses `diffusers.UNet2DModel`, `diffusers.DDPMScheduler`, `torch`.
- Produces:
  - `build_denoiser(kind: str = "unet", *, sample_size: int = 64, channels: int = 1) -> nn.Module` — returns a `diffusers.UNet2DModel` for `kind="unet"`; raises `NotImplementedError` for any other `kind`.
  - `train_diffusion_prior(denoiser: nn.Module, scheduler: DDPMScheduler, loader: DataLoader, *, n_epochs: int, lr: float, device: torch.device, log: Callable[..., None] | None = None) -> list[dict[str, float]]` — standard predict-noise DDPM loop; returns per-step history dicts `{"step", "epoch", "loss"}`.

- [ ] **Step 1: Create the subpackage `__init__.py` (prior exports only for now)**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py`:

```python
"""Diffusion-prior + DPS baseline for FWI posterior sampling (camp A comparison)."""

from physics_informed_flow_map.baselines.diffusion_prior import (
    build_denoiser,
    train_diffusion_prior,
)

__all__ = ["build_denoiser", "train_diffusion_prior"]
```

- [ ] **Step 2: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_diffusion_prior.py`:

```python
"""The diffusion-prior denoiser factory: correct output shape and the seam-only guard."""

import pytest
import torch

from physics_informed_flow_map.baselines.diffusion_prior import build_denoiser


def test_build_denoiser_unet_shape() -> None:
    # A small sample_size keeps UNet instantiation/forward fast.
    denoiser = build_denoiser("unet", sample_size=16, channels=1)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([3, 7])
    out = denoiser(x, t).sample
    assert out.shape == (2, 1, 16, 16)


def test_build_denoiser_unknown_kind_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_denoiser("dit")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_diffusion_prior.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'physics_informed_flow_map.baselines.diffusion_prior'` (or ImportError on `build_denoiser`).

- [ ] **Step 4: Implement `diffusion_prior.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_prior.py`:

```python
"""Unconditional diffusion prior over velocity maps (the camp-A baseline prior).

Imports the diffusion machinery from ``diffusers`` rather than hand-rolling it: the denoiser
is a ``UNet2DModel`` built through a small factory (the seam to swap in an alternative
backbone later), trained with the standard predict-noise DDPM objective over OpenFWI velocity
maps normalised to ``[-1, 1]`` — the same data that trained the flow prior.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.utils.data import DataLoader


def build_denoiser(
    kind: str = "unet", *, sample_size: int = 64, channels: int = 1
) -> nn.Module:
    """Construct the denoiser network.

    ``kind="unet"`` returns a ``diffusers.UNet2DModel`` sized for
    ``(channels, sample_size, sample_size)`` velocity maps (a modest config with attention at
    the lowest resolution). The factory is the seam to plug an alternative backbone (e.g. our
    DiT) for an architecture-controlled comparison later; any other ``kind`` raises
    ``NotImplementedError``.
    """
    if kind != "unet":
        raise NotImplementedError(f"denoiser kind {kind!r} is not implemented")
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def train_diffusion_prior(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    log: Callable[..., None] | None = None,
) -> list[dict[str, float]]:
    """Standard predict-noise DDPM training loop.

    Per step: draw a clean velocity map ``x1`` from ``loader`` (the dataset yields
    ``(map, label)``; the label is ignored), sample a timestep ``t ~ U[0, T)`` and Gaussian
    ``noise``, form ``x_t = scheduler.add_noise(x1, noise, t)``, predict the noise with the
    denoiser, and minimise ``mse(pred, noise)``. Returns a per-step history of
    ``{"step", "epoch", "loss"}`` dicts (and calls ``log(**rec)`` if provided).
    """
    denoiser = denoiser.to(device)
    denoiser.train()
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    num_timesteps = int(scheduler.config.num_train_timesteps)

    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(n_epochs):
        for x1, _ in loader:
            x1 = x1.to(device)
            noise = torch.randn_like(x1)
            t = torch.randint(0, num_timesteps, (x1.shape[0],), device=device)
            x_t = scheduler.add_noise(x1, noise, t)
            pred = denoiser(x_t, t).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rec = {
                "step": float(step),
                "epoch": float(epoch),
                "loss": float(loss.item()),
            }
            history.append(rec)
            if log is not None:
                log(**rec)
            step += 1
    return history
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_diffusion_prior.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Type-check and lint**

Run: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests && uv run ruff check packages/physics-informed-flow-map`
Expected: no mypy errors; ruff clean (if `ruff format` would change files, run `uv run ruff format packages/physics-informed-flow-map` and re-stage).

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py \
        packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_prior.py \
        packages/physics-informed-flow-map/tests/test_diffusion_prior.py
git commit -m "feat(baselines): diffusion prior denoiser factory + DDPM training loop"
```

(If the pre-commit `ruff format` hook reformats and aborts the commit, `git add -A` the reformatted files and re-run the same `git commit`.)

---

### Task 2: DPS sampler — `dps_sample`

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_dps.py`
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py` (add `dps_sample` to imports + `__all__`)
- Test: `packages/physics-informed-flow-map/tests/test_diffusion_dps.py`

**Interfaces:**
- Consumes: nothing from Task 1 at runtime (independent module). Uses `diffusers.DDPMScheduler`, `torch`.
- Produces:
  - `dps_sample(denoiser: nn.Module, scheduler: DDPMScheduler, shape: tuple[int, ...], forward_fn: Callable[[Tensor], Tensor], d_obs: Tensor, *, n_samples: int, num_steps: int, guidance_strength: float, device: torch.device, normalize_grad: bool = True) -> Tensor` — canonical DPS over a `diffusers` reverse process; returns samples at `t=0` of shape `(n_samples, *shape)`.

Note on `num_steps`: the spec pseudocode wrote `scheduler.set_timesteps(num_steps)` with an unclear inline comment; the faithful interpretation is an explicit caller-controlled step count, so `dps_sample` takes `num_steps` and calls `scheduler.set_timesteps(num_steps, device=device)` itself.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_diffusion_dps.py`:

```python
"""Canonical DPS over a diffusers reverse process: guidance reduces the data misfit.

Hermetic — a mock denoiser (predicts zero noise, so the Tweedie estimate still depends
differentiably on the state through the scheduler), a tiny real ``DDPMScheduler``, and a
cheap linear forward operator. Mirrors ``test_tilt``: with guidance the final sample's data
misfit must be lower than the unguided (``guidance_strength=0``) one.
"""

from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn

from physics_informed_flow_map.baselines.diffusion_dps import dps_sample


class _MockDenoiser(nn.Module):
    """Predicts zero noise; ``.sample`` keeps the diffusers output-object surface."""

    def forward(self, x: Tensor, t: Tensor) -> SimpleNamespace:
        return SimpleNamespace(sample=torch.zeros_like(x))


def test_guidance_reduces_data_misfit() -> None:
    torch.manual_seed(0)
    shape = (1, 8, 8)
    n_samples, n_meas = 4, 6
    a_mat = torch.randn(n_meas, 64)

    def forward_fn(v: Tensor) -> Tensor:
        return v.flatten(1) @ a_mat.T  # (B, 1, 8, 8) -> (B, n_meas)

    v_target = torch.randn(64)
    d_obs = v_target @ a_mat.T
    scheduler = DDPMScheduler(num_train_timesteps=10)
    dev = torch.device("cpu")

    torch.manual_seed(1)
    guided = dps_sample(
        _MockDenoiser(), scheduler, shape, forward_fn, d_obs,
        n_samples=n_samples, num_steps=10, guidance_strength=0.2, device=dev,
    )
    torch.manual_seed(1)
    unguided = dps_sample(
        _MockDenoiser(), scheduler, shape, forward_fn, d_obs,
        n_samples=n_samples, num_steps=10, guidance_strength=0.0, device=dev,
    )

    assert guided.shape == (n_samples, *shape)
    misfit_guided = float(((forward_fn(guided) - d_obs) ** 2).sum())
    misfit_unguided = float(((forward_fn(unguided) - d_obs) ** 2).sum())
    assert misfit_guided < misfit_unguided
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_diffusion_dps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'physics_informed_flow_map.baselines.diffusion_dps'`.

- [ ] **Step 3: Implement `diffusion_dps.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_dps.py`:

```python
"""Canonical Diffusion Posterior Sampling (DPS) over a diffusers reverse process.

Runs the reverse DDPM chain from noise to a clean sample, bending each step toward data
consistency with the gradient of a measurement misfit through a differentiable forward
operator. Unlike the flow ``guided_sample`` (which uses the cheap detached approximation),
this is the canonical DPS that backpropagates the likelihood gradient **through the
denoiser** — the faithful literature baseline. The denoiser and forward operator are passed
in as callables, so the sampler knows nothing about the specific network or wave solver.
"""

from __future__ import annotations

from typing import Callable

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn


def dps_sample(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, ...],
    forward_fn: Callable[[Tensor], Tensor],
    d_obs: Tensor,
    *,
    n_samples: int,
    num_steps: int,
    guidance_strength: float,
    device: torch.device,
    normalize_grad: bool = True,
) -> Tensor:
    """Canonical DPS over a ``diffusers`` reverse process. Returns ``(n_samples, *shape)`` at ``t=0``.

    For each reverse timestep: predict noise ``eps = denoiser(x, t).sample`` (with ``x``
    requiring grad), take the scheduler step to get the Tweedie estimate
    ``x0hat = step.pred_original_sample`` and the unguided next state
    ``step.prev_sample``, then subtract ``guidance_strength`` times the gradient of
    ``||forward_fn(x0hat) - d_obs||^2`` w.r.t. ``x`` (backpropagating through the denoiser).

    Args:
        denoiser: noise-prediction network; ``denoiser(x, t).sample`` is the predicted noise.
        scheduler: a ``diffusers.DDPMScheduler``; ``set_timesteps(num_steps)`` is called here.
        shape: per-sample shape, e.g. ``(channels, H, W)``.
        forward_fn: differentiable map from a clean sample to predicted data.
        d_obs: observed data, broadcastable against ``forward_fn``'s output.
        n_samples: number of posterior samples to draw.
        num_steps: number of reverse (inference) steps.
        guidance_strength: scale of the likelihood-gradient correction (``0`` = unguided).
        device: device to sample on.
        normalize_grad: if True, scale each sample's correction to unit norm before applying
            ``guidance_strength`` (the gradient-normalisation lesson from the flow PoC).

    Returns:
        Samples at ``t=0``, shape ``(n_samples, *shape)``.
    """
    scheduler.set_timesteps(num_steps, device=device)
    x = torch.randn(n_samples, *shape, device=device)
    for t in scheduler.timesteps:
        x = x.detach().requires_grad_(True)
        eps = denoiser(x, t).sample
        step = scheduler.step(eps, int(t), x)
        x0hat = step.pred_original_sample

        if guidance_strength != 0.0:
            loss = ((forward_fn(x0hat) - d_obs) ** 2).sum()
            (grad,) = torch.autograd.grad(loss, x)
            if normalize_grad:
                norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
                grad = grad / norm.reshape(-1, *([1] * (grad.ndim - 1)))
        else:
            grad = torch.zeros_like(x)

        x = (step.prev_sample - guidance_strength * grad).detach()
    return x
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_diffusion_dps.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Export `dps_sample` from the subpackage `__init__.py`**

Edit `packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py` to:

```python
"""Diffusion-prior + DPS baseline for FWI posterior sampling (camp A comparison)."""

from physics_informed_flow_map.baselines.diffusion_dps import dps_sample
from physics_informed_flow_map.baselines.diffusion_prior import (
    build_denoiser,
    train_diffusion_prior,
)

__all__ = ["build_denoiser", "dps_sample", "train_diffusion_prior"]
```

- [ ] **Step 6: Type-check, lint, and run the package tests**

Run: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests && uv run ruff check packages/physics-informed-flow-map && uv run pytest packages/physics-informed-flow-map/tests -q`
Expected: no mypy errors; ruff clean; all tests pass (including `test_diffusion_prior.py` and `test_diffusion_dps.py`).

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/__init__.py \
        packages/physics-informed-flow-map/src/physics_informed_flow_map/baselines/diffusion_dps.py \
        packages/physics-informed-flow-map/tests/test_diffusion_dps.py
git commit -m "feat(baselines): canonical DPS sampler over diffusers reverse process"
```

(If the pre-commit `ruff format` hook reformats and aborts, `git add -A` and re-run the commit.)

---

### Task 3: Experiment script — `train_diffusion.py`

**Files:**
- Create: `experiments/0002_fwi_tilting/train_diffusion.py`

**Interfaces:**
- Consumes: `build_denoiser`, `train_diffusion_prior`, `dps_sample` from `physics_informed_flow_map.baselines`; `simulate` from `physics_informed_flow_map.physics.forward`; `VMIN`, `VMAX` from `physics_informed_flow_map.flow_matching.openfwi`; `OpenFWIDatasetConfig` from `physics_informed_flow_map.flow_matching.datasets`.
- Produces: a runnable script (no library API). Not mypy-checked (lives under `experiments/`), but **is** ruff-checked/formatted.

This script mirrors the existing `experiments/0002_fwi_tilting/poc.py` (same held-out map, same `to_mps70` bridge, same `forward_fn`, same metrics/figure), swapping the flow prior + `guided_sample` for the diffusion prior + `dps_sample`. It has no unit test (matches the `poc.py` convention); the gate is a CPU smoke run with tiny settings plus ruff.

- [ ] **Step 1: Write the script**

Create `experiments/0002_fwi_tilting/train_diffusion.py`:

```python
"""Diffusion-prior + DPS baseline for FWI: train a diffusion prior over FlatVel-A velocity
maps, then DPS-invert the same held-out map the flow PoC used (camp-A comparison).

Trains an unconditional ``diffusers`` UNet diffusion prior, saves its weights, simulates
seismic data ``d`` from a held-out FlatVel-A map with the Deepwave forward operator, and runs
canonical DPS toward ``d`` to recover the velocity. Plain script (no Hydra). Example:

    uv run python experiments/0002_fwi_tilting/train_diffusion.py --epochs 100 --steps 200 \
        --guidance 0.5 --n-samples 4

A ``--smoke`` flag runs a tiny end-to-end pass (few epochs, few steps, CPU-friendly) for
plumbing checks. Reports per-sample MAE/RMSE (m/s) vs the true map and the data-misfit
reduction vs an unguided sample, and writes a ``true | best v_hat | error`` figure next to
this script. Quantitative head-to-head with the flow PoC is a follow-up.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from physics_informed_flow_map.baselines import (
    build_denoiser,
    dps_sample,
    train_diffusion_prior,
)
from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.flow_matching.openfwi import VMAX, VMIN
from physics_informed_flow_map.physics.forward import simulate

RESOLUTION = 64


def to_mps70(v_norm: torch.Tensor) -> torch.Tensor:
    """(B,1,64,64) in [-1,1] -> (B,70,70) velocity in m/s."""
    v70 = F.interpolate(v_norm, size=70, mode="bilinear", align_corners=False)
    return ((v70 + 1.0) / 2.0 * (VMAX - VMIN) + VMIN)[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-timesteps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=200, help="DPS reverse steps")
    ap.add_argument("--guidance", type=float, default=0.5)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument(
        "--ckpt", default=str(Path(__file__).parent / "diffusion_prior.pt")
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end pass for plumbing checks (overrides epochs/steps)",
    )
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps, args.train_timesteps, args.n_samples = 1, 10, 50, 2

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Diffusion prior: diffusers UNet + DDPM scheduler, trained on FlatVel-A maps.
    denoiser = build_denoiser("unet", sample_size=RESOLUTION, channels=1).to(dev)
    scheduler = DDPMScheduler(num_train_timesteps=args.train_timesteps)
    dataset = OpenFWIDatasetConfig(resolution=RESOLUTION).build()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"training diffusion prior: {len(dataset)} maps, {args.epochs} epochs")
    history = train_diffusion_prior(
        denoiser, scheduler, loader, n_epochs=args.epochs, lr=args.lr, device=dev,
        log=lambda **r: None,
    )
    if history:
        print(f"  final train loss {history[-1]['loss']:.4f}")
    torch.save({"model": denoiser.state_dict()}, args.ckpt)
    print(f"  saved prior -> {args.ckpt}")

    # Held-out FlatVel-A map (native 70x70, m/s) -> observed seismic data.
    f = sorted(glob.glob("data/openfwi/FlatVel_A/model/*.npy"))[-1]
    v_true = (
        torch.from_numpy(np.ascontiguousarray(np.load(f, mmap_mode="r")[-1, 0]))
        .float()
        .to(dev)
    )
    d_obs = simulate(v_true).detach()

    def forward_fn(v_norm: torch.Tensor) -> torch.Tensor:
        v_mps = to_mps70(v_norm)
        return torch.stack([simulate(v_mps[b]) for b in range(v_mps.shape[0])])

    denoiser.eval()
    guided = dps_sample(
        denoiser, scheduler, (1, RESOLUTION, RESOLUTION), forward_fn, d_obs,
        n_samples=args.n_samples, num_steps=args.steps, guidance_strength=args.guidance,
        device=dev, normalize_grad=True,
    )
    unguided = dps_sample(
        denoiser, scheduler, (1, RESOLUTION, RESOLUTION), forward_fn, d_obs,
        n_samples=args.n_samples, num_steps=args.steps, guidance_strength=0.0,
        device=dev, normalize_grad=True,
    )

    vg = to_mps70(guided)
    mae = (vg - v_true).abs().mean(dim=(1, 2))
    rmse = ((vg - v_true) ** 2).mean(dim=(1, 2)).sqrt()
    dm_g = ((forward_fn(guided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((forward_fn(unguided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    best = int(mae.argmin())

    print(f"guidance={args.guidance:g}  steps={args.steps}  n={args.n_samples}")
    print(f"  MAE (m/s):  {[round(x) for x in mae.tolist()]}  best={round(float(mae[best]))}")
    print(f"  RMSE (m/s): {[round(x) for x in rmse.tolist()]}")
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  "
        f"unguided={float(dm_u.mean()):.3e}  ratio={float(dm_g.mean() / dm_u.mean()):.3f}"
    )

    out = Path(__file__).parent / "train_diffusion_result.png"
    vt = v_true.cpu().numpy()
    vh = vg[best].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(vt, cmap="viridis")
    ax[0].set_title("true v")
    ax[0].axis("off")
    ax[1].imshow(vh, cmap="viridis", vmin=vt.min(), vmax=vt.max())
    ax[1].set_title(f"v_hat (MAE {round(float(mae[best]))} m/s)")
    ax[1].axis("off")
    im = ax[2].imshow(vh - vt, cmap="RdBu", vmin=-500, vmax=500)
    ax[2].set_title("error")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint/format the script**

Run: `uv run ruff check experiments/0002_fwi_tilting/train_diffusion.py && uv run ruff format --check experiments/0002_fwi_tilting/train_diffusion.py`
Expected: ruff clean. If `ruff format --check` reports it would reformat, run `uv run ruff format experiments/0002_fwi_tilting/train_diffusion.py`.

- [ ] **Step 3: Smoke-run the script end-to-end (plumbing check)**

Run: `uv run python experiments/0002_fwi_tilting/train_diffusion.py --smoke`
Expected: it trains a tiny prior, prints a final train loss, saves `diffusion_prior.pt`, runs DPS, prints `MAE`/`RMSE`/`data misfit` lines and `wrote .../train_diffusion_result.png` with no exception. (Quality of recovery is irrelevant at `--smoke` settings; this only verifies the end-to-end wiring — dataset → prior → checkpoint → forward operator → DPS → figure.)

- [ ] **Step 4: Commit**

```bash
git add experiments/0002_fwi_tilting/train_diffusion.py
git commit -m "feat(0002): diffusion-prior + DPS baseline experiment script"
```

(If the pre-commit `ruff format` hook reformats and aborts, `git add -A` and re-run the commit. Do not commit the generated `diffusion_prior.pt` / `train_diffusion_result.png` artifacts; the `--smoke` run writes them next to the script — if they are not already git-ignored, leave them unstaged.)

---

## Notes for the executor

- **Artifacts:** the `--smoke` run writes `diffusion_prior.pt` and `train_diffusion_result.png` into `experiments/0002_fwi_tilting/`. Keep them out of the commit (the existing `poc_result.png` lives there but is not the deliverable). Stage only `train_diffusion.py` in Task 3.
- **GPU optional:** all tests and the `--smoke` run are CPU-friendly; a full-quality prior (`--epochs 100`) wants a GPU but is not part of this plan's gates.
- **Out of scope (deferred to the next spec):** classical-FWI control, the quantitative head-to-head comparison (MAE/SSIM/#forward-solves, side-by-side figure), the DiT diffusion backbone, and any Hydra/wandb promotion of 0002.
