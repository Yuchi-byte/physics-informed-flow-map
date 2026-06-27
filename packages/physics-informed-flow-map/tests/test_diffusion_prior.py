"""The diffusion-prior factory + DDPM training: output shape, seam guard, lifecycle wiring."""

import pytest
import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, TensorDataset

from physics_informed_flow_map.baselines.diffusion_prior import (
    build_denoiser,
    train_diffusion_prior,
)


def _tiny_loader(n: int = 8, size: int = 16) -> DataLoader:
    """A handful of [-1, 1] images with dummy labels, batched 4 -> 2 steps/epoch."""
    x = torch.rand(n, 1, size, size) * 2 - 1
    y = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, drop_last=True)


def test_build_denoiser_unet_shape() -> None:
    # A small sample_size keeps UNet instantiation/forward fast.
    denoiser = build_denoiser("unet", sample_size=16, channels=1)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([3, 7])
    out = denoiser(x, t).sample
    assert out.shape == (2, 1, 16, 16)


def test_build_denoiser_dit_shape() -> None:
    # patch_size must divide sample_size; small dims keep the DiT forward fast.
    denoiser = build_denoiser("dit", sample_size=16, channels=1, patch_size=4)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([3, 7])
    out = denoiser(x, t).sample
    assert out.shape == (2, 1, 16, 16)


def test_build_denoiser_unknown_kind_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_denoiser("nope")


def test_train_diffusion_prior_runs_with_ema_and_logging() -> None:
    torch.manual_seed(0)
    denoiser = build_denoiser("unet", sample_size=16, channels=1)
    scheduler = DDPMScheduler(num_train_timesteps=10)  # type: ignore[no-untyped-call]
    logged: list[dict] = []
    history, ema = train_diffusion_prior(
        denoiser,
        scheduler,
        _tiny_loader(),
        n_epochs=2,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
        ema_enabled=True,
        ema_decay=0.5,
    )
    # Per-step history (2 epochs x 2 steps) with the shared loop's "total" key.
    assert len(history) == 4
    assert {"total", "epoch", "step"} == set(history[0].keys())
    # Per-epoch namespaced logging (2 epochs) + one end-of-run timing summary.
    epoch_logs = [r for r in logged if "train/loss" in r]
    assert len(epoch_logs) == 2
    assert all(
        {"train/loss", "train/grad_norm", "train/lr"} <= r.keys() for r in epoch_logs
    )
    assert sum("time/total_min" in r for r in logged) == 1
    # EMA enabled -> a distinct averaged module is returned.
    assert ema is not None and ema is not denoiser
