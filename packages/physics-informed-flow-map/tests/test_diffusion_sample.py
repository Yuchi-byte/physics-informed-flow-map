"""Unconditional DDPM sampling returns the requested shape."""

import torch
from diffusers import DDPMScheduler

from physics_informed_flow_map.baselines.diffusion_prior import build_denoiser
from physics_informed_flow_map.baselines.diffusion_sample import ddpm_sample


def test_ddpm_sample_shape() -> None:
    torch.manual_seed(0)
    denoiser = build_denoiser("unet", sample_size=16, channels=1)
    scheduler = DDPMScheduler(num_train_timesteps=10)  # type: ignore[no-untyped-call]
    s = ddpm_sample(
        denoiser,
        scheduler,
        (1, 16, 16),
        n_samples=2,
        num_steps=5,
        device=torch.device("cpu"),
    )
    assert s.shape == (2, 1, 16, 16)
