"""Unconditional reverse-DDPM sampling from a trained diffusion prior.

The counterpart to ``dps_sample`` with no measurement guidance — used to draw prior samples
for evaluation (visual grids, energy distance vs held-out reals).
"""

from __future__ import annotations

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn


@torch.no_grad()
def ddpm_sample(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, ...],
    *,
    n_samples: int,
    num_steps: int,
    device: torch.device,
) -> Tensor:
    """Draw ``n_samples`` from the diffusion prior via the reverse DDPM chain.

    Runs ``num_steps`` reverse steps from Gaussian noise: at each timestep predict the noise
    and take the scheduler's ancestral step. Returns samples at ``t=0`` of shape
    ``(n_samples, *shape)``.
    """
    denoiser = denoiser.to(device)
    denoiser.eval()
    scheduler.set_timesteps(num_steps, device=device)  # type: ignore[attr-defined]
    x = torch.randn(n_samples, *shape, device=device)
    for t in scheduler.timesteps:  # type: ignore[attr-defined]
        eps = denoiser(x, t).sample
        x = scheduler.step(eps, int(t), x).prev_sample  # type: ignore[attr-defined]
    return x
