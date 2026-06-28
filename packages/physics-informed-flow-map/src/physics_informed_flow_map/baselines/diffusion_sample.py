"""Unconditional reverse-DDPM sampling from a trained diffusion prior.

The counterpart to ``dps_sample`` with no measurement guidance — used to draw prior samples
for evaluation (visual grids).
"""

from __future__ import annotations

from typing import Callable

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
    on_step: Callable[..., None] | None = None,
) -> Tensor:
    """Draw ``n_samples`` from the diffusion prior via the reverse DDPM chain.

    Runs ``num_steps`` reverse steps from Gaussian noise: at each timestep predict the noise
    and take the scheduler's ancestral step. Returns samples at ``t=0`` of shape
    ``(n_samples, *shape)``. When ``on_step`` is provided it is called as
    ``on_step(step_idx, x0hat)`` where ``x0hat`` is the Tweedie clean estimate at each step.
    """

    denoiser = denoiser.to(device)
    denoiser.eval()
    scheduler.set_timesteps(num_steps, device=device)  # type: ignore[attr-defined]
    x = torch.randn(n_samples, *shape, device=device)
    for step_idx, t in enumerate(scheduler.timesteps):  # type: ignore[attr-defined]
        eps = denoiser(x, t).sample
        out = scheduler.step(eps, int(t), x)  # type: ignore[attr-defined]
        x = out.prev_sample
        if on_step is not None:
            on_step(step_idx, out.pred_original_sample)
    return x
