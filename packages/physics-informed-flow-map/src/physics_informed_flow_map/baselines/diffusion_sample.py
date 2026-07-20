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
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ``n_samples`` from the diffusion prior via the reverse DDPM chain.

    Runs ``num_steps`` reverse steps from Gaussian noise: at each timestep predict the noise
    and take the scheduler's ancestral step. Returns samples at ``t=0`` of shape
    ``(n_samples, *shape)``. When ``on_step`` is provided it is called as
    ``on_step(step_idx, x0hat)`` where ``x0hat`` is the Tweedie clean estimate at each step.

    ``generator`` seeds both the initial noise and the scheduler's ancestral noise, so passing a
    fixed-seed generator makes the whole reverse chain reproducible — used to draw the same
    samples each epoch so per-epoch grids track one image's evolution as the model improves.
    """

    denoiser = denoiser.to(device)
    denoiser.eval()
    scheduler.set_timesteps(num_steps, device=device)  # type: ignore[attr-defined]
    x = torch.randn(n_samples, *shape, device=device, generator=generator)
    for step_idx, t in enumerate(scheduler.timesteps):  # type: ignore[attr-defined]
        eps = denoiser(x, t).sample
        out = scheduler.step(eps, int(t), x, generator=generator)  # type: ignore[attr-defined]
        x = out.prev_sample
        if on_step is not None:
            on_step(step_idx, out.pred_original_sample)
    return x


@torch.no_grad()
def ddpm_sample_trajectory(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, ...],
    *,
    n_samples: int,
    num_steps: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    n_frames: int = 6,
) -> tuple[Tensor, Tensor]:
    """Reverse DDPM chain returning ``n_frames`` evenly-spaced intermediate snapshots.

    The DDPM counterpart of :func:`flow_matching.sample.sample_trajectory`. Returns a pair
    ``(states, x0hats)``, each of shape ``[n_frames, n_samples, *shape]``: the running noisy
    state ``x_t`` after each captured step, and the Tweedie clean estimate ``x0hat`` at that
    step. Pass a fixed-seed ``generator`` to track the same samples across epochs.
    """
    denoiser = denoiser.to(device)
    denoiser.eval()
    scheduler.set_timesteps(num_steps, device=device)  # type: ignore[attr-defined]
    n_frames = min(n_frames, num_steps)
    capture = {
        round(i * (num_steps - 1) / max(n_frames - 1, 1)) for i in range(n_frames)
    }
    x = torch.randn(n_samples, *shape, device=device, generator=generator)
    states, x0hats = [], []
    for step_idx, t in enumerate(scheduler.timesteps):  # type: ignore[attr-defined]
        eps = denoiser(x, t).sample
        out = scheduler.step(eps, int(t), x, generator=generator)  # type: ignore[attr-defined]
        x = out.prev_sample
        if step_idx in capture:
            states.append(x.clone())
            x0hats.append(out.pred_original_sample.clone())
    return torch.stack(states), torch.stack(x0hats)
