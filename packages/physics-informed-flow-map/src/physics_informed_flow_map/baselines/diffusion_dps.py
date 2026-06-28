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
    on_step: Callable[..., None] | None = None,
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
        on_step: optional callback ``(step_idx, x0hat, data_fidelity=...)`` for trajectory logging.

    Returns:
        Samples at ``t=0``, shape ``(n_samples, *shape)``.
    """
    scheduler.set_timesteps(num_steps, device=device)  # type: ignore[attr-defined]
    x = torch.randn(n_samples, *shape, device=device)
    for step_idx, t in enumerate(scheduler.timesteps):  # type: ignore[attr-defined]
        x = x.detach().requires_grad_(True)
        eps = denoiser(x, t).sample
        step = scheduler.step(eps, int(t), x)  # type: ignore[attr-defined]
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

        if on_step is not None:
            data_fidelity = float(((forward_fn(x0hat.detach()) - d_obs) ** 2).mean())
            on_step(step_idx, x0hat.detach(), data_fidelity=data_fidelity)
    return x
