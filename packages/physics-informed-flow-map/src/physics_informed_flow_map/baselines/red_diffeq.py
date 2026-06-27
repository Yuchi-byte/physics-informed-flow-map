"""RED-DiffEq: Regularization by Denoising with a diffusion prior for FWI.

A different way to use the learned diffusion prior of ``0003`` than DPS. Instead of running the
reverse diffusion chain, this is a model-domain *optimization*: gradient descent on the velocity
that descends the wave-equation data misfit while a Regularization-by-Denoising (RED) term pulls
each iterate toward the diffusion prior's manifold. The prior pull is the residual ``x - D(x)``
of a Tweedie denoise at a fixed noise level — the diffusion model acting as the denoiser ``D`` a
classical RED scheme would call. Steering stays at inference time (camp A); the prior is reused,
never retrained.

The optimization lives in the prior's normalized ``[-1, 1]`` space at the prior resolution; the
``forward_fn`` callable maps a normalized model to predicted data (it owns the resize to the
physical grid), so this module knows nothing about the wave solver.
"""

from __future__ import annotations

from typing import Callable

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn

from ..physics.classical import linear_gradient_init


def red_diffeq_sample(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, ...],
    forward_fn: Callable[[Tensor], Tensor],
    d_obs: Tensor,
    *,
    n_samples: int,
    iters: int,
    eta_data: float,
    eta_reg: float,
    t_denoise: int,
    device: torch.device,
    normalize_grad: bool = True,
) -> Tensor:
    """RED-DiffEq optimization; returns ``(n_samples, *shape)`` normalized models in ``[-1, 1]``.

    From a smooth gradient start, each iteration applies two corrections:

    * **data** — ``-eta_data * d/dx ||forward_fn(x) - d_obs||^2`` (the wave-equation pull;
      per-sample unit-normalized when ``normalize_grad`` so ``eta_data`` is a state-space step).
    * **prior** — ``-eta_reg * (x - D(x))``, the RED residual, where ``D`` is the diffusion
      Tweedie denoiser at fixed level ``t_denoise``: renoise ``x_t = sqrt(abar) x +
      sqrt(1-abar) eps``, predict the noise, and back out ``x0_hat = (x_t - sqrt(1-abar)
      eps_pred) / sqrt(abar)``. Fresh noise each step also gives the ``n_samples`` restarts a
      stochastic spread.

    Args:
        denoiser: eps-prediction network; ``denoiser(x_t, t).sample`` is the predicted noise.
        scheduler: a ``DDPMScheduler`` (for ``alphas_cumprod`` at ``t_denoise``).
        shape: per-sample shape, e.g. ``(channels, H, W)``.
        forward_fn: differentiable normalized-model -> predicted-data map.
        d_obs: observed data, broadcastable against ``forward_fn``'s output.
        n_samples: number of restarts (posterior-like samples).
        iters: optimization steps. ``t_denoise``: the fixed DDPM noise level for the RED denoiser.
        eta_data / eta_reg: data and prior step sizes. device: device to optimize on.
        normalize_grad: unit-normalize each sample's data gradient before stepping.

    Returns:
        ``(n_samples, *shape)`` normalized models in ``[-1, 1]``.
    """
    _, h, w = shape
    x = linear_gradient_init(n_samples, h, w, device).unsqueeze(1)  # (n, 1, H, W)
    abar = scheduler.alphas_cumprod.to(device)[t_denoise]  # type: ignore[attr-defined]
    sa, s1 = abar.sqrt(), (1.0 - abar).sqrt()
    t_batch = torch.full((n_samples,), t_denoise, device=device, dtype=torch.long)

    for _ in range(iters):
        x_g = x.detach().requires_grad_(True)
        loss = ((forward_fn(x_g) - d_obs) ** 2).sum()
        (grad,) = torch.autograd.grad(loss, x_g)
        if normalize_grad:
            norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
            grad = grad / norm.reshape(-1, *([1] * (grad.ndim - 1)))

        with torch.no_grad():
            eps = torch.randn_like(x)
            x_t = sa * x + s1 * eps
            eps_pred = denoiser(x_t, t_batch).sample
            x0_hat = (x_t - s1 * eps_pred) / sa
            x = (x - eta_data * grad - eta_reg * (x - x0_hat)).clamp(-1.0, 1.0)

    return x.detach()
