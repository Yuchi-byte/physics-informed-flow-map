"""DPS-style inference-time tilting: guided sampling of a flow prior toward measurements.

``guided_sample`` runs a flow prior's generative ODE from noise to a sample while bending
each step toward data consistency, using the gradient of a measurement misfit through a
differentiable forward operator. It is model- and physics-agnostic: the flow velocity and
the forward operator are passed in as callables, so the sampler knows nothing about the
specific flow model or the wave solver.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .misfit import MisfitFn


def guided_sample(
    velocity_fn: Callable[[Tensor, float], Tensor],
    x0: Tensor,
    forward_fn: Callable[[Tensor], Tensor],
    d_obs: Tensor,
    *,
    sampler_steps: int,
    guidance_strength: float,
    normalize_grad: bool = False,
    misfit_fn: MisfitFn | None = None,
    on_step: Callable[..., None] | None = None,
) -> Tensor:
    """Guided Euler sampling of a flow prior toward an observation ``d_obs``.

    Integrates ``dx/dt = velocity_fn(x, t)`` from ``t=0`` (``x = x0``, noise) to ``t=1``,
    adding at each step a likelihood-gradient correction
    ``-guidance_strength * d/dx ||forward_fn(x1_hat) - d_obs||^2``, where
    ``x1_hat = x + (1 - t) * v`` is the clean-sample estimate for the linear interpolant
    ``x_t = (1 - t) * noise + t * x1``. The velocity is evaluated without grad (cheap); the
    correction backpropagates only through ``forward_fn`` (the forward operator's adjoint),
    i.e. the standard DPS approximation that treats ``x1_hat``'s dependence on ``x`` as
    the identity term.

    Args:
        velocity_fn: maps ``(state, t)`` to the velocity field; ``t`` is a float in ``[0, 1)``.
        x0: initial noise, shape ``(B, *sample_shape)``.
        forward_fn: differentiable map from a clean sample (prior space) to predicted data.
        d_obs: observed data, broadcastable against ``forward_fn``'s output.
        sampler_steps: number of Euler steps.
        guidance_strength: scale of the likelihood-gradient correction (``0`` = unguided).
        normalize_grad: if True, scale each sample's correction to unit norm before
            applying ``guidance_strength`` (so the step size is independent of the wildly
            scale-dependent raw gradient magnitude). ``guidance_strength`` is then a
            state-space step size per Euler step.
        misfit_fn: guidance data-misfit ``pred -> (B,)`` (see ``physics.misfit``);
            ``None`` keeps the historical pointwise L2 against ``d_obs``. The
            ``data_fidelity`` diagnostic stays L2 either way so runs remain comparable.
        on_step: optional callback ``(step, x1_hat, xt=..., data_fidelity=..., grad_norm=...)``
            called every step for trajectory logging and visualisation (``xt`` is the
            noisy interpolant state the Tweedie estimate was taken from).

    Returns:
        Samples at ``t=1``, shape ``(B, *sample_shape)``.
    """
    x = x0
    dt = 1.0 / sampler_steps

    def data_loss(pred: Tensor) -> Tensor:
        if misfit_fn is not None:
            return misfit_fn(pred).sum()
        return ((pred - d_obs) ** 2).sum()

    for i in range(sampler_steps):
        t = i * dt
        with torch.no_grad():
            v = velocity_fn(x, t)

        if guidance_strength != 0.0:
            x_g = x.detach().requires_grad_(True)
            x1_hat = x_g + (1.0 - t) * v
            loss = data_loss(forward_fn(x1_hat))
            (grad,) = torch.autograd.grad(loss, x_g)
            if normalize_grad:
                norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
                grad = grad / norm.reshape(-1, *([1] * (grad.ndim - 1)))
        else:
            x1_hat = (x + (1.0 - t) * v).detach()
            grad = torch.zeros_like(x)

        if on_step is not None:
            data_fidelity = float(((forward_fn(x1_hat.detach()) - d_obs) ** 2).mean())
            grad_norm = float(grad.flatten(1).norm(dim=1).mean())
            on_step(
                i,
                x1_hat.detach(),
                xt=x.detach(),
                data_fidelity=data_fidelity,
                grad_norm=grad_norm,
            )

        with torch.no_grad():
            x = x + dt * v - guidance_strength * grad
    return x
