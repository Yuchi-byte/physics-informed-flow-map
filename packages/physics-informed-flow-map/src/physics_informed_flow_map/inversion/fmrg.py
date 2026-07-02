"""FMRG-E: Flow Map Reward Guidance (Euclidean variant) for FWI.

Implements FMRG-E from Huang et al. 2026 (ICML) ported to the MFM prior
(t=0 noise, t=1 data convention). The key idea: instead of one DPS gradient step on
the Tweedie estimate, run n_opt inner gradient steps in x1-space, then backproject the
resulting correction onto the trajectory with a time-dependent weight derived from
optimal control.

Update rule at each outer Euler step (MFM convention):

    v = velocity_fn(xt, t)                           # marginal prior velocity
    x1_hat = xt + (1 - t_cur) * v                   # Tweedie marginal mean
    x1_opt = x1_hat                                  # refined in x1-space:
    for _ in range(n_opt):
        x1_opt = x1_opt - alpha * grad_{x1} ||F(x1_opt) - d_obs||^2
    wt = (1 - t_cur) * t_next                        # FMRG backprojection weight
    xt_next = xt + dt * v + wt * (x1_opt - x1_hat)  # prior advance + correction

wt peaks near t≈0.5 and decays to zero at both endpoints t=0 and t=1, which is the
optimal weighting from the FMRG control derivation (converted from the FLUX convention
wt_FLUX = t_cur*(1-t_next) where t=1 is noise).

With n_opt=1, this is DPS with the FMRG time-dependent weight replacing constant
guidance_strength. Both flow_tilt and FMRG-E use the marginal velocity v(t,t,xt,0,0);
the difference is entirely in what they do with the gradient.
"""

from __future__ import annotations

from typing import Callable

import torch
from mfm.models.base_model import BaseModel
from torch import Tensor

from .base import InversionResult
from .bridge import seismic_forward, to_mps_native


def fmrg_e_sample(
    velocity_fn: Callable[[Tensor, float], Tensor],
    x0: Tensor,
    forward_fn: Callable[[Tensor], Tensor],
    d_obs: Tensor,
    *,
    sampler_steps: int,
    guidance_strength: float,
    n_opt: int = 1,
    normalize_grad: bool = False,
    on_step: Callable[..., None] | None = None,
) -> Tensor:
    """FMRG-E Euler sampling of a flow prior toward an observation d_obs.

    Args:
        velocity_fn: (xt, t) -> velocity; t is a float in [0, 1).
            Should be the marginal velocity: prior.v(t, t, xt, zeros, zeros).
        x0: initial noise, shape (B, C, H, W).
        forward_fn: differentiable map from a clean sample to predicted seismic data.
        d_obs: observed data, broadcastable against forward_fn's output.
        sampler_steps: number of outer Euler steps.
        guidance_strength: step size alpha for inner gradient descent in x1-space.
            At n_opt=1 this is identical to guidance_strength in DPS/flow_tilt but
            with the FMRG time-dependent weight rather than a constant multiplier.
        n_opt: inner gradient steps per outer step. 1 = single-step (DPS + wt).
        normalize_grad: rescale each sample's inner gradient to the velocity norm before
            applying guidance_strength (so the step size is geometry-independent).
        on_step: optional callback (step, x1_opt, data_fidelity=..., correction_norm=...)
            called every step for trajectory logging and visualisation.

    Returns:
        Samples at t=1, shape (B, C, H, W).
    """
    x = x0
    dt = 1.0 / sampler_steps

    for i in range(sampler_steps):
        t_cur = i * dt
        t_next = (i + 1) * dt

        with torch.no_grad():
            v = velocity_fn(x, t_cur)

        if guidance_strength != 0.0:
            x1_hat = (x + (1.0 - t_cur) * v).detach()
            x1_opt = x1_hat.clone().requires_grad_(True)

            for _ in range(n_opt):
                loss = ((forward_fn(x1_opt) - d_obs) ** 2).sum()
                (grad,) = torch.autograd.grad(loss, x1_opt)
                if normalize_grad:
                    v_norm = v.flatten(1).norm(dim=1).clamp_min(1e-12)
                    g_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
                    grad = grad / g_norm.view(-1, *([1] * (grad.ndim - 1)))
                    grad = grad * v_norm.view(-1, *([1] * (grad.ndim - 1)))
                x1_opt = (x1_opt - guidance_strength * grad).detach().requires_grad_(True)

            x1_opt = x1_opt.detach()
            correction = x1_opt - x1_hat
            wt = (1.0 - t_cur) * t_next
        else:
            x1_hat = (x + (1.0 - t_cur) * v).detach()
            x1_opt = x1_hat
            correction = torch.zeros_like(x)
            wt = 0.0

        if on_step is not None:
            data_fidelity = float(((forward_fn(x1_opt.detach()) - d_obs) ** 2).mean())
            correction_norm = float(correction.flatten(1).norm(dim=1).mean())
            on_step(
                i, x1_opt.detach(),
                data_fidelity=data_fidelity,
                correction_norm=correction_norm,
            )

        with torch.no_grad():
            x = x + dt * v + wt * correction

    return x


class FmrgEModule:
    """FMRG-E steering of a flow-map prior toward seismic data.

    n_opt inner gradient steps in x1-space per outer Euler step, with the
    FMRG optimal-control backprojection weight wt = (1-t_cur)*t_next.
    Wave solves per run = sampler_steps * n_opt * n_samples.
    """

    def __init__(
        self,
        prior: BaseModel,
        *,
        guidance: float,
        steps: int,
        n_opt: int,
        n_samples: int,
        device: torch.device,
        resolution: int = 64,
        normalize_grad: bool = False,
    ) -> None:
        self.name = f"fmrg_e·g{guidance:g}·n{n_opt}"
        self.prior = prior
        self.guidance = guidance
        self.steps = steps
        self.n_opt = n_opt
        self.n_samples = n_samples
        self.device = device
        self.resolution = resolution
        self.normalize_grad = normalize_grad

    def invert(self, d_obs: Tensor) -> InversionResult:
        b = self.n_samples
        x0 = torch.randn(b, 1, self.resolution, self.resolution, device=self.device)
        t_cond = torch.zeros(b, device=self.device)
        x_cond = torch.zeros_like(x0)  # explicit zeros: marginal (unconditional) velocity

        def velocity_fn(x: Tensor, t: float) -> Tensor:
            tb = torch.full((b,), t, device=self.device)
            return self.prior.v(tb, tb, x, t_cond, x_cond)

        samples = fmrg_e_sample(
            velocity_fn,
            x0,
            seismic_forward,
            d_obs,
            sampler_steps=self.steps,
            guidance_strength=self.guidance,
            n_opt=self.n_opt,
            normalize_grad=self.normalize_grad,
        )
        return InversionResult(
            to_mps_native(samples),
            n_solves=self.steps * self.n_opt * b,
        )
