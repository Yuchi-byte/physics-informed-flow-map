"""Sampling from a flow prior: the multi-step ODE sampler, and the few-step flow-map
(consistency) sampler that only a meta flow map supports."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from mfm.SI.samplers import consistency_sampler_fn, ode_sampler_fn
from mfm.models.base_model import BaseModel


@torch.no_grad()
def sample(
    model: BaseModel,
    n_samples: int,
    shape: tuple[int, ...],
    *,
    sampler_steps: int,
    device: torch.device,
) -> Tensor:
    """Sample from the flow prior via Euler ODE integration (torchdiffeq)."""
    model.eval()
    x_noise = torch.randn(n_samples, *shape, device=device)
    t_cond = torch.zeros(n_samples, device=device)
    return cast(
        Tensor,
        ode_sampler_fn(
            model,
            xt_cond=x_noise,
            t_cond=t_cond,
            n_steps=sampler_steps,
            solver="euler",
            eps_start=x_noise,
            v_type="standard",
        ),
    )


@torch.no_grad()
def sample_few_step(
    model: BaseModel,
    n_samples: int,
    shape: tuple[int, ...],
    *,
    n_steps: int,
    device: torch.device,
) -> Tensor:
    """Few-step flow-map sampling: walk ``s→u`` in ``n_steps`` jumps via ``model.X(s,u,x,v)``.

    Meaningful only for a trained meta flow map (``0002``); on a pure flow-matching prior the
    off-diagonal ``v(s,u)`` is unconstrained, so few-step samples will be poor.
    """
    model.eval()
    x_noise = torch.randn(n_samples, *shape, device=device)
    t_cond = torch.zeros(n_samples, device=device)
    return cast(
        Tensor,
        consistency_sampler_fn(
            model,
            xt_cond=x_noise,
            t_cond=t_cond,
            n_steps=n_steps,
            eps_start=x_noise,
        ),
    )


@torch.no_grad()
def sample_posterior(
    model: BaseModel,
    refs: Tensor,
    t_cond: float,
    *,
    n_steps: int,
    device: torch.device,
) -> Tensor:
    """Draw a posterior sample $x_1\\sim p(x_1\\mid x_{t})$ per reference.

    Builds the partially-noised intermediate state $x_t=(1-t)\\epsilon+t\\,x_1$ at ``t_cond``
    from each reference, then conditions the flow map on it. A **time-conditional** flow map
    (trained with ``t_cond>0``) reconstructs the reference — sharper as ``t_cond``→1; an
    unconditional map ignores the conditioning and returns generic samples. The test that the
    ``t_cond>0`` training actually took.
    """
    model.eval()
    refs = refs.to(device)
    tb = torch.full((refs.shape[0],) + (1,) * (refs.ndim - 1), t_cond, device=device)
    xt_cond = (1 - tb) * torch.randn_like(refs) + tb * refs
    t_cond_vec = torch.full((refs.shape[0],), t_cond, device=device)
    return cast(
        Tensor,
        consistency_sampler_fn(
            model,
            xt_cond=xt_cond,
            t_cond=t_cond_vec,
            n_steps=n_steps,
            eps_start=torch.randn_like(refs),
        ),
    )
