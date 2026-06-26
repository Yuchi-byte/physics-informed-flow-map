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

    Meaningful only for a trained meta flow map (``0004``); on a pure flow-matching prior the
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
