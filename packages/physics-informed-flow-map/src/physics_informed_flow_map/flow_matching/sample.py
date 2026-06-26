"""Sampling via the mfm ODE sampler from noise."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from mfm.SI.samplers import ode_sampler_fn
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
