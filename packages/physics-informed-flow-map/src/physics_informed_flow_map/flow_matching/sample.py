"""Sampling (mfm ODE sampler from noise) + an energy-distance eval metric."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

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


def _pdist_mean(a: Tensor, b: Tensor) -> Tensor:
    return torch.cdist(a, b).mean()


def _self_pdist_mean(a: Tensor) -> Tensor:
    """Mean pairwise distance within a set, excluding the (zero) diagonal."""
    n = a.shape[0]
    return torch.cdist(a, a).sum() / max(n * (n - 1), 1)


def energy_distance(x: Tensor, y: Tensor) -> float:
    """Energy distance between two point sets (lower = closer distributions)."""
    x = x.flatten(1) if x.ndim > 2 else x
    y = y.flatten(1) if y.ndim > 2 else y
    val = 2 * _pdist_mean(x, y) - _self_pdist_mean(x) - _self_pdist_mean(y)
    return float(val.item())


def real_reference(dataset: Dataset, n: int, device: torch.device) -> Tensor:
    idx = torch.randperm(len(dataset))[:n]  # type: ignore[arg-type]
    xs = torch.stack([dataset[int(i)][0] for i in idx])
    return xs.to(device)
