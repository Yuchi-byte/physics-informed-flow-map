"""Velocity models. A small MLP for low-dim data; mfm's DiT for images.

Both implement mfm's BaseModel.v interface so mfm's loss/sampler drive them
unchanged. The MLP subclasses BaseModel — it does not modify the mfm library.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from mfm.SI import Linear
from mfm.models import DiTMFM
from mfm.models.base_model import BaseModel
from mfm.models.model_wrapper import SIModelWrapper

from physics_informed_flow_map.flow_matching.datasets import DatasetSpec


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar time in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class VelocityMLP(BaseModel):  # type: ignore[misc]
    """Time-conditioned MLP velocity field v(t, x) for vector data.

    Ignores t_cond/x_cond/class_labels (unconditional flow matching).
    """

    def __init__(self, dim: int, width: int = 256, depth: int = 4, time_dim: int = 128) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def v(self, s: Tensor, t: Tensor, x: Tensor, t_cond: Tensor, x_cond: Tensor, **kwargs: object) -> Tensor:
        temb = self.time_embed(s)
        result: Tensor = self.net(torch.cat([x, temb], dim=-1))
        return result


def build_model(
    spec: DatasetSpec,
    *,
    mlp_width: int = 256,
    mlp_depth: int = 4,
    dit_hidden: int = 128,
    dit_depth: int = 4,
    num_heads: int = 4,
) -> BaseModel:
    if len(spec.shape) == 1:
        return VelocityMLP(dim=spec.shape[0], width=mlp_width, depth=mlp_depth)
    if len(spec.shape) == 3:
        c, h, _ = spec.shape
        dit = DiTMFM(
            learn_loss_weighting=False,
            input_size=h,
            patch_size=4,
            in_channels=c,
            hidden_size=dit_hidden,
            depth=dit_depth,
            num_heads=num_heads,
            label_dim=spec.num_classes or 1,
            encoder_depth=2,
            attn_func="base",
            is_zero_data=True,
            learn_sigma=False,
        )
        return SIModelWrapper(dit, Linear(t_max=1.0), use_parametrization=False)
    raise ValueError(f"unsupported sample shape {spec.shape}")
