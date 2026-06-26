"""Velocity models + a discriminated-union model config.

Both architectures implement mfm's BaseModel.v interface so mfm's loss/sampler
drive them unchanged. The MLP subclasses BaseModel — it does not modify the mfm
library. ``build_model`` dispatches on the (discriminated-union) model config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from mfm.SI import Linear
from mfm.models import DiTMFM
from mfm.models.base_model import BaseModel
from mfm.models.model_wrapper import SIModelWrapper

from physics_informed_flow_map.experiment import Config


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

    def __init__(
        self, dim: int, width: int = 256, depth: int = 4, time_dim: int = 128
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def v(
        self,
        s: Tensor,
        t: Tensor,
        x: Tensor,
        t_cond: Tensor,
        x_cond: Tensor,
        **kwargs: object,
    ) -> Tensor:
        temb = self.time_embed(s)
        result: Tensor = self.net(torch.cat([x, temb], dim=-1))
        return result


class MLPModelConfig(Config):
    """Config for the MLP velocity field (vector data)."""

    kind: Literal["mlp"] = "mlp"
    width: int = 256
    depth: int = 4


class DiTModelConfig(Config):
    """Config for the DiT velocity field (image data)."""

    kind: Literal["dit"] = "dit"
    hidden: int = 128
    depth: int = 4
    num_heads: int = 4
    patch_size: int = 4


ModelConfig = Annotated[MLPModelConfig | DiTModelConfig, Field(discriminator="kind")]


def build_model(
    shape: tuple[int, ...], num_classes: int | None, cfg: ModelConfig
) -> BaseModel:
    """Build the velocity model for a per-sample ``shape`` from a model config.

    ``MLPModelConfig`` requires vector data (``len(shape) == 1``);
    ``DiTModelConfig`` requires square images (``len(shape) == 3``, H == W).
    """
    if isinstance(cfg, MLPModelConfig):
        if len(shape) != 1:
            raise ValueError(f"mlp model requires vector data, got {shape}")
        return VelocityMLP(dim=shape[0], width=cfg.width, depth=cfg.depth)
    if isinstance(cfg, DiTModelConfig):
        if len(shape) != 3:
            raise ValueError(f"dit model requires image data, got {shape}")
        c, h, w = shape
        if h != w:
            raise ValueError(f"DiTMFM requires square images, got {shape}")
        dit = DiTMFM(
            learn_loss_weighting=False,
            input_size=h,
            patch_size=cfg.patch_size,
            in_channels=c,
            hidden_size=cfg.hidden,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            label_dim=num_classes or 1,
            encoder_depth=2,
            attn_func="base",
            is_zero_data=True,
            learn_sigma=False,
        )
        return SIModelWrapper(dit, Linear(t_max=1.0), use_parametrization=False)
    raise ValueError(f"unsupported model config {cfg!r}")


def activate_x_cond_conditioning(model: BaseModel) -> bool:
    """Enable the ``x_cond`` (posterior-conditioning) pathway for time-conditional training.

    mfm zero-initializes ``x_cond_embedder`` and closes its gate (``scale=-1``); with
    ``preserve_t_cond_0=False`` that is a zero-gradient fixed point — ``x_cond_emb==0`` zeroes
    the gradient to both the embedder and the gate's scale, so a from-scratch model never learns
    to use ``x_cond`` (its velocity stays bit-identical across conditioning states). mfm escapes
    this only inside ``init_from_dmf`` by copying ``x_embedder`` into ``x_cond_embedder``; we
    replicate that copy here. Returns ``False`` (no-op) for models without an x_cond pathway,
    e.g. the MLP. Call after any warm-start so it copies the warm-started ``x_embedder``.
    """
    inner = getattr(model, "model", model)
    dit = getattr(inner, "dit", inner)
    if not hasattr(dit, "x_cond_embedder") or not hasattr(dit, "x_embedder"):
        return False
    dit.x_cond_embedder.load_state_dict(dit.x_embedder.state_dict())
    return True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts. They differ for the DiT,
    whose sin-cos positional embedding is a frozen (non-trainable) parameter."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass(frozen=True)
class ModelCase:
    """A model config paired with a representative input contract (for tests/tooling)."""

    config: ModelConfig
    sample_shape: tuple[int, ...]
    num_classes: int | None


MODELS: dict[str, ModelCase] = {
    "mlp": ModelCase(MLPModelConfig(), (2,), None),
    "dit": ModelCase(DiTModelConfig(), (1, 32, 32), 10),
}
