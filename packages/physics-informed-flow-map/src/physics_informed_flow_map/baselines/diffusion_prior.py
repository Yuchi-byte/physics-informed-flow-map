"""Unconditional diffusion prior over velocity maps (the camp-A baseline prior).

Imports the diffusion machinery from ``diffusers`` rather than hand-rolling it: the denoiser
is a ``UNet2DModel`` built through a small factory (the seam to swap in an alternative
backbone later), trained with the standard predict-noise DDPM objective over OpenFWI velocity
maps normalised to ``[-1, 1]`` — the same data that trained the flow prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch import Tensor, nn
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.training.loop import train_loop


@dataclass
class _DenoiserOutput:
    sample: (
        Tensor  # mirrors the diffusers `.sample` attribute the trainer/samplers read
    )


class DiTDenoiser(nn.Module):
    """The flow priors' DiT backbone as an unconditional DDPM ε-denoiser.

    Wraps the same ``build_model`` DiT that 0001/0002 use, called diagonally and
    unconditionally — ``ε(x_t, t) = DiT.v(t, t, x_t | t_cond=0, x_cond=0)`` — so the
    diffusion baseline differs from the flow priors only in objective (predict-noise vs
    flow-matching) and sampler, not in architecture. The integer DDPM timestep is mapped to the
    network's continuous ``[0, 1)`` time.
    """

    def __init__(
        self,
        *,
        sample_size: int,
        channels: int,
        hidden: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 4,
        num_train_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        self.prior = build_model(
            (channels, sample_size, sample_size),
            None,
            DiTModelConfig(
                hidden=hidden, depth=depth, num_heads=num_heads, patch_size=patch_size
            ),
        )
        self.num_train_timesteps = num_train_timesteps

    def forward(self, x_t: Tensor, t: Tensor) -> _DenoiserOutput:
        # Samplers pass a scalar timestep; the DiT needs one continuous time per batch element.
        t = torch.as_tensor(t, device=x_t.device)
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        tc = t.float() / self.num_train_timesteps
        zeros_t = torch.zeros_like(tc)
        eps = self.prior.v(tc, tc, x_t, zeros_t, torch.zeros_like(x_t))
        return _DenoiserOutput(sample=eps)


def build_denoiser(
    kind: str = "unet",
    *,
    sample_size: int = 64,
    channels: int = 1,
    hidden: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    patch_size: int = 4,
    num_train_timesteps: int = 1000,
) -> nn.Module:
    """Construct the denoiser network for ``(channels, sample_size, sample_size)`` maps.

    ``kind="unet"`` returns a ``diffusers.UNet2DModel`` (a modest config with attention at the
    lowest resolution). ``kind="dit"`` returns a :class:`DiTDenoiser` — the flow priors' DiT
    backbone as an ε-denoiser, for an architecture-controlled flow-vs-diffusion comparison (the
    ``hidden``/``depth``/``num_heads``/``patch_size`` should match the flow prior). Any other
    ``kind`` raises ``NotImplementedError``.
    """
    if kind == "unet":
        return UNet2DModel(  # type: ignore[no-untyped-call,return-value]
            sample_size=sample_size,
            in_channels=channels,
            out_channels=channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
    if kind == "dit":
        return DiTDenoiser(
            sample_size=sample_size,
            channels=channels,
            hidden=hidden,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            num_train_timesteps=num_train_timesteps,
        )
    raise NotImplementedError(f"denoiser kind {kind!r} is not implemented")


def train_diffusion_prior(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    log: Callable[..., None] | None = None,
    log_step: Callable[..., None] | None = None,
    warmup_steps: int = 0,
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[..., float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
    precision: str = "fp32",
) -> tuple[list[dict[str, float]], nn.Module | None]:
    """Standard predict-noise DDPM training via the shared ``train_loop``.

    The per-step loss draws a clean map ``x1`` from ``loader`` (the dataset yields
    ``(map, label)``; the label is ignored), samples a timestep ``t ~ U[0, T)`` and Gaussian
    ``noise``, forms ``x_t = scheduler.add_noise(x1, noise, t)``, predicts the noise, and
    minimises ``mse(pred, noise)``. The lifecycle (warmup, EMA, per-epoch ``train/`` logging,
    eval/ckpt cadence) is the shared loop's; returns ``(history, ema_model | None)``.
    """
    num_timesteps = int(scheduler.config.num_train_timesteps)  # type: ignore[attr-defined]

    def compute_loss(model: nn.Module, batch: Any, step: int) -> Tensor:
        x1, _ = batch
        x1 = x1.to(device)
        noise = torch.randn_like(x1)
        t = torch.randint(0, num_timesteps, (x1.shape[0],), device=device)
        x_t = scheduler.add_noise(x1, noise, t)  # type: ignore[attr-defined]
        pred = model(x_t, t).sample
        return F.mse_loss(pred, noise)

    return train_loop(
        denoiser,
        loader,
        compute_loss,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        log=log,
        log_step=log_step,
        warmup_steps=warmup_steps,
        ema_enabled=ema_enabled,
        ema_decay=ema_decay,
        ema_warmup_steps=ema_warmup_steps,
        eval_every_epochs=eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
        precision=precision,
    )
