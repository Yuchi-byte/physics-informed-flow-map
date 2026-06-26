"""Unconditional diffusion prior over velocity maps (the camp-A baseline prior).

Imports the diffusion machinery from ``diffusers`` rather than hand-rolling it: the denoiser
is a ``UNet2DModel`` built through a small factory (the seam to swap in an alternative
backbone later), trained with the standard predict-noise DDPM objective over OpenFWI velocity
maps normalised to ``[-1, 1]`` — the same data that trained the flow prior.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch import Tensor, nn
from torch.utils.data import DataLoader

from physics_informed_flow_map.training.loop import train_loop


def build_denoiser(
    kind: str = "unet", *, sample_size: int = 64, channels: int = 1
) -> nn.Module:
    """Construct the denoiser network.

    ``kind="unet"`` returns a ``diffusers.UNet2DModel`` sized for
    ``(channels, sample_size, sample_size)`` velocity maps (a modest config with attention at
    the lowest resolution). The factory is the seam to plug an alternative backbone (e.g. our
    DiT) for an architecture-controlled comparison later; any other ``kind`` raises
    ``NotImplementedError``.
    """
    if kind != "unet":
        raise NotImplementedError(f"denoiser kind {kind!r} is not implemented")
    return UNet2DModel(  # type: ignore[no-untyped-call,return-value]
        sample_size=sample_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def train_diffusion_prior(
    denoiser: nn.Module,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    log: Callable[..., None] | None = None,
    warmup_steps: int = 0,
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[..., float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
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
        warmup_steps=warmup_steps,
        ema_enabled=ema_enabled,
        ema_decay=ema_decay,
        ema_warmup_steps=ema_warmup_steps,
        eval_every_epochs=eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
