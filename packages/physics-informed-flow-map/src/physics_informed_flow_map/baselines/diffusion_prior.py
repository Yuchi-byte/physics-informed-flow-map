"""Unconditional diffusion prior over velocity maps (the camp-A baseline prior).

Imports the diffusion machinery from ``diffusers`` rather than hand-rolling it: the denoiser
is a ``UNet2DModel`` built through a small factory (the seam to swap in an alternative
backbone later), trained with the standard predict-noise DDPM objective over OpenFWI velocity
maps normalised to ``[-1, 1]`` — the same data that trained the flow prior.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch import nn
from torch.utils.data import DataLoader


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
) -> list[dict[str, float]]:
    """Standard predict-noise DDPM training loop.

    Per step: draw a clean velocity map ``x1`` from ``loader`` (the dataset yields
    ``(map, label)``; the label is ignored), sample a timestep ``t ~ U[0, T)`` and Gaussian
    ``noise``, form ``x_t = scheduler.add_noise(x1, noise, t)``, predict the noise with the
    denoiser, and minimise ``mse(pred, noise)``. Returns a per-step history of
    ``{"step", "epoch", "loss"}`` dicts (and calls ``log(**rec)`` if provided).
    """
    denoiser = denoiser.to(device)
    denoiser.train()
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    num_timesteps = int(scheduler.config.num_train_timesteps)  # type: ignore[attr-defined]

    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(n_epochs):
        for x1, _ in loader:
            x1 = x1.to(device)
            noise = torch.randn_like(x1)
            t = torch.randint(0, num_timesteps, (x1.shape[0],), device=device)
            x_t = scheduler.add_noise(x1, noise, t)  # type: ignore[attr-defined]
            pred = denoiser(x_t, t).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            rec = {
                "step": float(step),
                "epoch": float(epoch),
                "loss": float(loss.item()),
            }
            history.append(rec)
            if log is not None:
                log(**rec)
            step += 1
    return history
