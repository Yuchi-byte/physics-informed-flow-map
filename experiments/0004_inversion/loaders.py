"""Load a trained prior network from a checkpoint for the inversion entry points.

Pure I/O at the edges: build the architecture for the requested prior family at the given DiT
shape, then load a ``run.py`` checkpoint's ``["model"]`` state. ``run.py`` (single-target) and
``eval.py`` (multi-map sweep) share these so the prior-loading rules live in one place; how to
*invert* stays with each entry point.
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DDPMScheduler
from mfm.models.base_model import BaseModel
from torch import nn

from physics_informed_flow_map.baselines import build_denoiser
from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model

# Flow-matching (0001) and flow-map (0002) priors share this DiT backbone; the diffusion prior
# (0003) wraps the same backbone as an eps-denoiser when denoiser_kind="dit".
FLOW_PRIORS = ("flow_matching", "flow_map")
DIFFUSION_PRIOR = "diffusion"


def _train_hint(prior: str) -> str:
    fw = {
        "flow_matching": "0001_flow_matching",
        "flow_map": "0002_flow_map",
        "diffusion": "0003_baselines",
    }[prior]
    return (
        f"Train a prior first (uv run python experiments/{fw}/run.py experiment=openfwi) "
        f"and point ckpt= at its checkpoint (runs/{fw}/<ts>/checkpoints/step_<N>*.pt)."
    )


def _load_state(net: nn.Module, ckpt: str, device: torch.device, prior: str) -> None:
    if not ckpt:
        print(
            f"[inversion] no ckpt= given: using an UNTRAINED {prior} prior (plumbing only)"
        )
        return
    if not Path(ckpt).is_file():
        raise SystemExit(f"ckpt not found: {ckpt}\n{_train_hint(prior)}")
    net.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )


def load_flow_prior(
    shape: tuple[int, ...],
    *,
    hidden: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    ckpt: str,
    device: torch.device,
    prior: str = "flow_matching",
) -> BaseModel:
    """Flow / flow-map prior: the DiT velocity model loaded from a checkpoint (eval mode)."""
    net = build_model(
        shape,
        None,
        DiTModelConfig(
            hidden=hidden, depth=depth, num_heads=num_heads, patch_size=patch_size
        ),
    ).to(device)
    _load_state(net, ckpt, device, prior)
    net.eval()
    return net


def load_diffusion_prior(
    shape: tuple[int, ...],
    *,
    denoiser_kind: str,
    hidden: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    num_train_timesteps: int,
    ckpt: str,
    device: torch.device,
) -> tuple[nn.Module, DDPMScheduler]:
    """Diffusion prior: the diffusers denoiser + its DDPM scheduler, loaded from a checkpoint."""
    channels, size, _ = shape
    denoiser = build_denoiser(
        denoiser_kind,
        sample_size=size,
        channels=channels,
        hidden=hidden,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        num_train_timesteps=num_train_timesteps,
    ).to(device)
    _load_state(denoiser, ckpt, device, "diffusion")
    denoiser.eval()
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)  # type: ignore[no-untyped-call]
    return denoiser, scheduler
