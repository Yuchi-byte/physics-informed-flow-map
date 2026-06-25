"""The diffusion-prior denoiser factory: correct output shape and the seam-only guard."""

import pytest
import torch

from physics_informed_flow_map.baselines.diffusion_prior import build_denoiser


def test_build_denoiser_unet_shape() -> None:
    # A small sample_size keeps UNet instantiation/forward fast.
    denoiser = build_denoiser("unet", sample_size=16, channels=1)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([3, 7])
    out = denoiser(x, t).sample
    assert out.shape == (2, 1, 16, 16)


def test_build_denoiser_unknown_kind_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_denoiser("dit")
