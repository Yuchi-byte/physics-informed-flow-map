"""Canonical DPS over a diffusers reverse process: guidance reduces the data misfit.

Hermetic — a mock denoiser (predicts zero noise, so the Tweedie estimate still depends
differentiably on the state through the scheduler), a tiny real ``DDPMScheduler``, and a
cheap linear forward operator. Mirrors ``test_tilt``: with guidance the final sample's data
misfit must be lower than the unguided (``guidance_strength=0``) one.
"""

from types import SimpleNamespace

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn

from physics_informed_flow_map.baselines.diffusion_dps import dps_sample


class _MockDenoiser(nn.Module):
    """Predicts zero noise; ``.sample`` keeps the diffusers output-object surface."""

    def forward(self, x: Tensor, t: Tensor) -> SimpleNamespace:
        return SimpleNamespace(sample=torch.zeros_like(x))


def test_guidance_reduces_data_misfit() -> None:
    torch.manual_seed(0)
    shape = (1, 8, 8)
    n_samples, n_meas = 4, 6
    a_mat = torch.randn(n_meas, 64)

    def forward_fn(v: Tensor) -> Tensor:
        return v.flatten(1) @ a_mat.T  # (B, 1, 8, 8) -> (B, n_meas)

    v_target = torch.randn(64)
    d_obs = v_target @ a_mat.T
    scheduler = DDPMScheduler(num_train_timesteps=10)  # type: ignore[no-untyped-call]
    dev = torch.device("cpu")

    torch.manual_seed(1)
    guided = dps_sample(
        _MockDenoiser(),
        scheduler,
        shape,
        forward_fn,
        d_obs,
        n_samples=n_samples,
        num_steps=10,
        guidance_strength=0.2,
        device=dev,
    )
    torch.manual_seed(1)
    unguided = dps_sample(
        _MockDenoiser(),
        scheduler,
        shape,
        forward_fn,
        d_obs,
        n_samples=n_samples,
        num_steps=10,
        guidance_strength=0.0,
        device=dev,
    )

    assert guided.shape == (n_samples, *shape)
    misfit_guided = float(((forward_fn(guided) - d_obs) ** 2).sum())
    misfit_unguided = float(((forward_fn(unguided) - d_obs) ** 2).sum())
    assert misfit_guided < misfit_unguided
