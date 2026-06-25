"""Guided-sampling tilt: the core property is that guidance reduces the data misfit.

Hermetic — uses a trivial zero-velocity flow and a cheap linear forward operator, so no
flow model or wave solver is needed. With zero velocity the ODE leaves the state at the
noise, so guided sampling reduces to gradient descent on ||A x - d_obs||^2.
"""

import torch

from physics_informed_flow_map.physics.tilt import guided_sample


def test_guidance_reduces_data_misfit() -> None:
    torch.manual_seed(0)
    n_dim, n_meas, n_samples = 8, 5, 3
    a_mat = torch.randn(n_meas, n_dim, dtype=torch.float64)

    def forward_fn(v: torch.Tensor) -> torch.Tensor:
        return v @ a_mat.T  # (B, n_dim) -> (B, n_meas)

    def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return torch.zeros_like(x)

    v_target = torch.randn(n_dim, dtype=torch.float64)
    d_obs = v_target @ a_mat.T  # (n_meas,)
    x0 = torch.randn(n_samples, n_dim, dtype=torch.float64)

    x_hat = guided_sample(
        velocity_fn, x0, forward_fn, d_obs, sampler_steps=100, guidance_strength=0.01
    )
    assert x_hat.shape == x0.shape

    misfit_unguided = float(((forward_fn(x0) - d_obs) ** 2).sum())
    misfit_guided = float(((forward_fn(x_hat) - d_obs) ** 2).sum())
    assert misfit_guided < misfit_unguided


def test_zero_guidance_is_pure_ode() -> None:
    # With zero velocity and zero guidance, the sample must be left at the initial noise.
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, dtype=torch.float64)

    def forward_fn(v: torch.Tensor) -> torch.Tensor:
        return v

    def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        return torch.zeros_like(x)

    x_hat = guided_sample(
        velocity_fn,
        x0,
        forward_fn,
        torch.zeros(4, dtype=torch.float64),
        sampler_steps=10,
        guidance_strength=0.0,
    )
    assert torch.allclose(x_hat, x0)
