"""RED-DiffEq optimization core, exercised with a stub denoiser and a cheap linear forward
operator so neither a trained network nor the wave solver is needed."""

from dataclasses import dataclass

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn

from physics_informed_flow_map.baselines import red_diffeq_sample


@dataclass
class _Out:
    sample: Tensor


class _ZeroEps(nn.Module):
    """A perfect denoiser for clean inputs: predicts zero noise, so the Tweedie estimate
    ``x0_hat = x_t / sqrt(abar)`` ~ x (the RED residual vanishes at the true model)."""

    def forward(self, x_t: Tensor, t: Tensor) -> _Out:
        return _Out(torch.zeros_like(x_t))


def test_red_diffeq_shapes_and_range() -> None:
    torch.manual_seed(0)
    sched = DDPMScheduler(num_train_timesteps=1000)  # type: ignore[no-untyped-call]

    # Linear forward: data is a fixed projection of the model, so the misfit is differentiable
    # and minimized at models matching d_obs. d_obs built from a known [-1,1] target.
    proj = torch.randn(12, 1 * 16 * 16)

    def forward_fn(x: Tensor) -> Tensor:
        return x.flatten(1) @ proj.T  # (B, 12)

    target = (torch.rand(1, 16, 16) * 2 - 1).clamp(-1, 1)
    d_obs = forward_fn(target.unsqueeze(0)).squeeze(0)  # (12,)

    out = red_diffeq_sample(
        _ZeroEps(),
        sched,
        (1, 16, 16),
        forward_fn,
        d_obs,
        n_samples=3,
        iters=30,
        eta_data=0.1,
        eta_reg=0.05,
        t_denoise=100,
        device=torch.device("cpu"),
    )

    assert out.shape == (3, 1, 16, 16)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_red_diffeq_reduces_data_misfit() -> None:
    # The data term should pull the iterate toward d_obs: misfit after < misfit at the start.
    torch.manual_seed(0)
    sched = DDPMScheduler(num_train_timesteps=1000)  # type: ignore[no-untyped-call]
    proj = torch.randn(20, 16 * 16)

    def forward_fn(x: Tensor) -> Tensor:
        return x.flatten(1) @ proj.T

    target = (torch.rand(1, 16, 16) * 2 - 1).clamp(-1, 1)
    d_obs = forward_fn(target.unsqueeze(0)).squeeze(0)

    from physics_informed_flow_map.physics.classical import linear_gradient_init

    start = linear_gradient_init(1, 16, 16, torch.device("cpu")).unsqueeze(1)
    start_misfit = ((forward_fn(start) - d_obs) ** 2).sum().item()

    out = red_diffeq_sample(
        _ZeroEps(),
        sched,
        (1, 16, 16),
        forward_fn,
        d_obs,
        n_samples=1,
        iters=80,
        eta_data=0.2,
        eta_reg=0.0,
        t_denoise=100,
        device=torch.device("cpu"),
    )
    end_misfit = ((forward_fn(out) - d_obs) ** 2).sum().item()
    assert end_misfit < start_misfit
