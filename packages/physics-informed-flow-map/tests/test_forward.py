"""Deepwave forward operator: shape/finiteness + a finite-difference gradient check.

The gradient check is the load-bearing test — it proves the operator is differentiable
w.r.t. the velocity model (the VJP that inference-time tilting depends on). Runs on a
small grid in double precision so finite differences are reliable.
"""

import torch

from physics_informed_flow_map.physics.forward import simulate


def _sim(v: torch.Tensor) -> torch.Tensor:
    """Small, fast, energetic survey (freq high enough the wavelet fits the short record)."""
    return simulate(v, dx=10.0, dt=1e-3, nt=120, n_sources=1, n_receivers=12, freq=40.0)


def test_simulate_shape_and_finite() -> None:
    v = torch.full((16, 16), 1500.0)
    v[8:] = 2000.0
    d = _sim(v)
    assert d.shape == (1, 12, 120)
    assert torch.isfinite(d).all()
    assert float((d**2).sum()) > 0.0  # the source actually injected energy


def test_gradient_matches_finite_difference() -> None:
    torch.manual_seed(0)
    v_true = torch.full((16, 16), 1500.0, dtype=torch.float64)
    v_true[8:] = 1900.0
    d_obs = _sim(v_true).detach()

    v = torch.full((16, 16), 1500.0, dtype=torch.float64)
    v[8:] = 1700.0
    v.requires_grad_(True)
    loss = ((_sim(v) - d_obs) ** 2).sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert v.grad is not None
    g_auto = v.grad.detach().clone()
    assert torch.isfinite(g_auto).all()

    def loss_at(vv: torch.Tensor) -> float:
        return float(((_sim(vv) - d_obs) ** 2).sum())

    # Directional finite difference: dot the gradient with a random direction and
    # compare to a central difference of the loss along it. Aggregating over all
    # cells makes this robust (not dominated by individual near-zero-gradient cells).
    direction = torch.randn(16, 16, dtype=torch.float64)
    v0 = v.detach()
    eps = 1e-2  # m/s — small enough to stay in the linear regime of the wave loss
    fd_dir = (loss_at(v0 + eps * direction) - loss_at(v0 - eps * direction)) / (2 * eps)
    auto_dir = float((g_auto * direction).sum())
    assert abs(fd_dir - auto_dir) <= 5e-3 * abs(auto_dir) + 1e-9
