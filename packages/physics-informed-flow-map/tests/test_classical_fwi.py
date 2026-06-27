"""Classical regularized FWI: the regularisers and the optimiser loop, exercised with a cheap
identity forward operator (data == velocity in m/s) so no wave solver is needed."""

import torch

from physics_informed_flow_map.flow_matching.openfwi import VMAX, VMIN
from physics_informed_flow_map.physics.classical import (
    linear_gradient_init,
    regularization,
    regularized_fwi,
)


def _identity(v: torch.Tensor) -> torch.Tensor:
    return v


def test_regularization_zero_on_constant() -> None:
    v = torch.full((2, 8, 8), 0.3)
    assert regularization(v, "tikhonov").item() == 0.0
    # TV of a constant field is eps per interior pixel, not zero.
    assert abs(regularization(v, "tv", tv_eps=1e-3).item() - 1e-3 * 2 * 49) < 1e-6


def test_regularization_penalizes_roughness() -> None:
    smooth = torch.linspace(-1, 1, 8).reshape(1, 1, 8).expand(1, 8, 8).contiguous()
    rough = torch.randn(1, 8, 8)
    for kind in ("tikhonov", "tv"):
        assert regularization(rough, kind) > regularization(smooth, kind)


def test_linear_gradient_init_shape_and_range() -> None:
    x = linear_gradient_init(3, 16, 16, torch.device("cpu"))
    assert x.shape == (3, 16, 16)
    assert x.min() >= -1.0 and x.max() <= 1.0
    # Depth gradient: the bottom row sits above the top row on average.
    assert x[:, -1, :].mean() > x[:, 0, :].mean()


def test_regularized_fwi_recovers_identity_target() -> None:
    # With an identity forward, FWI just fits v to d_obs; check it drives the misfit down and
    # returns physical-range velocities of the right shape.
    torch.manual_seed(0)
    target = torch.full((6, 6), 3000.0)  # m/s, mid-range
    d_obs = _identity(target)

    v, n_solves = regularized_fwi(
        _identity,
        d_obs,
        shape=(6, 6),
        n_samples=2,
        iters=150,
        lr=0.05,
        reg="tikhonov",
        reg_weight=0.0,
        device=torch.device("cpu"),
    )

    assert v.shape == (2, 6, 6)
    assert n_solves == 300
    assert VMIN <= float(v.min()) and float(v.max()) <= VMAX
    # Started from a depth gradient; fitting a constant target should land close to 3000.
    assert (v - target).abs().mean() < 150.0
