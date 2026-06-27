"""The inversion scoring core: SSIM, per-target metrics, aggregation, and the Evaluator loop
— all exercised with a cheap identity forward operator (no wave solver)."""

import torch

from physics_informed_flow_map.inversion import (
    Evaluator,
    InversionResult,
    score_target,
    ssim,
)
from physics_informed_flow_map.inversion.evaluate import InversionStats


def _identity(v: torch.Tensor) -> torch.Tensor:
    """A stand-in forward operator: data == velocity, so misfit == squared error."""
    return v


def test_ssim_identity_and_degradation() -> None:
    torch.manual_seed(0)
    x = (
        torch.rand(16, 16) * 2.0 - 1.0
    )  # [-1, 1] normalised scale (default data_range=2)
    assert ssim(x, x) == 1.0 or abs(ssim(x, x) - 1.0) < 1e-4
    assert ssim(x, x + torch.randn_like(x) * 0.3) < 0.99


def test_score_target_expected_over_samples() -> None:
    # Metrics are on the [-1, 1] scale: m/s are normalised by (v - 1500) / 3000 * 2 - 1.
    # 1500 -> -1; samples at 1500, 1800, 1650 -> -1, -0.8, -0.9.
    v_true = torch.full((8, 8), 1500.0)
    v_hat = torch.stack([v_true, v_true + 300.0, v_true + 150.0])
    s = score_target(v_hat, v_true, _identity(v_true), _identity)

    # Expected MAE = mean(|0|, |0.2|, |0.1|) = 0.1 on the normalised scale.
    assert abs(s["mae"] - 0.1) < 1e-4
    # Exact sample has SSIM 1; the expected SSIM is below 1 because two samples are offset.
    assert s["ssim"] < 1.0
    # Misfit stays in physical units (identity forward): mean(0, 300^2*64, 150^2*64).
    assert abs(s["misfit"] - (300.0**2 + 150.0**2) * 64 / 3) < 1.0


def test_aggregate_mean_std() -> None:
    v_true = torch.full((4, 4), 1500.0)  # -> -1 normalised
    pt_a = score_target(
        torch.stack([v_true + 300.0]), v_true, _identity(v_true), _identity
    )  # MAE 0.2
    pt_b = score_target(
        torch.stack([v_true + 900.0]), v_true, _identity(v_true), _identity
    )  # MAE 0.6
    stats = InversionStats.aggregate("m", [pt_a, pt_b], solves=[7, 7])

    assert stats.n_targets == 2
    assert stats.n_solves == 7.0
    assert abs(stats.agg["mae_mean"] - 0.4) < 1e-4  # mean(0.2, 0.6)
    assert abs(stats.agg["mae_std"] - 0.2) < 1e-4  # population std


def test_evaluator_loops_targets() -> None:
    class DummyModule:
        name = "dummy"

        def invert(self, d_obs: torch.Tensor) -> InversionResult:
            # Always returns 1500 m/s (-> -1 normalised): exact on target 0, off on target 1.
            return InversionResult(torch.full((2, 8, 8), 1500.0), n_solves=5)

    targets = [(0, torch.full((8, 8), 1500.0)), (1, torch.full((8, 8), 1800.0))]
    ev = Evaluator(targets, device=torch.device("cpu"), simulate_fn=_identity)
    stats = ev.evaluate(DummyModule())

    assert stats.module == "dummy"
    assert stats.n_targets == 2 and stats.n_solves == 5.0
    # Normalised MAE: 0 on target 0, |−1 − (−0.8)| = 0.2 on target 1 -> mean 0.1.
    assert abs(stats.agg["mae_mean"] - 0.1) < 1e-4
