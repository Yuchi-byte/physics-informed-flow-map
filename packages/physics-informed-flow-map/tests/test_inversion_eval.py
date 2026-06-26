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
    x = torch.rand(16, 16) * 3000.0
    assert ssim(x, x) == 1.0 or abs(ssim(x, x) - 1.0) < 1e-4
    assert ssim(x, x + torch.randn_like(x) * 400.0) < 0.99


def test_score_target_picks_and_postmean() -> None:
    v_true = torch.zeros(8, 8)
    # Sample 0 is exact; others are offset constants.
    v_hat = torch.stack([v_true, v_true + 100.0, v_true + 50.0])
    s = score_target(v_hat, v_true, _identity(v_true), _identity)

    # Exact sample wins both the oracle (min-MAE) and GT-free (min-misfit) picks.
    assert s["mae_oracle"] == 0.0
    assert s["mae_gt_free"] == 0.0
    assert abs(s["ssim_oracle"] - 1.0) < 1e-4
    # Posterior mean = mean(0, 100, 50) = 50 everywhere -> MAE 50.
    assert abs(s["mae_postmean"] - 50.0) < 1e-4
    # GT-free pick's achieved misfit is the exact sample's -> 0.
    assert s["misfit_gt_free"] == 0.0


def test_aggregate_mean_std() -> None:
    v_true = torch.zeros(4, 4)
    pt_a = score_target(
        torch.stack([v_true + 10.0]), v_true, _identity(v_true), _identity
    )
    pt_b = score_target(
        torch.stack([v_true + 30.0]), v_true, _identity(v_true), _identity
    )
    stats = InversionStats.aggregate("m", [pt_a, pt_b], solves=[7, 7])

    assert stats.n_targets == 2
    assert stats.n_solves == 7.0
    assert abs(stats.agg["mae_oracle_mean"] - 20.0) < 1e-4  # mean(10, 30)
    assert abs(stats.agg["mae_oracle_std"] - 10.0) < 1e-4  # population std


def test_evaluator_loops_targets() -> None:
    class DummyModule:
        name = "dummy"

        def invert(self, d_obs: torch.Tensor) -> InversionResult:
            # Always returns zeros: perfect on the zero target, off-by-100 on the other.
            return InversionResult(torch.zeros(2, 8, 8), n_solves=5)

    targets = [(0, torch.zeros(8, 8)), (1, torch.full((8, 8), 100.0))]
    ev = Evaluator(targets, device=torch.device("cpu"), simulate_fn=_identity)
    stats = ev.evaluate(DummyModule())

    assert stats.module == "dummy"
    assert stats.n_targets == 2 and stats.n_solves == 5.0
    # MAE oracle: 0 on target 0, 100 on target 1 -> mean 50.
    assert abs(stats.agg["mae_oracle_mean"] - 50.0) < 1e-4
