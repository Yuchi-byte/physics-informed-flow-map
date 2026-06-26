"""Score inversion modules against held-out velocity maps.

The scoring core (:func:`score_target`, :func:`ssim`, :meth:`InversionStats.aggregate`) is
pure — it takes tensors and a forward callable, so it is unit-testable without the wave
solver. :class:`Evaluator` wires in the real Deepwave operator and the held-out OpenFWI split
at the edges. Because the ground-truth-free pick (lowest data misfit) is unreliable, every
metric is reported under three selection rules: ``oracle`` (lowest MAE — needs the truth),
``gt_free`` (lowest data misfit — what a real inversion must use), and ``posterior_mean``
(the per-pixel mean of the samples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..flow_matching.openfwi import VMAX, VMIN
from ..physics.forward import simulate
from .base import InversionModule
from .bridge import held_out_targets

DATA_RANGE = VMAX - VMIN  # velocity span (m/s) for SSIM constants

ForwardFn = Callable[[Tensor], Tensor]  # (H, W) m/s -> seismic data


def _gaussian_window(size: int, sigma: float, device: torch.device) -> Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)[None, None]


def ssim(a: Tensor, b: Tensor, *, data_range: float = DATA_RANGE) -> float:
    """Single-scale Gaussian-windowed SSIM between two ``(H, W)`` maps (1.0 = identical)."""
    win = _gaussian_window(11, 1.5, a.device)
    a4, b4 = a[None, None].float(), b[None, None].float()
    pad = win.shape[-1] // 2
    mu_a, mu_b = F.conv2d(a4, win, padding=pad), F.conv2d(b4, win, padding=pad)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = F.conv2d(a4 * a4, win, padding=pad) - mu_a2
    var_b = F.conv2d(b4 * b4, win, padding=pad) - mu_b2
    cov = F.conv2d(a4 * b4, win, padding=pad) - mu_ab
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    smap = ((2 * mu_ab + c1) * (2 * cov + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return float(smap.mean())


def _mae(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).abs().mean(dim=(-2, -1))


def _rmse(a: Tensor, b: Tensor) -> Tensor:
    return ((a - b) ** 2).mean(dim=(-2, -1)).sqrt()


def score_target(
    v_hat: Tensor,
    v_true: Tensor,
    d_obs: Tensor,
    forward_fn: ForwardFn,
) -> dict[str, float]:
    """Metrics for one target's posterior samples against the truth.

    Args:
        v_hat: ``(n, H, W)`` velocity samples in m/s.
        v_true: ``(H, W)`` ground-truth velocity in m/s.
        d_obs: observed data for this target.
        forward_fn: ``(H, W) m/s -> data`` (for the data-misfit of each sample).

    Returns a flat dict of MAE/RMSE/SSIM under the three selection rules, plus the achieved
    data misfit of the GT-free pick / posterior mean / per-sample mean.
    """
    mae = _mae(v_hat, v_true)  # (n,)
    rmse = _rmse(v_hat, v_true)
    misfit = torch.stack(
        [((forward_fn(v_hat[i]) - d_obs) ** 2).sum() for i in range(v_hat.shape[0])]
    )
    oracle = int(mae.argmin())
    gt_free = int(misfit.argmin())
    v_mean = v_hat.mean(dim=0)
    return {
        "mae_oracle": float(mae[oracle]),
        "mae_gt_free": float(mae[gt_free]),
        "mae_postmean": float(_mae(v_mean, v_true)),
        "rmse_oracle": float(rmse[oracle]),
        "rmse_gt_free": float(rmse[gt_free]),
        "rmse_postmean": float(_rmse(v_mean, v_true)),
        "ssim_oracle": ssim(v_hat[oracle], v_true),
        "ssim_gt_free": ssim(v_hat[gt_free], v_true),
        "ssim_postmean": ssim(v_mean, v_true),
        "misfit_gt_free": float(misfit[gt_free]),
        "misfit_postmean": float(((forward_fn(v_mean) - d_obs) ** 2).sum()),
        "misfit_mean": float(misfit.mean()),
    }


@dataclass
class InversionStats:
    """Aggregated scores for one module over the evaluation set."""

    module: str
    n_targets: int
    n_solves: float  # mean forward PDE solves per inversion
    per_target: list[dict[str, float]]
    agg: dict[str, float]  # "<metric>_mean" / "<metric>_std" across targets

    @classmethod
    def aggregate(
        cls, module: str, per_target: list[dict[str, float]], solves: list[int]
    ) -> "InversionStats":
        agg: dict[str, float] = {}
        for key in per_target[0]:
            vals = torch.tensor([pt[key] for pt in per_target])
            agg[f"{key}_mean"] = float(vals.mean())
            agg[f"{key}_std"] = float(vals.std(unbiased=False))
        mean_solves = sum(solves) / max(len(solves), 1)
        return cls(module, len(per_target), mean_solves, per_target, agg)

    def summary(self) -> dict[str, float]:
        """Flat ``module``-prefixed aggregates (handy for a DataFrame row)."""
        return {"n_targets": self.n_targets, "n_solves": self.n_solves, **self.agg}

    def __str__(self) -> str:
        rows = [f"[{self.module}]  n={self.n_targets}  solves/inv={self.n_solves:.0f}"]
        for metric in ("mae", "rmse", "ssim"):
            cells = "  ".join(
                f"{rule}={self.agg[f'{metric}_{rule}_mean']:.3g}±{self.agg[f'{metric}_{rule}_std']:.2g}"
                for rule in ("oracle", "gt_free", "postmean")
            )
            rows.append(f"  {metric:5s} {cells}")
        rows.append(f"  misfit gt_free={self.agg['misfit_gt_free_mean']:.3g}")
        return "\n".join(rows)


class Evaluator:
    """Scores :class:`InversionModule`s on a fixed set of held-out velocity maps.

    Owns the held-out targets and the forward operator: builds each target's observed data
    once, then runs any module and scores its samples. Inject ``simulate_fn`` (default the
    Deepwave operator) to test without the solver.
    """

    def __init__(
        self,
        targets: list[tuple[int, Tensor]],
        *,
        device: torch.device,
        simulate_fn: ForwardFn = simulate,
    ) -> None:
        self.device = device
        self.simulate_fn = simulate_fn
        self.targets = [(gidx, v.to(device)) for gidx, v in targets]
        self.d_obs = [simulate_fn(v).detach() for _, v in self.targets]

    @classmethod
    def from_openfwi(
        cls,
        families: list[str],
        n_targets: int,
        *,
        device: torch.device,
        resolution: int = 64,
        simulate_fn: ForwardFn = simulate,
    ) -> "Evaluator":
        cfg = OpenFWIDatasetConfig(families=families, resolution=resolution)
        return cls(
            held_out_targets(cfg, n_targets), device=device, simulate_fn=simulate_fn
        )

    def evaluate(self, module: InversionModule) -> InversionStats:
        per_target: list[dict[str, float]] = []
        solves: list[int] = []
        for (_, v_true), d_obs in zip(self.targets, self.d_obs):
            res = module.invert(d_obs)
            solves.append(res.n_solves)
            per_target.append(
                score_target(res.v_hat.to(self.device), v_true, d_obs, self.simulate_fn)
            )
        return InversionStats.aggregate(module.name, per_target, solves)
