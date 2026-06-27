"""Score inversion modules against held-out velocity maps.

The scoring core (:func:`score_target`, :func:`ssim`, :meth:`InversionStats.aggregate`) is
pure — it takes tensors and a forward callable, so it is unit-testable without the wave
solver. :class:`Evaluator` wires in the real Deepwave operator and the held-out OpenFWI split
at the edges. Metrics follow the OpenFWI convention — MAE/RMSE/SSIM on the velocity map
normalised to ``[-1, 1]`` — and report the *expected value across posterior samples*
(``mean_i metric(x_i, x_true)``), i.e. the quality of a typical sample rather than of a
single selected one or of the blurry per-pixel sample mean. ``misfit`` is the lone
data-domain number (mean over samples of ``||forward(x_i) - d_obs||^2``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..physics.forward import simulate
from .base import InversionModule
from .bridge import held_out_targets, mps_to_norm

DATA_RANGE = 2.0  # span of the [-1, 1] normalised velocity, for SSIM constants

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
    """Expected metrics over one target's posterior samples against the truth.

    Args:
        v_hat: ``(n, H, W)`` velocity samples in m/s.
        v_true: ``(H, W)`` ground-truth velocity in m/s.
        d_obs: observed data for this target.
        forward_fn: ``(H, W) m/s -> data`` (for each sample's data misfit, in m/s).

    MAE/RMSE/SSIM are computed on the ``[-1, 1]``-normalised velocity (OpenFWI scale) and
    averaged over samples — the expected quality of a typical posterior draw. ``misfit`` is the
    mean data-domain residual over samples, kept in physical units (the forward operator's).
    """
    vh, vt = (
        mps_to_norm(v_hat),
        mps_to_norm(v_true),
    )  # [-1, 1] for the OpenFWI-scale metrics
    misfit = torch.stack(
        [((forward_fn(v_hat[i]) - d_obs) ** 2).sum() for i in range(v_hat.shape[0])]
    )
    ssim_vals = torch.tensor([ssim(vh[i], vt) for i in range(vh.shape[0])])
    return {
        "mae": float(_mae(vh, vt).mean()),
        "rmse": float(_rmse(vh, vt).mean()),
        "ssim": float(ssim_vals.mean()),
        "misfit": float(misfit.mean()),
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
        cells = "  ".join(
            f"{metric.upper()}={self.agg[f'{metric}_mean']:.3g}±{self.agg[f'{metric}_std']:.2g}"
            for metric in ("mae", "rmse", "ssim")
        )
        return (
            f"[{self.module}]  n={self.n_targets}  solves/inv={self.n_solves:.0f}\n"
            f"  {cells}  misfit={self.agg['misfit_mean']:.3g}"
        )


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
