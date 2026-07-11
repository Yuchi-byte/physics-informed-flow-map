"""Score inversion modules against held-out velocity maps.

The scoring core (:func:`score_target`, :func:`ssim`, :meth:`InversionStats.aggregate`) is
pure — it takes tensors and a forward callable, so it is unit-testable without the wave
solver. :class:`Evaluator` wires in the real Deepwave operator and the held-out OpenFWI split
at the edges.

Two metric families, both on the ``[-1, 1]``-normalised velocity (OpenFWI scale):

* *Per-sample* (OpenFWI convention) — MAE/RMSE/SSIM as the *expected value across posterior
  samples* (``mean_i metric(x_i, x_true)``), i.e. the quality of a typical draw. Plus ``misfit``,
  the data-domain residual ``mean_i ||forward(x_i) - d_obs||^2``.
* *Distributional* — strictly-proper scores over the whole sample *set*: ``crps`` (per-pixel),
  ``energy`` (joint), and calibration ``cov50``/``cov90``/``cov_err``. These reward a calibrated
  posterior and penalise a confidently-wrong point estimate (zero spread), which the per-sample
  and posterior-mean metrics cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..physics.filters import highpass
from ..physics.forward import simulate
from ..physics.observation import Observation, ObservationConfig, observe
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


def crps_ensemble(samples: Tensor, truth: Tensor) -> float:
    """Mean per-pixel CRPS of the ensemble against the truth (lower is better).

    Empirical fair estimator (Gneiting & Raftery 2007):
    ``CRPS = (1/n)Σ_i|x_i-y| - (1/(2n(n-1)))Σ_{i,j}|x_i-x_j|``. A strictly proper score —
    rewards accuracy *and* calibrated spread, so a confidently-wrong point estimate (zero spread)
    is penalised. ``samples`` ``(n, H, W)``, ``truth`` ``(H, W)``, both on the ``[-1, 1]`` scale.
    """
    n = samples.shape[0]
    accuracy = (samples - truth).abs().mean(dim=0)  # (H, W) = (1/n)Σ|x_i-y|
    if n < 2:
        spread = torch.zeros_like(accuracy)
    else:
        pair = (
            (samples[:, None] - samples[None, :]).abs().sum(dim=(0, 1))
        )  # Σ_{i,j}|x_i-x_j|
        spread = pair / (2 * n * (n - 1))
    return float((accuracy - spread).mean())


def energy_score(samples: Tensor, truth: Tensor) -> float:
    """Per-pixel-RMS-normalised energy score (β=1) of the joint ensemble vs truth (lower better).

    Multivariate generalisation of CRPS over the whole map:
    ``ES = (1/n)Σ_i||x_i-y|| - (1/(2n(n-1)))Σ_{i,j}||x_i-x_j||``. Euclidean norms are divided by
    ``sqrt(D)`` (``D = H*W``) so the score stays on a per-pixel scale, comparable to MAE.
    """
    n = samples.shape[0]
    rms = samples[0].numel() ** 0.5
    s = samples.reshape(n, -1)
    y = truth.reshape(-1)
    accuracy = (s - y).norm(dim=1).mean() / rms  # mean_i ||x_i-y|| / sqrt(D)
    if n < 2:
        spread = accuracy.new_zeros(())
    else:
        pair = torch.cdist(s[None], s[None]).squeeze(0).sum()  # Σ_{i,j}||x_i-x_j||
        spread = pair / (2 * n * (n - 1)) / rms
    return float(accuracy - spread)


def coverage(
    samples: Tensor, truth: Tensor, levels: tuple[float, ...] = (0.5, 0.9)
) -> dict[str, float]:
    """Empirical central-credible-interval coverage per nominal level + mean calibration error.

    For each level ``a``, the per-pixel central ``a`` interval is taken from the ensemble quantiles;
    ``cov{a}`` is the fraction of pixels whose truth lands inside. ``cov_err`` is the mean
    ``|coverage - nominal|`` over levels (0 = perfectly calibrated; a zero-spread estimator under-
    covers badly). ``samples`` ``(n, H, W)``, ``truth`` ``(H, W)``.
    """
    out: dict[str, float] = {}
    errs: list[float] = []
    for a in levels:
        lo = torch.quantile(samples, (1 - a) / 2, dim=0)
        hi = torch.quantile(samples, (1 + a) / 2, dim=0)
        cov = float(((truth >= lo) & (truth <= hi)).float().mean())
        out[f"cov{int(a * 100)}"] = cov
        errs.append(abs(cov - a))
    out["cov_err"] = sum(errs) / len(errs)
    return out


def score_target(
    v_hat: Tensor,
    v_true: Tensor,
    d_obs: Tensor,
    forward_fn: ForwardFn,
    *,
    min_freq_hz: float = 0.0,
    noise_floor: float | None = None,
) -> dict[str, float]:
    """Expected metrics over one target's posterior samples against the truth.

    Args:
        v_hat: ``(n, H, W)`` velocity samples in m/s.
        v_true: ``(H, W)`` ground-truth velocity in m/s.
        d_obs: observed data for this target.
        forward_fn: ``(H, W) m/s -> data`` (for each sample's data misfit, in m/s).
        min_freq_hz: benchmark band limit — predictions are high-passed before the data
            misfit so the comparison is in-band (``d_obs`` arrives already filtered).
        noise_floor: expected ``||F(v_true) - d_obs||²`` under the matched-σ track; when
            given, adds ``misfit_over_floor`` (≈1 perfect; <1 = fitting noise).

    MAE/RMSE/SSIM are computed on the ``[-1, 1]``-normalised velocity (OpenFWI scale) and
    averaged over samples — the expected quality of a typical posterior draw. ``misfit`` is the
    mean data-domain residual over samples, kept in physical units (the forward operator's).
    """
    vh, vt = (
        mps_to_norm(v_hat),
        mps_to_norm(v_true),
    )  # [-1, 1] for the OpenFWI-scale metrics
    misfit = torch.stack(
        [
            ((highpass(forward_fn(v_hat[i]), min_freq_hz, 1e-3) - d_obs) ** 2).sum()
            for i in range(v_hat.shape[0])
        ]
    )
    ssim_vals = torch.tensor([ssim(vh[i], vt) for i in range(vh.shape[0])])
    scores = {
        "mae": float(_mae(vh, vt).mean()),
        "rmse": float(_rmse(vh, vt).mean()),
        "ssim": float(ssim_vals.mean()),
        "misfit": float(misfit.mean()),
        # Distributional scores over the posterior sample set (reward calibrated spread, not just
        # a typical sample). A near-deterministic method (e.g. DPS) should score poorly here.
        "crps": crps_ensemble(vh, vt),
        "energy": energy_score(vh, vt),
        **coverage(vh, vt),  # cov50, cov90, cov_err
    }
    if noise_floor is not None:
        scores["misfit_over_floor"] = float(misfit.mean()) / noise_floor
    return scores


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
        dist = (
            f"CRPS={self.agg['crps_mean']:.3g}  ES={self.agg['energy_mean']:.3g}  "
            f"cov50={self.agg['cov50_mean']:.2f} cov90={self.agg['cov90_mean']:.2f} "
            f"cov_err={self.agg['cov_err_mean']:.3g}"
        )
        return (
            f"[{self.module}]  n={self.n_targets}  solves/inv={self.n_solves:.0f}\n"
            f"  {cells}  misfit={self.agg['misfit_mean']:.3g}\n"
            f"  {dist}"
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
        obs_cfg: ObservationConfig | None = None,
    ) -> None:
        self.device = device
        self.simulate_fn = simulate_fn
        self.obs_cfg = obs_cfg or ObservationConfig()
        self.targets = [(gidx, v.to(device)) for gidx, v in targets]
        if simulate_fn is simulate:
            self.observations = [
                observe(v, self.obs_cfg, key=f"val{gidx}") for gidx, v in self.targets
            ]
        else:  # injected test operator: keep the legacy clean path (no hardening)
            self.observations = [
                Observation(simulate_fn(v).detach(), None, None)
                for _, v in self.targets
            ]
        self.d_obs = [o.d_obs for o in self.observations]

    @classmethod
    def from_openfwi(
        cls,
        families: list[str],
        n_targets: int,
        *,
        device: torch.device,
        resolution: int = 64,
        simulate_fn: ForwardFn = simulate,
        obs_cfg: ObservationConfig | None = None,
    ) -> "Evaluator":
        cfg = OpenFWIDatasetConfig(families=families, resolution=resolution)
        return cls(
            held_out_targets(cfg, n_targets),
            device=device,
            simulate_fn=simulate_fn,
            obs_cfg=obs_cfg,
        )

    def evaluate(self, module: InversionModule) -> InversionStats:
        per_target: list[dict[str, float]] = []
        solves: list[int] = []
        for (_, v_true), obs in zip(self.targets, self.observations):
            res = module.invert(obs.d_obs)
            solves.append(res.n_solves)
            per_target.append(
                score_target(
                    res.v_hat.to(self.device),
                    v_true,
                    obs.d_obs,
                    self.simulate_fn,
                    min_freq_hz=self.obs_cfg.min_freq_hz,
                    noise_floor=obs.noise_floor,
                )
            )
        return InversionStats.aggregate(module.name, per_target, solves)
