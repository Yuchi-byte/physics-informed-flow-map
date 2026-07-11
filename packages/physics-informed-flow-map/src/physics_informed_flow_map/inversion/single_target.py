"""Shared single-target inversion driver for the experiment entry points (flow tilting and
diffusion DPS). Each experiment supplies only how to *invert* (its prior + guidance scheme) via
an ``invert(d_obs, guidance_strength) -> samples`` callable; this module owns the parts they all
share: loading the held-out target, running guided + unguided passes, scoring (expected
MAE/RMSE/SSIM across samples on the OpenFWI ``[-1, 1]`` scale, plus the guided/unguided misfit
ratio), and the ``true | v_hat | error`` figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..physics.filters import highpass
from ..physics.misfit import MisfitFn
from ..physics.observation import Observation, ObservationConfig, observe
from .benchmark import InversionBenchmark
from .bridge import held_out_targets, mps_to_norm, seismic_forward, to_mps_native
from .evaluate import ssim

# An inverter: given observed seismic data and a guidance strength, return prior samples
# (B,1,res,res) in [-1,1]. guidance_strength=0 must yield an unguided prior sample.
Inverter = Callable[[Tensor, float], Tensor]


def load_target(
    dataset_cfg: OpenFWIDatasetConfig,
    target_index: int,
    device: torch.device,
    *,
    target: str | None = None,
    benchmark_root: Path | str = "data/inversion_bench",
    obs_cfg: ObservationConfig | None = None,
) -> tuple[int, str, Tensor, Observation]:
    """``(global_index, label, v_true native 70x70 m/s, observation)``.

    ``target`` (a benchmark id, e.g. ``style_a_03``) loads from the self-contained
    inversion benchmark — no bulk-data dependency. Otherwise ``target_index`` selects
    from the seed-0 validation split via the bulk dataset (legacy path). The label names
    the target for figures/captions; the global index goes to the run summary.

    ``obs_cfg`` selects the benchmark track (band limit / frozen noise / operator
    mismatch); ``None`` = the legacy clean inverse-crime observation. The noise key is
    the benchmark id (or ``val{gidx}``), so a target's realization is identical across
    runs and methods.
    """
    cfg = obs_cfg or ObservationConfig()
    if target is not None:
        bench = InversionBenchmark(benchmark_root)
        gidx = int(bench.entry(target)["global_index"])
        v_true = bench.velocity(target).to(device)
        return gidx, target, v_true, observe(v_true, cfg, key=target)
    gidx, native = held_out_targets(dataset_cfg, target_index + 1)[target_index]
    v_true = native.to(device)
    return gidx, f"val map {gidx}", v_true, observe(v_true, cfg, key=f"val{gidx}")


def invert_and_report(
    invert: Inverter,
    *,
    dataset_cfg: OpenFWIDatasetConfig,
    target_index: int,
    target: str | None = None,
    method_name: str,
    guidance: float,
    steps: int,
    device: torch.device,
    out_png: Path,
    out_obs_png: Path | None = None,
    cost: Callable[[], float] | None = None,
    misfit_factory: Callable[[Tensor], MisfitFn] | None = None,
    obs_cfg: ObservationConfig | None = None,
) -> tuple[dict[str, float], str]:
    """Run guided + unguided inversion on a held-out map, score it, and write the figure.

    Returns ``(summary_scalars, caption)`` — the caller logs the figure and the scalars to its
    own run. ``method_name == "unguided"`` forces guidance off (the no-physics control).
    Metrics are the expected MAE/RMSE/SSIM across samples on the OpenFWI ``[-1, 1]`` scale.

    If ``out_obs_png`` is given, also writes the observed seismic data ``d_obs`` — the input the
    velocity is inverted *from* — so the run folder shows what recovery was conditioned on.

    ``cost`` (called after the guided pass) supplies the total forward-solve count for the figure
    banner, so the inference cost is visible next to the quality metrics.

    ``misfit_factory`` (a non-L2 guidance misfit built from ``d_obs``, see ``physics.misfit``)
    only *adds* ``inv/guidance_misfit_{guided,unguided}`` to the summary — the scored metrics
    (MAE/RMSE/SSIM and the L2 misfit ratio) never change with the guidance misfit, so runs stay
    comparable across ``method.misfit`` settings.

    ``obs_cfg`` selects the benchmark track. Under a band limit the L2 misfit metrics are
    computed on high-passed predictions (the in-band operator F — a full-band comparison
    would carry an unfixable low-band residual); with noise on, ``inv/misfit_over_floor``
    reports the guided misfit against the known χ² noise floor (≈1 is perfect; <1 means
    fitting noise) and ``inv/sigma_true`` records the matched σ.
    """
    gidx, label, v_true, observation = load_target(
        dataset_cfg, target_index, device, target=target, obs_cfg=obs_cfg
    )
    d_obs = observation.d_obs
    min_freq = obs_cfg.min_freq_hz if obs_cfg is not None else 0.0
    print(f"target: {label} (global index {gidx}, native {tuple(v_true.shape)})")

    if out_obs_png is not None:
        _plot_seismic(d_obs, gidx, out_obs_png)

    gs = guidance if method_name != "unguided" else 0.0
    guided = invert(d_obs, gs)
    unguided = guided if gs == 0.0 else invert(d_obs, 0.0)

    vg = to_mps_native(guided)  # (n, 70, 70) m/s — for the figure
    vh, vt = mps_to_norm(vg), mps_to_norm(v_true)  # [-1, 1] for the metrics
    n = vg.shape[0]
    mae = (vh - vt).abs().mean(dim=(1, 2))  # (n,) per-sample, normalised
    rmse = ((vh - vt) ** 2).mean(dim=(1, 2)).sqrt()
    ssim_mean = sum(ssim(vh[i], vt) for i in range(n)) / n
    with torch.no_grad():
        pred_g_raw = seismic_forward(guided)
        pred_u_raw = pred_g_raw if gs == 0.0 else seismic_forward(unguided)
        # In-band comparison: the band limit is part of F (both sides); raw predictions
        # are kept for the guidance-misfit summary, whose factory applies its own filter.
        pred_g = highpass(pred_g_raw, min_freq, 1e-3) if min_freq > 0.0 else pred_g_raw
        pred_u = highpass(pred_u_raw, min_freq, 1e-3) if min_freq > 0.0 else pred_u_raw
    dm_g = ((pred_g - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((pred_u - d_obs) ** 2).sum(dim=(1, 2, 3))
    ratio = float(dm_g.mean() / dm_u.mean())

    print(f"method={method_name}  guidance={gs:g}  steps={steps}  n={n}")
    print(
        f"  MAE  mean={float(mae.mean()):.4f}  RMSE mean={float(rmse.mean()):.4f}  "
        f"SSIM mean={ssim_mean:.4f}  (normalised, E across samples)"
    )
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  unguided={float(dm_u.mean()):.3e}  ratio={ratio:.3f}"
    )

    n_solves = int(cost()) if cost is not None else None
    banner = (
        f"{method_name} · {label} · MAE {float(mae.mean()):.3f} · "
        f"RMSE {float(rmse.mean()):.3f} · SSIM {ssim_mean:.3f}"
    )
    if n_solves is not None:
        banner += f" · solves {n_solves}"
    _plot(
        v_true, vg[0], float(mae[0]), out_png, banner
    )  # sample 0: a representative draw
    summary = {
        "inv/mae_mean": float(mae.mean()),
        "inv/rmse_mean": float(rmse.mean()),
        "inv/ssim_mean": float(ssim_mean),
        "inv/misfit_ratio": ratio,
        "inv/target_index": gidx,
    }
    if observation.sigma is not None and observation.noise_floor is not None:
        summary["inv/sigma_true"] = observation.sigma
        summary["inv/misfit_over_floor"] = float(dm_g.mean()) / observation.noise_floor
    if misfit_factory is not None:
        with torch.no_grad():
            gm = misfit_factory(d_obs)  # applies its own band filter to raw predictions
            summary["inv/guidance_misfit_guided"] = float(gm(pred_g_raw).mean())
            summary["inv/guidance_misfit_unguided"] = float(gm(pred_u_raw).mean())
    return summary, f"{method_name} · {label}"


def _plot(
    v_true: Tensor, v_hat: Tensor, mae: float, out_png: Path, banner: str
) -> None:
    vt = v_true.cpu().numpy()
    vh = v_hat.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(vt, cmap="viridis")
    ax[0].set_title("true v")
    ax[0].axis("off")
    ax[1].imshow(vh, cmap="viridis", vmin=vt.min(), vmax=vt.max())
    ax[1].set_title(f"v_hat sample (norm. MAE {mae:.3f})")
    ax[1].axis("off")
    im = ax[2].imshow(vh - vt, cmap="RdBu", vmin=-500, vmax=500)
    ax[2].set_title("error (m/s)")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(
        banner, fontsize=10
    )  # quality metrics + total solves, all in one place
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_seismic(d_obs: Tensor, gidx: int, out_png: Path) -> None:
    """Shot gathers of the observed seismic ``d_obs`` (n_sources, n_receivers, nt) — the input the
    velocity is inverted from. One panel per source: time (down) x receiver, shared symmetric scale."""
    d = d_obs.detach().cpu().numpy()
    n_src = d.shape[0]
    vabs = float(np.percentile(np.abs(d), 99)) or 1.0
    fig, axes = plt.subplots(1, n_src, figsize=(2.2 * n_src, 3.6), squeeze=False)
    for s in range(n_src):
        ax = axes[0, s]
        # (n_receivers, nt) -> (nt, n_receivers): time on the vertical axis
        im = ax.imshow(d[s].T, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.set_title(f"source {s + 1}", fontsize=9)
        ax.set_xlabel("receiver", fontsize=8)
        ax.set_ylabel("time sample" if s == 0 else "", fontsize=8)
        if s > 0:
            ax.set_yticklabels([])
    fig.suptitle(f"observed seismic d_obs · val map {gidx}", fontsize=10)
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, label="amplitude")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
