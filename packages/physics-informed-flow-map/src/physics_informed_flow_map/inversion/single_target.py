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
import torch
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..physics.forward import simulate
from .bridge import held_out_targets, mps_to_norm, seismic_forward, to_mps_native
from .evaluate import ssim

# An inverter: given observed seismic data and a guidance strength, return prior samples
# (B,1,res,res) in [-1,1]. guidance_strength=0 must yield an unguided prior sample.
Inverter = Callable[[Tensor, float], Tensor]


def load_target(
    dataset_cfg: OpenFWIDatasetConfig, target_index: int, device: torch.device
) -> tuple[int, Tensor, Tensor]:
    """``(global_index, v_true native 70x70 m/s, observed seismic d_obs)`` for the
    ``target_index``-th seed-0 validation-split map (genuinely held out of training)."""
    gidx, native = held_out_targets(dataset_cfg, target_index + 1)[target_index]
    v_true = native.to(device)
    return gidx, v_true, simulate(v_true).detach()


def invert_and_report(
    invert: Inverter,
    *,
    dataset_cfg: OpenFWIDatasetConfig,
    target_index: int,
    method_name: str,
    guidance: float,
    steps: int,
    device: torch.device,
    out_png: Path,
) -> tuple[dict[str, float], str]:
    """Run guided + unguided inversion on a held-out map, score it, and write the figure.

    Returns ``(summary_scalars, caption)`` — the caller logs the figure and the scalars to its
    own run. ``method_name == "unguided"`` forces guidance off (the no-physics control).
    Metrics are the expected MAE/RMSE/SSIM across samples on the OpenFWI ``[-1, 1]`` scale.
    """
    gidx, v_true, d_obs = load_target(dataset_cfg, target_index, device)
    print(f"target: val map global index {gidx} (native {tuple(v_true.shape)})")

    gs = guidance if method_name != "unguided" else 0.0
    guided = invert(d_obs, gs)
    unguided = guided if gs == 0.0 else invert(d_obs, 0.0)

    vg = to_mps_native(guided)  # (n, 70, 70) m/s — for the figure
    vh, vt = mps_to_norm(vg), mps_to_norm(v_true)  # [-1, 1] for the metrics
    n = vg.shape[0]
    mae = (vh - vt).abs().mean(dim=(1, 2))  # (n,) per-sample, normalised
    rmse = ((vh - vt) ** 2).mean(dim=(1, 2)).sqrt()
    ssim_mean = sum(ssim(vh[i], vt) for i in range(n)) / n
    dm_g = ((seismic_forward(guided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((seismic_forward(unguided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    ratio = float(dm_g.mean() / dm_u.mean())

    print(f"method={method_name}  guidance={gs:g}  steps={steps}  n={n}")
    print(
        f"  MAE  mean={float(mae.mean()):.4f}  RMSE mean={float(rmse.mean()):.4f}  "
        f"SSIM mean={ssim_mean:.4f}  (normalised, E across samples)"
    )
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  unguided={float(dm_u.mean()):.3e}  ratio={ratio:.3f}"
    )

    _plot(v_true, vg[0], float(mae[0]), out_png)  # sample 0: a representative draw
    summary = {
        "inv/mae_mean": float(mae.mean()),
        "inv/rmse_mean": float(rmse.mean()),
        "inv/ssim_mean": float(ssim_mean),
        "inv/misfit_ratio": ratio,
        "inv/target_index": gidx,
    }
    return summary, f"{method_name} · val map {gidx}"


def _plot(v_true: Tensor, v_hat: Tensor, mae: float, out_png: Path) -> None:
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
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
