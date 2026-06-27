"""Shared single-target inversion driver for the experiment entry points (flow tilting and
diffusion DPS). Each experiment supplies only how to *invert* (its prior + guidance scheme) via
an ``invert(d_obs, guidance_strength) -> samples`` callable; this module owns the parts they all
share: loading the held-out target, running guided + unguided passes, scoring (MAE/RMSE/misfit,
oracle vs ground-truth-free pick), and the ``true | v_hat | error`` figure.
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
from .bridge import held_out_targets, seismic_forward, to_mps_native

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
    """
    gidx, v_true, d_obs = load_target(dataset_cfg, target_index, device)
    print(f"target: val map global index {gidx} (native {tuple(v_true.shape)})")

    gs = guidance if method_name != "unguided" else 0.0
    guided = invert(d_obs, gs)
    unguided = guided if gs == 0.0 else invert(d_obs, 0.0)

    vg = to_mps_native(guided)
    n = vg.shape[0]
    mae = (vg - v_true).abs().mean(dim=(1, 2))
    rmse = ((vg - v_true) ** 2).mean(dim=(1, 2)).sqrt()
    dm_g = ((seismic_forward(guided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((seismic_forward(unguided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    oracle = int(mae.argmin())  # needs ground truth — not available in a real inversion
    reported = int(dm_g.argmin())  # ground-truth-free pick: lowest data misfit
    ratio = float(dm_g.mean() / dm_u.mean())

    print(f"method={method_name}  guidance={gs:g}  steps={steps}  n={n}")
    print(f"  MAE (m/s):  {[round(x) for x in mae.tolist()]}")
    print(
        f"  MAE mean={round(float(mae.mean()))}  median={round(float(mae.median()))}  worst={round(float(mae.max()))}"
    )
    print(
        f"  oracle (needs GT) best MAE = {round(float(mae[oracle]))}  [sample {oracle}]"
    )
    print(
        f"  reported (min data misfit) MAE = {round(float(mae[reported]))}  [sample {reported}]"
    )
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  unguided={float(dm_u.mean()):.3e}  ratio={ratio:.3f}"
    )

    _plot(v_true, vg[reported], float(mae[reported]), out_png)
    summary = {
        "inv/mae_mean": float(mae.mean()),
        "inv/mae_reported": float(mae[reported]),
        "inv/rmse_mean": float(rmse.mean()),
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
    ax[1].set_title(f"v_hat, min-misfit (MAE {round(mae)} m/s)")
    ax[1].axis("off")
    im = ax[2].imshow(vh - vt, cmap="RdBu", vmin=-500, vmax=500)
    ax[2].set_title("error")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
