"""Classical FWI on the 0004 example pair: least-squares gradient descent, no learned prior.

Uses the *same* held-out target as ``experiments/0004_inversion`` (seed-0 validation map,
target_index 0 -> global 6044). Two directions:

  * FORWARD  — given the true velocity map, solve the wave equation for the seismic data
               ``d_obs = simulate(v_true)``  (Deepwave PDE solve).
  * INVERSE  — given that ``d_obs``, recover velocity by minimising the least-squares data
               misfit ``||simulate(v) - d_obs||^2`` with Adam. No regularisation, no prior.

The inverse is run from several *different random initial models*. Because FWI is non-convex
(cycle skipping), the restarts settle in different minima: visibly different velocity maps that
all reproduce the same ``d_obs`` almost perfectly — the fundamental ill-posedness of the inverse
problem. This is exactly the failure a learned prior (experiments 0001/0002) is meant to fix.

Run from the repo root (uses the GPU if present):
    uv run python data_visualisation/classical_fwi_inversion.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.flow_matching.openfwi import VMAX, VMIN
from physics_informed_flow_map.inversion.bridge import held_out_targets
from physics_informed_flow_map.physics.forward import simulate

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_INDEX = 0        # same held-out map as experiments/0004_inversion
N_RESTARTS = 5          # independent random initial models
ITERS = 400             # gradient-descent steps per restart
LR = 2e-2               # Adam step in normalised [-1, 1] velocity space
SHOW_SOURCE = 2         # which of the 5 shots to display (0..4); 2 = centre
SEED = 0


def to_mps(x_norm: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> m/s (the forward operator's units)."""
    return (x_norm + 1.0) / 2.0 * (VMAX - VMIN) + VMIN


def to_norm(v_mps: torch.Tensor) -> torch.Tensor:
    """m/s -> [-1, 1] (the OpenFWI metric scale)."""
    return (v_mps - VMIN) / (VMAX - VMIN) * 2.0 - 1.0


def random_smooth_init(n: int, h: int, w: int, g: torch.Generator, device) -> torch.Tensor:
    """``n`` genuinely different starting models in [-1, 1]: a random constant background
    spanning the velocity range plus a smooth low-frequency perturbation (upsampled coarse
    noise). Deliberately structure-free so no restart is biased toward the true layering."""
    level = torch.empty(n, 1, 1, device=device).uniform_(-0.6, 0.6, generator=g)
    coarse = torch.randn(n, 1, 8, 8, generator=g, device=device)
    smooth = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)[:, 0]
    return (level + 0.4 * smooth).clamp(-1.0, 1.0)


def fwi(d_obs: torch.Tensor, x0: torch.Tensor, iters: int, lr: float):
    """Least-squares FWI from one initial model ``x0`` (normalised). Returns
    ``(recovered_norm (H,W), final_rel_misfit)``. Pure data misfit — no regulariser."""
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    denom = (d_obs**2).sum().clamp_min(1e-12)
    rel = float("nan")
    for _ in range(iters):
        opt.zero_grad()
        pred = simulate(to_mps(x))
        loss = ((pred - d_obs) ** 2).sum() / denom
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
        rel = float(loss.detach())
    return x.detach(), rel


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev).manual_seed(SEED)

    # --- the specific 0004 example pair --------------------------------------
    gidx, v_true_mps = held_out_targets(OpenFWIDatasetConfig(), TARGET_INDEX + 1)[TARGET_INDEX]
    v_true_mps = v_true_mps.to(dev)
    h, w = v_true_mps.shape
    d_obs = simulate(v_true_mps).detach()          # FORWARD: PDE solve, (5, 70, 1001)
    print(f"target val map {gidx}  velocity {tuple(v_true_mps.shape)}  d_obs {tuple(d_obs.shape)}")

    # --- FORWARD figure: true velocity -> predicted seismic ------------------
    _plot_forward(v_true_mps.cpu().numpy(), d_obs.cpu().numpy(), gidx)

    # --- INVERSE: several restarts from different random starts ---------------
    inits = random_smooth_init(N_RESTARTS, h, w, g, dev)
    recovered, misfits, maes = [], [], []
    v_true_norm = to_norm(v_true_mps)
    for r in range(N_RESTARTS):
        x_hat, rel = fwi(d_obs, inits[r], ITERS, LR)
        mae = float((x_hat - v_true_norm).abs().mean())   # normalised MAE to truth
        recovered.append(x_hat)
        misfits.append(rel)
        maes.append(mae)
        print(f"  restart {r}: rel data misfit {rel:.2e}   norm MAE to true {mae:.3f}")

    _plot_nonuniqueness(
        v_true_mps.cpu().numpy(), d_obs.cpu().numpy(),
        [i.cpu().numpy() for i in inits],
        [to_mps(x).cpu().numpy() for x in recovered],
        [simulate(to_mps(x)).detach().cpu().numpy() for x in recovered],
        misfits, maes, gidx,
    )


def _plot_forward(v_true, d_obs, gidx) -> None:
    n_src = d_obs.shape[0]
    fig, axes = plt.subplots(1, n_src + 1, figsize=(2.1 * (n_src + 1), 3.4))
    im0 = axes[0].imshow(v_true, cmap="jet", vmin=VMIN, vmax=VMAX)
    axes[0].set_title("true velocity (m/s)", fontsize=10)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("depth")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    vabs = np.percentile(np.abs(d_obs), 99)
    for s in range(n_src):
        ax = axes[s + 1]
        ax.imshow(d_obs[s].T, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        ax.set_title(f"d_obs src {s + 1}", fontsize=9)
        ax.set_xlabel("receiver")
        ax.set_ylabel("time" if s == 0 else "")
    fig.suptitle(f"FORWARD: true velocity -> predicted seismic (PDE solve) · val map {gidx}",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "classical_fwi_forward.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


def _plot_nonuniqueness(v_true, d_obs, inits, recovered, rec_dobs, misfits, maes, gidx) -> None:
    n = len(recovered)
    ncols = n + 1
    fig, axes = plt.subplots(3, ncols, figsize=(2.1 * ncols, 6.6))
    vabs = np.percentile(np.abs(d_obs), 99)

    def vel(ax, arr):
        return ax.imshow(arr, cmap="jet", vmin=VMIN, vmax=VMAX)

    def seis(ax, arr):
        return ax.imshow(arr[SHOW_SOURCE].T, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)

    # column 0 = the truth reference
    vel(axes[0, 0], v_true); axes[0, 0].set_title("TRUE", fontsize=10, weight="bold")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "(no init\nfor truth)", ha="center", va="center", fontsize=8)
    seis(axes[2, 0], d_obs)

    for r in range(n):
        c = r + 1
        vel(axes[0, c], to_mps_from_norm(inits[r]))
        axes[0, c].set_title(f"init {r + 1}", fontsize=9)
        vel(axes[1, c], recovered[r])
        axes[1, c].set_title(f"recovered {r + 1}\nmisfit {misfits[r]:.1e} · MAE {maes[r]:.2f}",
                             fontsize=8)
        seis(axes[2, c], rec_dobs[r])
        axes[2, c].set_title(f"its d_obs (src {SHOW_SOURCE + 1})", fontsize=8)

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("initial /\ntrue", fontsize=9)
    axes[1, 0].set_ylabel("recovered\nvelocity", fontsize=9)
    axes[2, 0].set_ylabel(f"predicted\nd_obs", fontsize=9)
    fig.suptitle(
        "INVERSE non-uniqueness: different random starts -> different velocity maps, "
        "all fitting the SAME d_obs\n"
        f"(least-squares FWI, no prior · val map {gidx}) — "
        "bottom row is near-identical while middle row differs",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "classical_fwi_nonuniqueness.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


def to_mps_from_norm(x_norm_np):
    return (x_norm_np + 1.0) / 2.0 * (VMAX - VMIN) + VMIN


if __name__ == "__main__":
    main()
