"""Best-practice classical FWI on the 0004 example pair: does regularisation close the
gap between the trapped model and the truth?

Follow-up to ``cycle_skipping_escape.py`` / ``cycle_skipping_barrier.py``, which showed
that from a 25%-slow smoothed start, *unregularised* FWI (plain L2, multiscale, envelope)
fits d_obs to ~1e-4 while remaining ~18% (500+ m/s) wrong — artifact-soup models drawn
from the null space of the surface-only acquisition. The question here: how much of that
gap was self-inflicted by doing FWI naively? We assemble the standard toolkit of a
"properly done" classical inversion and measure what remains:

  * multiscale frequency continuation (3 -> 6 -> full Hz), the cycle-skip fix;
  * total-variation (TV) regularisation with weight continuation (strong -> weak), the
    standard choice for blocky layered media — it kills the high-frequency artifact
    models that fit the data but look nothing like geology;
  * box constraints (velocity clamped to the physical range);
  * same optimiser, start, and iteration budget as the unregularised runs, so the only
    new ingredient is the regularisation.

Three runs are compared: plain L2 (the naive baseline), multiscale (no reg), and
multiscale + TV ("best practice"). The figure shows recovered maps, difference maps to
truth in m/s, depth profiles, and the MAE/misfit trade-off.

Run from the repo root (uses the GPU if present):
    uv run python FWI_problem_exploration/classical_fwi_regularised.py
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
TARGET_INDEX = 0
START_ALPHA = 0.75
SMOOTH_KERNEL = 21
LR = 2e-2
DT = 1e-3
# stages: (low-pass Hz | None for full band, TV weight, iters). Same 500-iter budget.
STAGES_PLAIN = [(None, 0.0, 500)]
STAGES_MULTI = [(3.0, 0.0, 150), (6.0, 0.0, 150), (None, 0.0, 200)]
STAGES_TV = [(3.0, 3e-3, 150), (6.0, 1e-3, 150), (None, 3e-4, 200)]


def to_mps(x_norm: torch.Tensor) -> torch.Tensor:
    return (x_norm + 1.0) / 2.0 * (VMAX - VMIN) + VMIN


def to_norm(v_mps: torch.Tensor) -> torch.Tensor:
    return (v_mps - VMIN) / (VMAX - VMIN) * 2.0 - 1.0


def lowpass(d: torch.Tensor, cutoff_hz: float) -> torch.Tensor:
    nt = d.shape[-1]
    f = torch.fft.rfftfreq(nt, DT).to(d.device)
    taper = torch.clamp((1.5 * cutoff_hz - f) / (0.5 * cutoff_hz), 0.0, 1.0)
    mask = 0.5 - 0.5 * torch.cos(np.pi * taper)
    return torch.fft.irfft(torch.fft.rfft(d, dim=-1) * mask, n=nt, dim=-1)


def tv(x: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation of the normalised map (mean over pixels)."""
    dy = x[1:, :] - x[:-1, :]
    dx = x[:, 1:] - x[:, :-1]
    return (dy[:, :-1] ** 2 + dx[:-1, :] ** 2 + 1e-8).sqrt().mean()


def rel(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return ((pred - obs) ** 2).sum() / (obs**2).sum().clamp_min(1e-12)


def run_fwi(d_obs: torch.Tensor, x0: torch.Tensor, stages):
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=LR)
    for cutoff, w_tv, iters in stages:
        d_ref = d_obs if cutoff is None else lowpass(d_obs, cutoff)
        for _ in range(iters):
            opt.zero_grad()
            pred = simulate(to_mps(x))
            if cutoff is not None:
                pred = lowpass(pred, cutoff)
            loss = rel(pred, d_ref) + w_tv * tv(x)
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(-1.0, 1.0)
    return x.detach()


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gidx, v_true = held_out_targets(OpenFWIDatasetConfig(), TARGET_INDEX + 1)[TARGET_INDEX]
    v_true = v_true.to(dev)
    d_obs = simulate(v_true).detach()

    k = SMOOTH_KERNEL
    v_smooth = F.avg_pool2d(v_true[None, None], k, stride=1, padding=k // 2,
                            count_include_pad=False)[0, 0]
    x0 = to_norm((v_smooth * START_ALPHA).clamp(VMIN, VMAX))
    v_true_norm = to_norm(v_true)

    methods = {
        "plain L2": STAGES_PLAIN,
        "multiscale (no reg)": STAGES_MULTI,
        "multiscale + TV": STAGES_TV,
    }
    results = {}
    print(f"val map {gidx} · start MAE {float((x0 - v_true_norm).abs().mean()):.3f}")
    for name, stages in methods.items():
        x_hat = run_fwi(d_obs, x0, stages)
        v_hat = to_mps(x_hat)
        mis = float(rel(simulate(v_hat).detach(), d_obs))
        mae_norm = float((x_hat - v_true_norm).abs().mean())
        mae_mps = float((v_hat - v_true).abs().mean())
        results[name] = (v_hat.cpu(), mis, mae_norm, mae_mps)
        print(f"  {name:22s} rel misfit {mis:.2e}   MAE {mae_norm:.3f} ({mae_mps:.0f} m/s)")

    _plot(v_true.cpu(), results, gidx)


def _plot(v_true, results, gidx) -> None:
    names = list(results)
    ncols = 1 + len(names)
    fig = plt.figure(figsize=(3.1 * ncols, 8.6))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.25, 1.25, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(v_true, cmap="jet", vmin=VMIN, vmax=VMAX)
    ax.set_title("TRUE", fontsize=10, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    fig.add_subplot(gs[1, 0]).axis("off")

    for c, n in enumerate(names, start=1):
        v_hat, mis, mae_norm, mae_mps = results[n]
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(v_hat, cmap="jet", vmin=VMIN, vmax=VMAX)
        ax.set_title(f"{n}\nmisfit {mis:.1e} · MAE {mae_norm:.3f} ({mae_mps:.0f} m/s)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[1, c])
        im = ax.imshow(v_hat - v_true, cmap="RdBu_r", vmin=-900, vmax=900)
        ax.set_title("difference to true (m/s)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    # depth profiles: laterally averaged velocity vs depth
    ax = fig.add_subplot(gs[2, :])
    depth = np.arange(v_true.shape[0]) * 10.0
    ax.plot(depth, v_true.numpy().mean(axis=1), "k-", lw=2.0, label="true")
    for n in names:
        ax.plot(depth, results[n][0].numpy().mean(axis=1), lw=1.3, label=n)
    ax.set_xlabel("depth (m)"); ax.set_ylabel("mean velocity (m/s)")
    ax.legend(fontsize=9)
    ax.set_title("laterally averaged velocity–depth profile", fontsize=10)

    fig.suptitle(
        f"does best-practice regularisation fix classical FWI? (val map {gidx}, "
        f"same {START_ALPHA}x smooth start & budget)", fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "classical_fwi_regularised.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
