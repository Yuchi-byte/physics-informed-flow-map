"""Can "ambiguous fitting" rescue FWI from a cycle-skipped start? (0004 example pair)

Companion to ``cycle_skipping_landscape.py``. The landscape scan shows that a background
velocity error of ~15-25% puts the initial model in a spurious L2 basin (arrivals one
cycle off). Here we start FWI from exactly such a model — a smoothed, down-scaled copy of
the truth, the realistic "poor starting model" failure mode — and run three inversions
with the same iteration budget:

  * L2         — plain least-squares data misfit (the classical baseline; should stay
                 trapped: it aligns wrong cycles and sharpens the wrong model).
  * MULTISCALE — frequency continuation: fit 3 Hz low-passed data, then 6 Hz, then full
                 band. At 3 Hz the period is so long that no arrival is more than half a
                 cycle off, so the basin is convex-ish; each stage hands the next a model
                 inside its basin. The textbook fix (Bunks et al. 1995).
  * ENVELOPE   — fit the Hilbert envelope of the data first (phase-blind: a trace one
                 cycle late has nearly the right envelope location, so the misfit sees
                 the *time* error, not a phase error), then switch to L2. This is the
                 direct implementation of "settle for a velocity map whose data is
                 compatible with both d_obs and the skipped d_pred": the envelope metric
                 makes those two datasets close, so the minimiser can travel between them.

All three see the same d_obs, same starting model, same optimiser. The figure compares
recovered maps, MAE to truth, and the far-offset trace before/after.

Run from the repo root (uses the GPU if present):
    uv run python FWI_problem_exploration/cycle_skipping_escape.py
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
START_ALPHA = 0.75          # scale of the smoothed start: inside a skipped basin
SMOOTH_KERNEL = 21          # box-blur size for the starting model (smooth background)
ITERS = 500                 # total gradient steps per method
LR = 2e-2                   # Adam step in normalised [-1, 1] space
DT = 1e-3
FREQ = 15.0
TRACE_SRC, TRACE_REC = 0, 69
STAGES_MULTISCALE = [(3.0, 150), (6.0, 150), (None, 200)]   # (low-pass Hz | None, iters)
STAGES_ENVELOPE = [("env", 250), ("l2", 250)]


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


def envelope(d: torch.Tensor) -> torch.Tensor:
    """Hilbert envelope along time (differentiable analytic-signal magnitude)."""
    nt = d.shape[-1]
    X = torch.fft.fft(d, dim=-1)
    h = torch.zeros(nt, device=d.device, dtype=X.dtype)
    h[0] = 1.0
    if nt % 2 == 0:
        h[nt // 2] = 1.0
        h[1:nt // 2] = 2.0
    else:
        h[1:(nt + 1) // 2] = 2.0
    analytic = torch.fft.ifft(X * h, dim=-1)
    return (analytic.real**2 + analytic.imag**2 + 1e-12).sqrt()


def rel(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return ((pred - obs) ** 2).sum() / (obs**2).sum().clamp_min(1e-12)


def run_fwi(d_obs: torch.Tensor, x0: torch.Tensor, stages) -> tuple[torch.Tensor, list]:
    """Staged FWI. Each stage is (transform_spec, iters); transform_spec is None (raw L2),
    a float (low-pass cutoff Hz), 'l2', or 'env'. Returns final model and per-iter history
    of the *raw* relative L2 misfit (comparable across methods)."""
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=LR)
    history = []
    for spec, iters in stages:
        for _ in range(iters):
            opt.zero_grad()
            pred = simulate(to_mps(x))
            if spec is None or spec == "l2":
                loss = rel(pred, d_obs)
            elif spec == "env":
                loss = rel(envelope(pred), envelope(d_obs))
            else:  # low-pass cutoff in Hz
                loss = rel(lowpass(pred, spec), lowpass(d_obs, spec))
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(-1.0, 1.0)
                history.append(float(rel(pred.detach(), d_obs)))  # pre-step raw L2
    return x.detach(), history


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gidx, v_true = held_out_targets(OpenFWIDatasetConfig(), TARGET_INDEX + 1)[TARGET_INDEX]
    v_true = v_true.to(dev)
    d_obs = simulate(v_true).detach()

    # cycle-skipped smooth start: blur the truth, scale it down
    k = SMOOTH_KERNEL
    v_smooth = F.avg_pool2d(v_true[None, None], k, stride=1, padding=k // 2,
                            count_include_pad=False)[0, 0]
    v0 = (v_smooth * START_ALPHA).clamp(VMIN, VMAX)
    x0 = to_norm(v0)
    v_true_norm = to_norm(v_true)
    mae0 = float((x0 - v_true_norm).abs().mean())
    print(f"val map {gidx} · start = {START_ALPHA} x smooth(v_true) · start MAE {mae0:.3f} "
          f"· start rel misfit {float(rel(simulate(v0).detach(), d_obs)):.2e}")

    methods = {
        "L2 (plain)": [(None, ITERS)],
        "multiscale 3->6->full Hz": STAGES_MULTISCALE,
        "envelope -> L2": STAGES_ENVELOPE,
    }
    results = {}
    for name, stages in methods.items():
        x_hat, hist = run_fwi(d_obs, x0, stages)
        mae = float((x_hat - v_true_norm).abs().mean())
        results[name] = (x_hat, hist, mae)
        print(f"  {name:28s} final rel misfit {hist[-1]:.2e}   norm MAE to true {mae:.3f}")

    _plot(v_true.cpu().numpy(), v0.cpu().numpy(), d_obs, results, mae0, gidx)


def _plot(v_true, v0, d_obs, results, mae0, gidx) -> None:
    names = list(results)
    ncols = 2 + len(names)
    fig = plt.figure(figsize=(3.0 * ncols, 8.2))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.4, 1.0, 1.0])

    # row 0: velocity maps — truth, start, recovered per method
    panels = [("TRUE", v_true, None), (f"start ({START_ALPHA}x smooth)", v0, mae0)]
    panels += [(n, to_mps_np(results[n][0].cpu().numpy()), results[n][2]) for n in names]
    for c, (title, vmap, mae) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(vmap, cmap="jet", vmin=VMIN, vmax=VMAX)
        sub = "" if mae is None else f"\nMAE {mae:.3f}"
        ax.set_title(title + sub, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # row 1: raw-L2 misfit history (log)
    ax = fig.add_subplot(gs[1, :])
    for n in names:
        ax.plot(results[n][1], lw=1.4, label=n)
    ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("raw rel L2 misfit")
    ax.legend(fontsize=9)
    ax.set_title("data misfit (always measured in raw L2, whatever the training misfit)",
                 fontsize=9)

    # row 2: far-offset trace, truth vs each recovered model
    t = np.arange(d_obs.shape[-1]) * DT
    obs_tr = d_obs[TRACE_SRC, TRACE_REC].cpu().numpy()
    ax = fig.add_subplot(gs[2, :])
    ax.plot(t, obs_tr, "k-", lw=1.6, label="d_obs")
    for n in names:
        with torch.no_grad():
            pred = simulate(to_mps(results[n][0]))
        ax.plot(t, pred[TRACE_SRC, TRACE_REC].cpu().numpy(), lw=1.0, label=n)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=8)
    ax.set_title(f"far-offset trace (src {TRACE_SRC + 1}, rec {TRACE_REC + 1}) after inversion",
                 fontsize=9)

    fig.suptitle(
        f"escaping a cycle-skipped start (val map {gidx}): plain L2 vs relaxed misfits, "
        "same budget & optimiser", fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cycle_skipping_escape.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


def to_mps_np(x_norm):
    return (x_norm + 1.0) / 2.0 * (VMAX - VMIN) + VMIN


if __name__ == "__main__":
    main()
