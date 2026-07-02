"""How far is a *trapped* FWI model from the truth, and is there really a barrier
between them? (0004 example pair)

Companion to ``cycle_skipping_landscape.py`` / ``cycle_skipping_escape.py``. The landscape
scan showed the aggregate L2 misfit along the alpha-ray is unimodal-but-saturating, yet
plain-L2 FWI from a poor start still converges to a wrong model with a tiny misfit. This
script interrogates that trapped point directly:

  1. TRAP — run plain least-squares FWI from a cycle-skipped start
     (``START_ALPHA x smooth(v_true)``, as in the escape script) to obtain ``v_trapped``
     whose predicted data ``d_pred`` fits ``d_obs`` well. Report how different
     ``v_trapped`` is from ``v_true`` in m/s — the "how different are the velocity maps
     behind d_obs vs d_pred" question.

  2. BARRIER — scan the raw L2 misfit along the straight segment
     ``v(t) = (1-t) v_trapped + t v_true``. If the misfit rises before falling, the
     trapped point sits in a genuine separate basin (a barrier gradient descent cannot
     cross); if it falls monotonically, the trap is a flat/ill-conditioned direction
     rather than a wall. Also scan the envelope and 3 Hz low-passed misfits along the
     same segment — the relaxed metrics should remove the barrier.

  3. MIDPOINT — the "one velocity map that works for both d_obs and d_pred" test, now
     with a real trapped model: compare the data of the segment-midpoint velocity
     ``simulate((v_trapped + v_true)/2)`` against the L2 midpoint of the two datasets
     ``(d_obs + d_pred)/2`` on a far-offset trace. The former is one wavelet at an
     intermediate time; the latter is two half-amplitude wavelets no velocity model can
     produce — the geometric reason L2-ambiguous fitting cannot stabilise the inversion,
     while shift-aware ambiguity (envelope/OT/AWI) can.

Run from the repo root (uses the GPU if present):
    uv run python data_visualisation/cycle_skipping_barrier.py
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
ITERS = 500
LR = 2e-2
DT = 1e-3
FREQ = 15.0
TRACE_SRC, TRACE_REC = 0, 69
SEGMENT_TS = np.linspace(0.0, 1.0, 41)   # v(t) = (1-t) v_trapped + t v_true


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


def rel(pred: torch.Tensor, obs: torch.Tensor) -> float:
    return float(((pred - obs) ** 2).sum() / (obs**2).sum().clamp_min(1e-12))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gidx, v_true = held_out_targets(OpenFWIDatasetConfig(), TARGET_INDEX + 1)[TARGET_INDEX]
    v_true = v_true.to(dev)
    d_obs = simulate(v_true).detach()

    # --- 1. produce the trapped model (plain L2 from a cycle-skipped start) ----
    k = SMOOTH_KERNEL
    v_smooth = F.avg_pool2d(v_true[None, None], k, stride=1, padding=k // 2,
                            count_include_pad=False)[0, 0]
    x = to_norm((v_smooth * START_ALPHA).clamp(VMIN, VMAX)).clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=LR)
    for _ in range(ITERS):
        opt.zero_grad()
        pred = simulate(to_mps(x))
        loss = ((pred - d_obs) ** 2).sum() / (d_obs**2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
    v_trap = to_mps(x.detach())
    d_pred = simulate(v_trap).detach()

    dv = v_trap - v_true
    print(f"val map {gidx} · trapped after {ITERS} iters of plain L2:")
    print(f"  rel data misfit(d_pred, d_obs) = {rel(d_pred, d_obs):.2e}")
    print(f"  velocity error: mean |dv| = {dv.abs().mean():.0f} m/s "
          f"({(dv.abs() / v_true).mean() * 100:.1f}%), max |dv| = {dv.abs().max():.0f} m/s")

    # --- 2. misfit along the segment v_trapped -> v_true ----------------------
    d_obs_lp = lowpass(d_obs, 3.0)
    env_obs = envelope(d_obs)
    seg_l2, seg_lp, seg_env = [], [], []
    for t in SEGMENT_TS:
        with torch.no_grad():
            d_t = simulate((1.0 - float(t)) * v_trap + float(t) * v_true)
        seg_l2.append(rel(d_t, d_obs))
        seg_lp.append(rel(lowpass(d_t, 3.0), d_obs_lp))
        seg_env.append(rel(envelope(d_t), env_obs))
    barrier = max(seg_l2) > seg_l2[0] * 1.05
    print(f"  segment scan: misfit at trapped end {seg_l2[0]:.2e}, peak {max(seg_l2):.2e} "
          f"-> {'BARRIER (separate basin)' if barrier else 'no barrier (flat valley)'}")

    # --- 3. midpoint data test -------------------------------------------------
    with torch.no_grad():
        d_vmid = simulate(0.5 * (v_trap + v_true))

    _plot(v_true.cpu(), v_trap.cpu(), dv.cpu(), seg_l2, seg_lp, seg_env,
          d_obs.cpu().numpy(), d_pred.cpu().numpy(), d_vmid.cpu().numpy(), gidx)


def _plot(v_true, v_trap, dv, seg_l2, seg_lp, seg_env, d_obs, d_pred, d_vmid, gidx) -> None:
    fig = plt.figure(figsize=(12.5, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 1.0])

    for c, (title, arr, cmap, vmin, vmax) in enumerate([
        ("v_true", v_true, "jet", VMIN, VMAX),
        ("v_trapped (plain L2, tiny misfit)", v_trap, "jet", VMIN, VMAX),
        ("difference (m/s)", dv, "RdBu_r", -800, 800),
    ]):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(SEGMENT_TS, seg_l2, "k-", lw=1.8, label="raw L2")
    ax.plot(SEGMENT_TS, seg_lp, "b-", lw=1.4, label="low-passed 3 Hz")
    ax.plot(SEGMENT_TS, seg_env, "g-", lw=1.4, label="envelope")
    ax.set_yscale("log")
    ax.set_xlabel("t along segment:  v(t) = (1-t) v_trapped + t v_true")
    ax.set_ylabel("relative misfit to d_obs")
    ax.legend(fontsize=9)
    ax.set_title("misfit along the straight path from the trapped model to the truth",
                 fontsize=10)

    t_ax = np.arange(d_obs.shape[-1]) * DT
    win = slice(0, 900)
    obs = d_obs[TRACE_SRC, TRACE_REC]
    prd = d_pred[TRACE_SRC, TRACE_REC]
    vmid = d_vmid[TRACE_SRC, TRACE_REC]
    l2mid = 0.5 * (obs + prd)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(t_ax[win], obs[win], "k-", lw=1.5, label="d_obs (v_true)")
    ax.plot(t_ax[win], prd[win], "r-", lw=1.1, label="d_pred (v_trapped)")
    ax.plot(t_ax[win], l2mid[win], "m--", lw=1.2, label="(d_obs + d_pred)/2 — data-space midpoint")
    ax.plot(t_ax[win], vmid[win], "c-", lw=1.2, label="simulate((v_true + v_trapped)/2) — model-space midpoint")
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=8, ncols=2)
    ax.set_title(f'the "works for both" midpoint test on the far-offset trace '
                 f"(src {TRACE_SRC + 1}, rec {TRACE_REC + 1})", fontsize=10)

    fig.suptitle(f"anatomy of a trapped FWI model · val map {gidx}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cycle_skipping_barrier.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
