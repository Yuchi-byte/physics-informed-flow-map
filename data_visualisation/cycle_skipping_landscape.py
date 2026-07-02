"""Cycle skipping anatomy on the 0004 example pair: the L2 misfit landscape along a
1-parameter velocity family, and what the spurious minima *mean* in velocity space.

Uses the same held-out target as ``experiments/0004_inversion`` (seed-0 validation map,
target_index 0 -> global 6044). Three questions, three figures:

  1. LANDSCAPE — scan ``v(alpha) = alpha * v_true`` for alpha in [0.6, 1.4] and plot the
     relative L2 misfit ``||simulate(v(alpha)) - d_obs||^2 / ||d_obs||^2``. Cycle skipping
     shows up as local minima away from alpha=1. Repeating the scan on low-passed data
     (3 Hz / 6 Hz) shows the classic multiscale story: the spurious minima disappear when
     the period is long enough that no arrival is shifted by more than half a cycle.

  2. WHAT A SKIPPED MINIMUM LOOKS LIKE — at each spurious local minimum alpha*, overlay a
     far-offset trace of ``d_pred(alpha*)`` on ``d_obs``: the wavelets are one cycle apart
     yet the L2 misfit is locally minimal. Quantify the corresponding velocity error
     (|1-alpha*| of the whole map, i.e. hundreds of m/s) — the "how different are the
     velocity maps" question.

  3. THE MIDPOINT TEST — is there a velocity map that "works for both" d_obs and the
     cycle-skipped d_pred? Compare (a) the L2 midpoint of the two *datasets* (two half-
     amplitude wavelets — not achievable by any velocity model) against (b) the data of
     the midpoint *velocity* model (one wavelet at intermediate time). The forward-map
     image {simulate(v)} is a curved manifold: L2 interpolation in data space leaves the
     manifold, which is exactly why L2 misfit has spurious minima and why a phase/shift-
     aware metric (OT, AWI, envelope) is the principled version of "ambiguous fitting".

Run from the repo root (uses the GPU if present):
    uv run python data_visualisation/cycle_skipping_landscape.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.inversion.bridge import held_out_targets
from physics_informed_flow_map.physics.forward import simulate

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_INDEX = 0
ALPHAS = np.linspace(0.6, 1.4, 81)      # velocity scaling factors
LOWPASS_HZ = [3.0, 6.0]                 # multiscale variants of the misfit
DT = 1e-3                               # forward operator time step (s)
FREQ = 15.0                             # Ricker centre frequency (Hz)
TRACE_SRC, TRACE_REC = 0, 69            # far-offset trace: corner shot, last receiver


def lowpass(d: torch.Tensor, cutoff_hz: float, dt: float) -> torch.Tensor:
    """Zero-phase low-pass along the time axis (last dim) with a cosine roll-off."""
    nt = d.shape[-1]
    f = torch.fft.rfftfreq(nt, dt).to(d.device)
    # flat to cutoff, cosine taper to 1.5*cutoff, zero above
    taper = torch.clamp((1.5 * cutoff_hz - f) / (0.5 * cutoff_hz), 0.0, 1.0)
    mask = 0.5 - 0.5 * torch.cos(np.pi * taper)
    return torch.fft.irfft(torch.fft.rfft(d, dim=-1) * mask, n=nt, dim=-1)


def rel_misfit(pred: torch.Tensor, obs: torch.Tensor) -> float:
    return float(((pred - obs) ** 2).sum() / (obs**2).sum().clamp_min(1e-12))


def local_minima(y: np.ndarray) -> list[int]:
    return [i for i in range(1, len(y) - 1) if y[i] < y[i - 1] and y[i] < y[i + 1]]


def xcorr_lag_ms(a: np.ndarray, b: np.ndarray, dt: float, max_lag: int = 150) -> float:
    """Lag (ms) that best aligns b to a, by cross-correlation over +-max_lag samples."""
    lags = np.arange(-max_lag, max_lag + 1)
    cc = [np.dot(a, np.roll(b, k)) for k in lags]
    return float(lags[int(np.argmax(cc))] * dt * 1e3)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gidx, v_true = held_out_targets(OpenFWIDatasetConfig(), TARGET_INDEX + 1)[TARGET_INDEX]
    v_true = v_true.to(dev)
    d_obs = simulate(v_true).detach()
    print(f"target val map {gidx}  velocity {tuple(v_true.shape)}  d_obs {tuple(d_obs.shape)}")

    d_obs_lp = {hz: lowpass(d_obs, hz, DT) for hz in LOWPASS_HZ}

    # --- 1. misfit landscape over alpha ---------------------------------------
    preds: list[torch.Tensor] = []
    mis_full, mis_lp = [], {hz: [] for hz in LOWPASS_HZ}
    for a in ALPHAS:
        with torch.no_grad():
            pred = simulate(v_true * float(a))
        preds.append(pred)
        mis_full.append(rel_misfit(pred, d_obs))
        for hz in LOWPASS_HZ:
            mis_lp[hz].append(rel_misfit(lowpass(pred, hz, DT), d_obs_lp[hz]))
    mis_full = np.array(mis_full)

    minima = local_minima(mis_full)
    spurious = [i for i in minima if abs(ALPHAS[i] - 1.0) > 0.02]
    print("local minima at alpha =", [f"{ALPHAS[i]:.3f}" for i in minima])

    # arrival-time lag of the far-offset trace vs alpha (the cycle-skip mechanism)
    obs_tr = d_obs[TRACE_SRC, TRACE_REC].cpu().numpy()
    lags = [xcorr_lag_ms(obs_tr, p[TRACE_SRC, TRACE_REC].cpu().numpy(), DT) for p in preds]

    # per-trace misfits: the aggregate curve averages traces that skip at different
    # alpha, hiding the oscillation — single traces should show the classic wells
    trace_recs = [17, 40, 69]
    mis_trace = {}
    for r in trace_recs:
        o = d_obs[TRACE_SRC, r]
        mis_trace[r] = np.array([rel_misfit(p[TRACE_SRC, r], o) for p in preds])

    _plot_landscape(mis_full, mis_lp, mis_trace, lags, minima, gidx)

    # --- 2 & 3. per-spurious-minimum anatomy + the midpoint test ---------------
    for i in spurious:
        a_star = float(ALPHAS[i])
        v_err_mps = float((v_true * abs(1.0 - a_star)).mean())
        print(f"spurious minimum alpha*={a_star:.3f}: mean |dv| = {v_err_mps:.0f} m/s "
              f"({abs(1 - a_star) * 100:.0f}% of v_true), "
              f"far-trace lag {lags[i]:+.0f} ms (period {1e3 / FREQ:.0f} ms)")
        a_mid = 0.5 * (1.0 + a_star)
        with torch.no_grad():
            d_vmid = simulate(v_true * a_mid)
        _plot_minimum_and_midpoint(
            d_obs.cpu().numpy(), preds[i].cpu().numpy(), d_vmid.cpu().numpy(),
            v_true.cpu().numpy(), a_star, a_mid, lags[i], mis_full[i], gidx,
        )


def _plot_landscape(mis_full, mis_lp, mis_trace, lags, minima, gidx) -> None:
    fig, (ax1, ax_tr, ax2) = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 1.4, 1]})
    ax1.plot(ALPHAS, mis_full, "k-", lw=1.8, label="full band (15 Hz Ricker)")
    for hz, m in mis_lp.items():
        ax1.plot(ALPHAS, m, lw=1.4, label=f"low-passed {hz:.0f} Hz")
    for i in minima:
        ax1.plot(ALPHAS[i], mis_full[i], "rv", ms=8)
    ax1.axvline(1.0, color="g", ls=":", lw=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("relative L2 misfit")
    ax1.legend(fontsize=9)
    ax1.set_title(
        f"L2 misfit landscape along v = alpha * v_true  (val map {gidx})\n"
        "red markers: local minima — spurious ones vanish at low frequency",
        fontsize=11,
    )
    for r, m in mis_trace.items():
        ax_tr.plot(ALPHAS, m, lw=1.3, label=f"receiver {r + 1} only (src {TRACE_SRC + 1})")
        for i in local_minima(m):
            ax_tr.plot(ALPHAS[i], m[i], "rv", ms=6)
    ax_tr.set_yscale("log")
    ax_tr.set_ylabel("single-trace rel L2 misfit")
    ax_tr.legend(fontsize=8)
    ax_tr.set_title("single traces show the classic cycle-skip wells the aggregate "
                    "averages away", fontsize=9)

    period_ms = 1e3 / FREQ
    ax2.plot(ALPHAS, lags, "b-", lw=1.5)
    for k in (-1, 1):
        ax2.axhline(k * period_ms, color="r", ls="--", lw=0.8)
        ax2.axhline(k * period_ms / 2, color="orange", ls=":", lw=0.8)
    ax2.axhline(0, color="g", ls=":", lw=1)
    ax2.set_xlabel("alpha (velocity scale)")
    ax2.set_ylabel("far-trace lag (ms)")
    ax2.set_title("arrival-time shift of far-offset trace: skips lock in near +-1 period "
                  "(dashed red); half-period (dotted orange) is the basin edge", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "cycle_skipping_landscape.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


def _plot_minimum_and_midpoint(d_obs, d_star, d_vmid, v_true, a_star, a_mid, lag_ms,
                               misfit, gidx) -> None:
    t = np.arange(d_obs.shape[-1]) * DT
    obs = d_obs[TRACE_SRC, TRACE_REC]
    star = d_star[TRACE_SRC, TRACE_REC]
    vmid = d_vmid[TRACE_SRC, TRACE_REC]
    l2mid = 0.5 * (obs + star)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    win = slice(int(0.0 / DT), int(0.9 / DT))

    ax = axes[0]
    ax.plot(t[win], obs[win], "k-", lw=1.4, label="d_obs  (v_true)")
    ax.plot(t[win], star[win], "r-", lw=1.2,
            label=f"d_pred at spurious minimum (alpha*={a_star:.2f}, lag {lag_ms:+.0f} ms)")
    ax.set_title(
        f"cycle-skipped minimum: wavelets ~1 period apart, rel misfit {misfit:.2e}\n"
        f"velocity maps differ by {abs(1 - a_star) * 100:.0f}% everywhere "
        f"(mean |dv| = {np.abs(v_true * (1 - a_star)).mean():.0f} m/s)",
        fontsize=10,
    )
    ax.legend(fontsize=9); ax.set_ylabel("amplitude")

    ax = axes[1]
    ax.plot(t[win], l2mid[win], "m-", lw=1.4,
            label="L2 midpoint of the two DATASETS: (d_obs + d_pred)/2 — two wavelets")
    ax.plot(t[win], vmid[win], "c-", lw=1.4,
            label=f"data of the midpoint VELOCITY model (alpha={a_mid:.2f}) — one wavelet")
    ax.set_title(
        'the "works for both" test: no velocity model produces the L2 data midpoint —\n'
        "the forward-map image is curved, so L2-ambiguous fitting has no velocity meaning; "
        "a time-shift midpoint does",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")

    fig.suptitle(f"far-offset trace (src {TRACE_SRC + 1}, rec {TRACE_REC + 1}) · val map {gidx}",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"cycle_skipping_minimum_a{a_star:.2f}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
