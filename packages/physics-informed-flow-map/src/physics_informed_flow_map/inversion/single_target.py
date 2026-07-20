"""Shared single-target inversion driver for the experiment entry points (flow tilting and
diffusion DPS). Each experiment supplies only how to *invert* (its prior + guidance scheme) via
an ``invert(d_obs, guidance_strength) -> samples`` callable; this module owns the parts they all
share: loading the held-out target, running guided + unguided passes, scoring (expected
MAE/RMSE/SSIM across samples on the OpenFWI ``[-1, 1]`` scale, plus the guided/unguided misfit
ratio), and the ``true | v_hat | error`` figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.image import AxesImage
import torch
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
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
    benchmark_root: Path
    | str = "/home/markhaoxiang/Projects/physics-informed-flow-map/data/inversion_bench",
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
    out_npy: Path | None = None,
    out_dobs_cmp_png: Path | None = None,
    out_dobs_cmp_npz: Path | None = None,
    cmp_label: str | None = None,
    cost: Callable[[], float] | None = None,
    misfit_factory: Callable[[Tensor], MisfitFn] | None = None,
    obs_cfg: ObservationConfig | None = None,
    traj_capture: dict[str, Any] | None = None,
    dobs_scales: tuple[str, ...] = ("linear",),
    forward_fn: Callable[[Tensor], Tensor] = seismic_forward,
) -> tuple[dict[str, float], str]:
    """Run guided + unguided inversion on a held-out map, score it, and write the figure.

    Returns ``(summary_scalars, caption)`` — the caller logs the figure and the scalars to its
    own run. ``method_name == "unguided"`` forces guidance off (the no-physics control).
    Metrics are the expected MAE/RMSE/SSIM across samples on the OpenFWI ``[-1, 1]`` scale.

    If ``out_obs_png`` is given, also writes the observed seismic data ``d_obs`` — the input the
    velocity is inverted *from* — so the run folder shows what recovery was conditioned on.

    If ``out_dobs_cmp_png`` is given, writes the data-space fit figure (observed vs re-simulated
    seismic for sample 0, plus residual); ``out_dobs_cmp_npz`` persists those two arrays and
    ``cmp_label`` prefixes the figure title (e.g. ``flow-matching + Tweedie + OT``).

    ``cost`` (called after the guided pass) supplies the total forward-solve count for the figure
    banner, so the inference cost is visible next to the quality metrics.

    ``misfit_factory`` (a non-L2 guidance misfit built from ``d_obs``, see ``physics.misfit``)
    only *adds* ``inv/guidance_misfit_{guided,unguided}`` to the summary — the scored metrics
    (MAE/RMSE/SSIM and the L2 misfit ratio) never change with the guidance misfit, so runs stay
    comparable across ``method.misfit`` settings.
    """
    gidx, label, v_true, observation = load_target(
        dataset_cfg, target_index, device, target=target, obs_cfg=obs_cfg
    )
    d_obs = observation.d_obs
    print(f"target: {label} (global index {gidx}, native {tuple(v_true.shape)})")
    if out_obs_png is not None:
        base = out_obs_png.with_suffix("")  # accept "d_obs" or "d_obs.png"
        for sc in dobs_scales:
            _plot_seismic(
                d_obs, gidx, base.with_name(f"{base.name}_{sc}.png"), scale=sc
            )

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
        pred_g = seismic_forward(guided)
        pred_u = pred_g if gs == 0.0 else seismic_forward(unguided)
        # In-band comparison: the band limit is part of F (both sides); raw predictions
        # are kept for the guidance-misfit summary, whose factory applies its own filter.
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
    if (
        out_npy is not None
    ):  # persist every posterior draw, not just the plotted sample 0
        np.savez(
            out_npy,
            recon_mps=vg.detach()
            .cpu()
            .numpy(),  # (n, 70, 70) m/s — the n final inversions
            v_true_mps=v_true.cpu().numpy(),  # (70, 70) m/s ground truth
            mae=mae.detach().cpu().numpy(),  # (n,) per-sample normalised MAE
            rmse=rmse.detach().cpu().numpy(),  # (n,) per-sample normalised RMSE
            target=label,
        )
    if (
        out_dobs_cmp_png is not None
    ):  # data-space fit: observed vs re-simulated seismic (sample 0)
        d_inv0 = pred_g[
            0
        ]  # in-band prediction, matches how d_obs is compared for the misfit
        plot_dobs_compare(
            d_obs,
            d_inv0,
            out_dobs_cmp_png,
            title=f"{cmp_label or method_name} · {label} · sample 0",
        )
        plot_dobs_spectrum_compare(  # f-k spectral twin, sibling _fk PNG
            d_obs,
            d_inv0,
            out_dobs_cmp_png.with_name(
                f"{out_dobs_cmp_png.stem}_fk{out_dobs_cmp_png.suffix}"
            ),
            title=f"{cmp_label or method_name} · {label} · sample 0",
        )
        if out_dobs_cmp_npz is not None:
            np.savez(
                out_dobs_cmp_npz,
                d_obs_inverted=d_inv0.detach()
                .cpu()
                .numpy(),  # (n_src, n_rec, nt) from v_hat[0]
                d_obs_true=d_obs.detach().cpu().numpy(),  # (n_src, n_rec, nt) observed
            )
    if traj_capture and traj_capture.get("frames") is not None:
        frames0 = traj_capture["frames"][:, 0]  # sample 0: (n_frames, C?, H, W)
        if frames0.ndim == 3:  # FWI native path delivers (n_frames, H, W)
            frames0 = frames0[:, None]
        frames0 = frames0.to(device)
        map_label = (
            "iterate"
            if method_name in ("classical_fwi", "realistic_fwi")
            else "Tweedie"
        )
        traj_base = out_png.with_name(f"{method_name}_g{gs:.2g}_dobs_traj")
        for sc in dobs_scales:
            plot_dobs_trajectory(
                v_true,
                frames0,
                list(traj_capture["steps"]),
                d_obs,
                forward_fn,
                traj_base.with_name(f"{traj_base.name}_{sc}.png"),
                scale=sc,
                title=f"{cmp_label or method_name} · {label} · sample 0",
                total_steps=int(traj_capture["total_steps"]),
                map_label=map_label,
            )
        # f-k spectral twin: emitted once (dB is its own scale, no linear/log pair).
        plot_dobs_spectrum_trajectory(
            v_true,
            frames0,
            list(traj_capture["steps"]),
            d_obs,
            forward_fn,
            out_png.with_name(f"{method_name}_g{gs:.2g}_dobs_fk_traj.png"),
            title=f"{cmp_label or method_name} · {label} · sample 0",
            total_steps=int(traj_capture["total_steps"]),
            map_label=map_label,
        )
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
            summary["inv/guidance_misfit_guided"] = float(gm(pred_g).mean())
            summary["inv/guidance_misfit_unguided"] = float(gm(pred_u).mean())
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


def plot_dobs_compare(
    d_true: Tensor, d_inv: Tensor, out_png: Path, *, title: str
) -> None:
    """Data-space fit check: observed seismic (from ``v_true``) vs seismic re-simulated from
    the inverted velocity (sample 0), per source, with the residual. Complements the
    model-space ``true | v_hat | error`` figure — it shows whether recovery reproduces the
    *waveforms* the inversion was conditioned on. Arrays are ``(n_src, n_receivers, nt)``.

    Columns: ``true d_obs`` | ``inverted d_obs`` (shared symmetric scale) | ``inverted − true``
    residual (own, tighter scale). The header reports the sample-0 residual RMS as a percent of
    the data RMS (the fraction of the observed energy left unexplained)."""
    dt = d_true.detach().cpu().numpy()
    di = d_inv.detach().cpu().numpy()
    resid = di - dt
    n_src = dt.shape[0]
    vabs = float(np.percentile(np.abs(np.stack([dt, di])), 99)) or 1.0
    rabs = float(np.percentile(np.abs(resid), 99)) or 1.0
    r_rms = float(np.sqrt(np.mean(resid**2)))
    d_rms = float(np.sqrt(np.mean(dt**2))) or 1.0
    cols = [
        ("true d_obs", dt, vabs),
        ("inverted d_obs", di, vabs),
        ("inverted − true", resid, rabs),
    ]
    fig, axes = plt.subplots(n_src, 3, figsize=(9, 2.4 * n_src), squeeze=False)
    for s in range(n_src):
        for c, (name, data, scale) in enumerate(cols):
            ax = axes[s, c]
            # (n_receivers, nt) -> (nt, n_receivers): time on the vertical axis
            im = ax.imshow(
                data[s].T, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale
            )
            if s == 0:
                ax.set_title(name, fontsize=10)
            ax.set_ylabel(f"source {s + 1}\ntime sample" if c == 0 else "", fontsize=8)
            if s == n_src - 1 and c == 0:
                ax.set_xlabel("receiver", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"{title}\nd_obs from forward operator · residual RMS = {r_rms:.3f} "
        f"({100 * r_rms / d_rms:.1f}% of data RMS {d_rms:.2f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_dobs_trajectory(
    v_true: Tensor,
    frames_norm: Tensor,
    frame_steps: list[int],
    d_obs_true: Tensor,
    forward_fn: Callable[[Tensor], Tensor],
    out_png: Path,
    *,
    scale: str,
    title: str,
    total_steps: int,
    map_label: str = "Tweedie",
) -> None:
    """Data-space trajectory grid for sample 0. Top row: the true velocity + each captured
    ``frames_norm`` prediction (viridis, shared scale, titled with its trajectory step). Rows below:
    one per seismic source — column 0 is the true velocity's ``d_obs``, columns 1.. are the d_obs
    re-simulated from each prediction via ``forward_fn`` (shared symmetric scale, linear or symlog).

    ``frames_norm`` is ``(n_frames, 1, res, res)`` normalized [-1,1]; ``d_obs_true`` is
    ``(n_src, n_rec, nt)``. ``map_label`` names the prediction ("Tweedie" for prior methods,
    "iterate" for the FWI baselines). The ``scale`` switch applies to the seismic rows only."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' | 'log', got {scale!r}")
    with torch.no_grad():
        d_frames = (
            forward_fn(frames_norm).detach().cpu().numpy()
        )  # (n_frames, n_src, n_rec, nt)
    v_hat = to_mps_native(frames_norm).detach().cpu().numpy()  # (n_frames, 70, 70) m/s
    vt = v_true.detach().cpu().numpy()
    dt = d_obs_true.detach().cpu().numpy()  # (n_src, n_rec, nt)
    n_frames = int(frames_norm.shape[0])
    n_src = int(dt.shape[0])
    n_cols = 1 + n_frames
    # One symmetric scale across the true column and every frame, so panels are comparable.
    vabs = (
        float(np.percentile(np.abs(np.concatenate([dt[None], d_frames], axis=0)), 99))
        or 1.0
    )
    vlo, vhi = float(vt.min()), float(vt.max())

    fig, axes = plt.subplots(
        1 + n_src, n_cols, figsize=(2.1 * n_cols, 2.1 * (1 + n_src)), squeeze=False
    )
    # Row 0 — velocity maps (viridis; scale switch does NOT apply here).
    vimg = axes[0, 0].imshow(vt, cmap="viridis", vmin=vlo, vmax=vhi)
    axes[0, 0].set_title("true v", fontsize=9)
    for j in range(n_frames):
        axes[0, 1 + j].imshow(v_hat[j], cmap="viridis", vmin=vlo, vmax=vhi)
        axes[0, 1 + j].set_title(f"{map_label}\nstep {frame_steps[j]}", fontsize=9)
    for c in range(n_cols):
        axes[0, c].axis("off")
    fig.colorbar(vimg, ax=axes[0, n_cols - 1], fraction=0.046, label="m/s")

    # Rows 1..n_src — shot gathers, column 0 = true, columns 1.. = frames.
    im = None
    for s in range(n_src):
        r = 1 + s
        im = _seismic_imshow(axes[r, 0], dt[s], scale=scale, vabs=vabs)
        axes[r, 0].set_ylabel(f"source {s + 1}\ntime", fontsize=8)
        if s == 0:
            axes[r, 0].set_title("true d_obs", fontsize=9)
        for j in range(n_frames):
            _seismic_imshow(axes[r, 1 + j], d_frames[j, s], scale=scale, vabs=vabs)
            if s == 0:
                axes[r, 1 + j].set_title(f"step {frame_steps[j]}", fontsize=9)
            axes[r, 1 + j].set_yticklabels([])
        for c in range(n_cols):
            axes[r, c].set_xticklabels([])
    label = "amplitude (symlog)" if scale == "log" else "amplitude"
    fig.colorbar(
        im, ax=axes[1:, n_cols - 1].ravel().tolist(), fraction=0.046, label=label
    )
    tag = " · log" if scale == "log" else ""
    fig.suptitle(
        f"{title}{tag}\nd_obs from {map_label} predictions over {total_steps} steps",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_dobs_spectrum_compare(
    d_true: Tensor,
    d_inv: Tensor,
    out_png: Path,
    *,
    title: str,
    dt: float = 1e-3,
    dx: float = 10.0,
    fmax: float = 60.0,
) -> None:
    """f-k spectral twin of :func:`plot_dobs_compare`. Columns: true d_obs | inverted d_obs |
    residual (inverted − true), each rendered as its 2-D f-k spectrum in dB. All three columns
    share one global dB peak (the loudest bin across the true and inverted gathers), so the
    residual reads honestly as N dB below the data rather than being re-normalised to its own
    peak. One row per source. Arrays are ``(n_src, n_receivers, nt)``; ``dt``/``dx`` set the
    frequency/wavenumber axes and ``fmax`` crops the frequency axis (default 60 Hz)."""
    d_t = d_true.detach().cpu().numpy()  # (n_src, n_rec, nt)
    d_i = d_inv.detach().cpu().numpy()
    resid = d_i - d_t
    n_src = d_t.shape[0]

    # Shared peak over the true + inverted gathers only (not the residual): 0 dB is the loudest
    # data bin everywhere, so the residual column shows how far below the data the misfit sits.
    peak = 1.0
    for s in range(n_src):
        peak = max(peak, float(_fk_mag(d_t[s], dt, dx)[0].max()))
        peak = max(peak, float(_fk_mag(d_i[s], dt, dx)[0].max()))

    cols = [
        ("true d_obs", d_t),
        ("inverted d_obs", d_i),
        ("inverted − true", resid),
    ]
    fig, axes = plt.subplots(n_src, 3, figsize=(7.5, 2.4 * n_src), squeeze=False)
    im = None
    for s in range(n_src):
        for c, (name, data) in enumerate(cols):
            im = _fk_imshow(axes[s, c], data[s], dt=dt, dx=dx, peak=peak, fmax=fmax)
            if s == 0:
                axes[s, c].set_title(name, fontsize=10)
            if c == 0:
                axes[s, c].set_ylabel(f"source {s + 1}\nfrequency (Hz)", fontsize=8)
            else:
                axes[s, c].set_yticklabels([])
            if s == n_src - 1:
                axes[s, c].set_xlabel("wavenumber (cyc/m)", fontsize=8)
            else:
                axes[s, c].set_xticklabels([])
    fig.colorbar(
        im, ax=axes.ravel().tolist(), fraction=0.046, label="magnitude (dB, rel. peak)"
    )
    fig.suptitle(f"{title} · f-k spectrum", fontsize=11)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _seismic_imshow(
    ax: plt.Axes, gather: np.ndarray, *, scale: str, vabs: float, cmap: str = "RdBu_r"
) -> AxesImage:
    """imshow one ``(n_receivers, nt)`` gather with time on the vertical axis, on a linear or
    symmetric-log amplitude scale.

    ``scale="linear"`` is the plain symmetric ``±vabs`` diverging map. ``scale="log"`` swaps in a
    ``SymLogNorm`` (a ±``linthresh`` linear band around zero, log beyond) so low-amplitude coda is
    lifted without discarding sign. ``vabs`` is a shared symmetric colour limit; pass one global
    value across panels so they stay comparable."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' | 'log', got {scale!r}")
    data = gather.T  # (nt, n_receivers): time down
    if scale == "log":
        linthresh = max(vabs * 1e-2, 1e-12)
        norm = SymLogNorm(linthresh=linthresh, vmin=-vabs, vmax=vabs)
        return ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    return ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)


def _fk_mag(
    gather: np.ndarray, dt: float, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear f-k magnitude of one ``(n_receivers, nt)`` gather. ``rfft`` over time gives the
    positive-frequency half (the real signal's spectrum is Hermitian); a full ``fft`` +
    ``fftshift`` over receivers gives signed wavenumber so left/right-going wavefronts separate.
    Returns ``(mag, f_axis, k_axis)`` with ``mag`` shaped ``(n_freq, n_rec)`` = (f, k), both axes
    ascending. ``dt`` in seconds, ``dx`` receiver spacing in metres."""
    n_rec, nt = gather.shape
    s = np.fft.fftshift(
        np.fft.fft(np.fft.rfft(gather, axis=1), axis=0), axes=0
    )  # (n_rec, n_freq): fft over receivers (centred), rfft over time
    mag = np.abs(s).T  # (n_freq, n_rec) = (f, k)
    f_axis = np.fft.rfftfreq(nt, dt)  # 0 .. f_Nyq, ascending
    k_axis = np.fft.fftshift(np.fft.fftfreq(n_rec, dx))  # -k_Nyq .. +k_Nyq, ascending
    return mag, f_axis, k_axis


def fk_spectrum(
    gather: np.ndarray,
    dt: float,
    dx: float,
    *,
    peak: float | None = None,
    floor_db: float = -80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D f-k spectrum of one ``(n_receivers, nt)`` gather in dB. Magnitude is
    ``20*log10(|S| / peak)`` floored at ``floor_db``; ``peak`` defaults to this gather's own
    magnitude peak, but callers pass a shared global peak so panels stay comparable (the peak
    then maps to 0 dB everywhere). Returns ``(mag_db, f_axis, k_axis)`` — see :func:`_fk_mag`
    for axis conventions."""
    mag, f_axis, k_axis = _fk_mag(gather, dt, dx)
    p = (float(mag.max()) if peak is None else peak) or 1.0
    mag_db = 20.0 * np.log10(np.clip(mag / p, 10.0 ** (floor_db / 20.0), None))
    return mag_db, f_axis, k_axis


def _fk_imshow(
    ax: plt.Axes,
    gather: np.ndarray,
    *,
    dt: float,
    dx: float,
    peak: float,
    fmax: float,
    floor_db: float = -80.0,
    cmap: str = "magma",
) -> AxesImage:
    """imshow one ``(n_receivers, nt)`` gather's f-k spectrum: temporal frequency (Hz) up the
    vertical axis (cropped to ``fmax``), signed wavenumber (cyc/m) across, magnitude in dB
    relative to the shared ``peak``. ``vmin``/``vmax`` are fixed to ``floor_db``/``0`` so colour
    reads the same in every panel."""
    mag_db, f_axis, k_axis = fk_spectrum(gather, dt, dx, peak=peak, floor_db=floor_db)
    fmask = f_axis <= fmax
    img = mag_db[fmask]  # (n_freq_crop, n_rec)
    extent = (
        float(k_axis[0]),
        float(k_axis[-1]),
        float(f_axis[fmask][0]),
        float(f_axis[fmask][-1]),
    )
    return ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=floor_db,
        vmax=0.0,
    )


def plot_dobs_spectrum_trajectory(
    v_true: Tensor,
    frames_norm: Tensor,
    frame_steps: list[int],
    d_obs_true: Tensor,
    forward_fn: Callable[[Tensor], Tensor],
    out_png: Path,
    *,
    title: str,
    total_steps: int,
    map_label: str = "Tweedie",
    dt: float = 1e-3,
    dx: float = 10.0,
    fmax: float = 60.0,
) -> None:
    """f-k spectral twin of :func:`plot_dobs_trajectory` for sample 0. Row 0 is the identical
    velocity-map row (true v + each prediction, viridis, shared scale). Rows below (one per
    seismic source) replace each ``d_obs`` gather with its 2-D f-k spectrum: column 0 is the true
    velocity's ``d_obs``, columns 1.. are the d_obs re-simulated from each ``frames_norm``
    prediction via ``forward_fn``. All spectra share one global dB peak so colour is comparable.

    ``frames_norm`` is ``(n_frames, 1, res, res)`` in [-1,1]; ``d_obs_true`` is
    ``(n_src, n_rec, nt)``. ``dt``/``dx`` set the frequency/wavenumber axes; ``fmax`` crops the
    frequency axis (default 60 Hz for the 15 Hz Ricker source)."""
    with torch.no_grad():
        d_frames = (
            forward_fn(frames_norm).detach().cpu().numpy()
        )  # (n_frames, n_src, n_rec, nt)
    v_hat = to_mps_native(frames_norm).detach().cpu().numpy()  # (n_frames, 70, 70) m/s
    vt = v_true.detach().cpu().numpy()
    dt_arr = d_obs_true.detach().cpu().numpy()  # (n_src, n_rec, nt)
    n_frames = int(frames_norm.shape[0])
    n_src = int(dt_arr.shape[0])
    n_cols = 1 + n_frames
    vlo, vhi = float(vt.min()), float(vt.max())

    # One shared linear-magnitude peak across the true column and every frame, all sources,
    # so the dB scale (peak -> 0 dB) is comparable across the whole grid.
    peak = 1.0
    for s in range(n_src):
        peak = max(peak, float(_fk_mag(dt_arr[s], dt, dx)[0].max()))
        for j in range(n_frames):
            peak = max(peak, float(_fk_mag(d_frames[j, s], dt, dx)[0].max()))

    fig, axes = plt.subplots(
        1 + n_src, n_cols, figsize=(2.1 * n_cols, 2.1 * (1 + n_src)), squeeze=False
    )
    # Row 0 — velocity maps (identical to plot_dobs_trajectory).
    vimg = axes[0, 0].imshow(vt, cmap="viridis", vmin=vlo, vmax=vhi)
    axes[0, 0].set_title("true v", fontsize=9)
    for j in range(n_frames):
        axes[0, 1 + j].imshow(v_hat[j], cmap="viridis", vmin=vlo, vmax=vhi)
        axes[0, 1 + j].set_title(f"{map_label}\nstep {frame_steps[j]}", fontsize=9)
    for c in range(n_cols):
        axes[0, c].axis("off")
    fig.colorbar(vimg, ax=axes[0, n_cols - 1], fraction=0.046, label="m/s")

    # Rows 1..n_src — f-k spectra, column 0 = true, columns 1.. = frames.
    im = None
    for s in range(n_src):
        r = 1 + s
        im = _fk_imshow(axes[r, 0], dt_arr[s], dt=dt, dx=dx, peak=peak, fmax=fmax)
        axes[r, 0].set_ylabel(f"source {s + 1}\nfrequency (Hz)", fontsize=8)
        if s == 0:
            axes[r, 0].set_title("true d_obs", fontsize=9)
        for j in range(n_frames):
            _fk_imshow(
                axes[r, 1 + j], d_frames[j, s], dt=dt, dx=dx, peak=peak, fmax=fmax
            )
            if s == 0:
                axes[r, 1 + j].set_title(f"step {frame_steps[j]}", fontsize=9)
            axes[r, 1 + j].set_yticklabels([])
        for c in range(n_cols):
            if s < n_src - 1:
                axes[r, c].set_xticklabels([])
            else:
                axes[r, c].set_xlabel("wavenumber (cyc/m)", fontsize=8)
    fig.colorbar(
        im,
        ax=axes[1:, n_cols - 1].ravel().tolist(),
        fraction=0.046,
        label="magnitude (dB, rel. peak)",
    )
    fig.suptitle(
        f"{title} · f-k\nd_obs f-k spectrum from {map_label} predictions over "
        f"{total_steps} steps",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_seismic(
    d_obs: Tensor, gidx: int, out_png: Path, *, scale: str = "linear"
) -> None:
    """Shot gathers of the observed seismic ``d_obs`` (n_sources, n_receivers, nt) — the input the
    velocity is inverted from. One panel per source: time (down) x receiver, shared symmetric scale.
    ``scale`` selects linear or symmetric-log amplitude (:func:`_seismic_imshow`)."""
    d = d_obs.detach().cpu().numpy()
    n_src = d.shape[0]
    vabs = float(np.percentile(np.abs(d), 99)) or 1.0
    fig, axes = plt.subplots(1, n_src, figsize=(2.2 * n_src, 3.6), squeeze=False)
    for s in range(n_src):
        ax = axes[0, s]
        im = _seismic_imshow(ax, d[s], scale=scale, vabs=vabs)
        ax.set_title(f"source {s + 1}", fontsize=9)
        ax.set_xlabel("receiver", fontsize=8)
        ax.set_ylabel("time sample" if s == 0 else "", fontsize=8)
        if s > 0:
            ax.set_yticklabels([])
    tag = " · log" if scale == "log" else ""
    fig.suptitle(f"observed seismic d_obs · val map {gidx}{tag}", fontsize=10)
    label = "amplitude (symlog)" if scale == "log" else "amplitude"
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, label=label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
