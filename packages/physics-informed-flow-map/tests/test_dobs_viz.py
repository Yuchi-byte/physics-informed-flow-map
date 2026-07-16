"""d_obs plotting: linear/symlog scale switch and the Tweedie-trajectory grid."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from physics_informed_flow_map.inversion.single_target import (
    _plot_seismic,
    fk_spectrum,
    plot_dobs_trajectory,
)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_plot_seismic_writes_png(tmp_path: Path, scale: str) -> None:
    d_obs = torch.randn(5, 70, 1001)  # (n_src, n_rec, nt)
    out = tmp_path / f"d_obs_{scale}.png"
    _plot_seismic(d_obs, gidx=42, out_png=out, scale=scale)
    assert out.exists() and out.stat().st_size > 0


def test_plot_seismic_rejects_bad_scale(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scale"):
        _plot_seismic(
            torch.randn(2, 8, 16), gidx=0, out_png=tmp_path / "x.png", scale="db"
        )


def _stub_forward(n_src: int, n_rec: int, nt: int):
    # frames_norm (n_frames,1,res,res) -> (n_frames, n_src, n_rec, nt)
    def fwd(frames: torch.Tensor) -> torch.Tensor:
        return torch.randn(frames.shape[0], n_src, n_rec, nt)

    return fwd


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_plot_dobs_trajectory_writes_png(tmp_path: Path, scale: str) -> None:
    n_frames, n_src, n_rec, nt = 2, 5, 12, 40
    v_true = torch.rand(70, 70) * 1000 + 1500
    frames_norm = torch.rand(n_frames, 1, 16, 16) * 2 - 1
    d_obs_true = torch.randn(n_src, n_rec, nt)
    out = tmp_path / f"traj_{scale}.png"
    plot_dobs_trajectory(
        v_true,
        frames_norm,
        [0, 3],
        d_obs_true,
        _stub_forward(n_src, n_rec, nt),
        out,
        scale=scale,
        title="demo",
        total_steps=4,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_dobs_trajectory_panel_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_informed_flow_map.inversion.single_target as st

    seen: dict[str, tuple[int, int]] = {}
    real = st.plt.subplots

    def spy(nrows: int, ncols: int, **kw: object):
        seen["shape"] = (nrows, ncols)
        return real(nrows, ncols, **kw)

    monkeypatch.setattr(st.plt, "subplots", spy)
    plot_dobs_trajectory(
        torch.rand(70, 70),
        torch.rand(2, 1, 16, 16),
        [0, 3],
        torch.randn(5, 12, 40),
        _stub_forward(5, 12, 40),
        tmp_path / "t.png",
        scale="linear",
        title="d",
        total_steps=4,
    )
    assert seen["shape"] == (1 + 5, 1 + 2)  # (1 + n_src) rows, (1 + n_frames) cols


def test_fk_spectrum_axes_ranges() -> None:
    n_rec, nt, dt, dx = 70, 1001, 1e-3, 10.0
    gather = np.random.randn(n_rec, nt)
    mag_db, f_axis, k_axis = fk_spectrum(gather, dt, dx)
    # temporal: rfft -> nt//2 + 1 positive freqs, 0 .. Nyquist(=1/(2dt)=500 Hz)
    assert f_axis.shape == (nt // 2 + 1,)
    assert f_axis[0] == 0.0
    # odd nt tops out one bin short of the exact Nyquist (500 Hz); allow one bin
    assert f_axis[-1] <= 1.0 / (2 * dt)
    assert np.isclose(f_axis[-1], 1.0 / (2 * dt), atol=f_axis[1] - f_axis[0])
    # spatial: full fft over receivers, fftshifted -> symmetric about 0, Nyquist=1/(2dx)=0.05
    assert k_axis.shape == (n_rec,)
    assert np.isclose(np.abs(k_axis).max(), 1.0 / (2 * dx))  # 0.05 cyc/m
    assert np.all(np.diff(k_axis) > 0)  # ascending
    assert mag_db.shape == (f_axis.size, k_axis.size)  # (f, k)


def test_fk_spectrum_db_normalized_to_peak() -> None:
    gather = np.random.randn(32, 128)
    mag_db, _, _ = fk_spectrum(gather, 1e-3, 10.0)
    assert mag_db.max() <= 0.0 + 1e-9  # peak normalized to 0 dB
    assert np.isclose(mag_db.max(), 0.0)
    assert mag_db.min() >= -80.0 - 1e-9  # floored


def test_fk_spectrum_locates_single_tone() -> None:
    # A pure temporal tone at f0 with a single spatial wavelength should peak
    # at the matching (f, k) bin.
    n_rec, nt, dt, dx = 64, 512, 1e-3, 10.0
    f0, k0 = 20.0, 1.0 / (8 * dx)  # 20 Hz, wavelength 8 receivers
    t = np.arange(nt) * dt
    x = np.arange(n_rec) * dx
    gather = np.cos(2 * np.pi * f0 * t)[None, :] * np.cos(2 * np.pi * k0 * x)[:, None]
    mag_db, f_axis, k_axis = fk_spectrum(gather, dt, dx)
    fi, ki = np.unravel_index(np.argmax(mag_db), mag_db.shape)
    assert np.isclose(f_axis[fi], f0, atol=f_axis[1] - f_axis[0])
    assert np.isclose(abs(k_axis[ki]), k0, atol=k_axis[1] - k_axis[0])
