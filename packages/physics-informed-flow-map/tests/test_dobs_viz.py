"""d_obs plotting: linear/symlog scale switch and the Tweedie-trajectory grid."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

from physics_informed_flow_map.inversion.single_target import (
    _plot_seismic,
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
