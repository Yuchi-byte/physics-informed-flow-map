"""d_obs plotting: linear/symlog scale switch and the Tweedie-trajectory grid."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest
import torch

from physics_informed_flow_map.inversion.single_target import _plot_seismic


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
