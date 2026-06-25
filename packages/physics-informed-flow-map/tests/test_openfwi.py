"""OpenFWI velocity-map dataset: hermetic tests on synthetic .npy fixtures."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.flow_matching.openfwi import (
    NATIVE,
    VMAX,
    VMIN,
    OpenFWIVelocityDataset,
    viz_velocity,
)


def _write_family(
    root: Path, family: str, layout: str, n_files: int = 2, rows: int = 3
) -> None:
    """Write `n_files` synthetic velocity files for one family, in the given layout.

    layout="model" -> <family>/model/model{i}.npy
    layout="flat"  -> <family>/vel{i}.npy
    Each file holds `rows` maps of shape (1, 70, 70) spanning [VMIN, VMAX].
    """
    if layout == "model":
        out_dir = root / family / "model"
        names = [f"model{i}.npy" for i in range(n_files)]
    else:
        out_dir = root / family
        names = [f"vel{i}.npy" for i in range(n_files)]
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.linspace(VMIN, VMAX, rows * 70 * 70, dtype=np.float32).reshape(
        rows, 1, 70, 70
    )
    for name in names:
        np.save(out_dir / name, arr)


def test_len_counts_all_rows(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A"])
    assert len(ds) == 6  # 2 files * 3 rows


def test_sample_shape_and_range(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model")
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A"], resolution=64)
    x, label = ds[0]
    assert x.shape == (1, 64, 64)
    assert x.dtype == torch.float32
    assert int(label) == 0
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0


def test_normalization_endpoints_native_resolution(tmp_path: Path) -> None:
    # At native resolution (no resize), VMIN -> -1 and VMAX -> +1 exactly.
    _write_family(tmp_path, "FlatVel_A", "model", n_files=1, rows=1)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A"], resolution=NATIVE)
    x, _ = ds[0]
    assert x.shape == (1, NATIVE, NATIVE)
    assert float(x.min()) == pytest.approx(-1.0, abs=1e-4)
    assert float(x.max()) == pytest.approx(1.0, abs=1e-4)


def test_flat_vel_layout_loads(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=2, rows=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatFault_A"])
    assert len(ds) == 6
    x, _ = ds[0]
    assert x.shape == (1, 64, 64)


def test_multiple_families(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=1, rows=3)
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=1, rows=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A", "FlatFault_A"])
    assert len(ds) == 6


def test_missing_family_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenFWIVelocityDataset(tmp_path, ["DoesNotExist"])


def test_viz_velocity_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "vel.png"
    viz_velocity(torch.randn(16, 1, 64, 64), out)
    assert out.exists() and out.stat().st_size > 0


def test_openfwi_config_split_is_disjoint(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=4)  # 8 maps total
    cfg = OpenFWIDatasetConfig(
        data_dir=str(tmp_path), families=["FlatVel_A"], val_fraction=0.25
    )
    train_ds = cfg.build()
    val_ds = cfg.build_val()
    assert isinstance(train_ds, Subset) and isinstance(val_ds, Subset)
    assert len(train_ds) + len(val_ds) == 8
    assert len(val_ds) == max(1, int(0.25 * 8))  # 2
    assert set(train_ds.indices).isdisjoint(set(val_ds.indices))
