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
    RandomHFlip,
    viz_velocity,
)


def _write_family(
    root: Path,
    family: str,
    layout: str,
    n_files: int = 2,
    rows: int = 3,
    ndim: int = 4,
) -> None:
    """Write `n_files` synthetic velocity files for one family, in the given layout.

    layout="model" -> <family>/model/model{i}.npy
    layout="flat"  -> <family>/vel{i}.npy
    Each file holds `rows` maps spanning [VMIN, VMAX]; ndim=4 -> (rows, 1, 70, 70)
    (the OpenFWI shape), ndim=3 -> (rows, 70, 70) (channel-less variant).
    """
    if layout == "model":
        out_dir = root / family / "model"
        names = [f"model{i}.npy" for i in range(n_files)]
    else:
        out_dir = root / family
        names = [f"vel{i}.npy" for i in range(n_files)]
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = (rows, 1, 70, 70) if ndim == 4 else (rows, 70, 70)
    arr = np.linspace(VMIN, VMAX, rows * 70 * 70, dtype=np.float32).reshape(shape)
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


def test_channelless_files_load(tmp_path: Path) -> None:
    # Some distributions store (N, 70, 70) instead of (N, 1, 70, 70).
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=1, rows=3, ndim=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatFault_A"])
    assert len(ds) == 3
    x, _ = ds[0]
    assert x.shape == (1, 64, 64)


def test_family_ids_and_slices(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=1, rows=3)
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=1, rows=5)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A", "FlatFault_A"])
    assert ds.family_names == ["FlatVel_A", "FlatFault_A"]
    assert list(ds.family_ids) == [0] * 3 + [1] * 5
    assert ds.family_slices == [slice(0, 3), slice(3, 8)]


def _val_provenance(cfg: OpenFWIDatasetConfig, family: str) -> set[tuple[str, int]]:
    """Val-split membership of one family as (filename, row) pairs — global-index-free."""
    full, _, val_idx = cfg._split()
    fid = full.family_names.index(family)
    return {
        (full.index[i][0].name, full.index[i][1])
        for i in val_idx
        if full.family_ids[i] == fid
    }


def test_per_family_split_stable_under_family_changes(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=10)
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=2, rows=10)
    solo = OpenFWIDatasetConfig(
        data_dir=str(tmp_path), families=["FlatVel_A"], split_scheme="per_family"
    )
    both = OpenFWIDatasetConfig(
        data_dir=str(tmp_path),
        families=["FlatFault_A", "FlatVel_A"],  # different order + extra family
        split_scheme="per_family",
    )
    assert _val_provenance(solo, "FlatVel_A") == _val_provenance(both, "FlatVel_A")


def test_per_family_split_exact_val_counts(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=10)  # 20 maps
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=3, rows=10)  # 30 maps
    cfg = OpenFWIDatasetConfig(
        data_dir=str(tmp_path),
        families=["FlatVel_A", "FlatFault_A"],
        val_fraction=0.1,
        split_scheme="per_family",
    )
    full, train_idx, val_idx = cfg._split()
    assert len(train_idx) + len(val_idx) == 50
    assert set(train_idx).isdisjoint(set(val_idx))
    for fid, expected in [(0, 2), (1, 3)]:  # exactly 10% of each family
        assert sum(1 for i in val_idx if full.family_ids[i] == fid) == expected


def test_hflip_wraps_train_split_only(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=10)
    cfg = OpenFWIDatasetConfig(
        data_dir=str(tmp_path), families=["FlatVel_A"], hflip=True
    )
    train_ds = cfg.build()
    val_ds = cfg.build_val()
    assert not isinstance(val_ds, RandomHFlip)
    assert isinstance(train_ds, RandomHFlip)
    # Over repeated draws of one asymmetric sample, both orientations must appear.
    torch.manual_seed(0)
    base_x, _ = train_ds.base[0]
    seen_flipped, seen_plain = False, False
    for _ in range(64):
        x, _ = train_ds[0]
        if torch.equal(x, base_x):
            seen_plain = True
        elif torch.equal(x, torch.flip(base_x, dims=[-1])):
            seen_flipped = True
        else:
            raise AssertionError("hflip produced something other than a mirror")
    assert seen_plain and seen_flipped


def test_fingerprint_contents(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=10)
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=1, rows=10)
    cfg = OpenFWIDatasetConfig(
        data_dir=str(tmp_path),
        families=["FlatVel_A", "FlatFault_A"],
        split_scheme="per_family",
    )
    fp = cfg.fingerprint()
    assert fp["split_scheme"] == "per_family"
    assert fp["n_train"] + fp["n_val"] == 30  # type: ignore[operator]
    fams = fp["families"]
    assert fams["FlatVel_A"] == {  # type: ignore[index]
        "maps": 20,
        "files": 2,
        "split_seed": __import__("zlib").crc32(b"FlatVel_A"),
    }
