"""Registry-driven dataset tests: every entry in DATASETS is exercised."""

from pathlib import Path
from typing import Sized, cast

import pytest
import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import (
    DATASETS,
    GaussiansDatasetConfig,
)


@pytest.fixture(params=sorted(DATASETS))
def dataset_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def test_registry_metadata(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    assert isinstance(cfg.shape, tuple) and all(d > 0 for d in cfg.shape)
    assert cfg.num_classes is None or cfg.num_classes > 0
    assert isinstance(cfg.requires_download, bool)


def test_build_shapes(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    if cfg.requires_download:
        pytest.skip(f"{dataset_name} requires download; exercised by the live run")
    ds = cfg.build()
    x1, label = ds[0]
    assert x1.shape == cfg.shape
    if cfg.num_classes is None:
        assert int(label) == 0
    else:
        assert 0 <= int(label) < cfg.num_classes
    loader = DataLoader(ds, batch_size=16)
    xb, lb = next(iter(loader))
    assert xb.shape == (16, *cfg.shape)
    assert lb.shape == (16,)


def test_visualize_writes_file(dataset_name: str, tmp_path: Path) -> None:
    cfg = DATASETS[dataset_name]
    samples = torch.randn(16, *cfg.shape)
    out = tmp_path / f"{dataset_name}.png"
    cfg.visualize(samples, out)
    assert out.exists() and out.stat().st_size > 0


def test_build_val_shapes(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    if cfg.requires_download:
        pytest.skip(f"{dataset_name} requires download; exercised by the live run")
    ds = cfg.build_val()
    x1, label = ds[0]
    assert x1.shape == cfg.shape
    if cfg.num_classes is None:
        assert int(label) == 0
    else:
        assert 0 <= int(label) < cfg.num_classes


def test_gaussians_build_val_is_distinct() -> None:
    cfg = GaussiansDatasetConfig(n_samples=64, val_samples=32)
    train_ds = cfg.build()
    val_ds = cfg.build_val()
    assert len(cast(Sized, val_ds)) == 32
    assert not torch.equal(
        train_ds[0][0], val_ds[0][0]
    )  # distinct seed -> different draw
