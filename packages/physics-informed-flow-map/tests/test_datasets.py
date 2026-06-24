"""Registry-driven dataset tests: every entry in DATASETS is exercised."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS


@pytest.fixture(params=sorted(DATASETS))
def dataset_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def test_registry_metadata(dataset_name: str) -> None:
    spec = DATASETS[dataset_name]
    assert isinstance(spec.shape, tuple) and all(d > 0 for d in spec.shape)
    assert spec.num_classes is None or spec.num_classes > 0
    assert callable(spec.make_dataset)
    assert callable(spec.visualize)
    assert isinstance(spec.requires_download, bool)


def test_make_dataset_shapes(dataset_name: str) -> None:
    spec = DATASETS[dataset_name]
    if spec.requires_download:
        pytest.skip(f"{dataset_name} requires download; exercised by the live run")
    ds = spec.make_dataset()
    x1, label = ds[0]
    assert x1.shape == spec.shape
    if spec.num_classes is None:
        assert int(label) == 0
    else:
        assert 0 <= int(label) < spec.num_classes
    loader = DataLoader(ds, batch_size=16)
    xb, lb = next(iter(loader))
    assert xb.shape == (16, *spec.shape)
    assert lb.shape == (16,)


def test_visualize_writes_file(dataset_name: str, tmp_path: Path) -> None:
    spec = DATASETS[dataset_name]
    samples = torch.randn(16, *spec.shape)
    out = tmp_path / f"{dataset_name}.png"
    spec.visualize(samples, out)
    assert out.exists() and out.stat().st_size > 0
