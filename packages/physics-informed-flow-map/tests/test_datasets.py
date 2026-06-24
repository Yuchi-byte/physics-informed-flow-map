from pathlib import Path

import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS


def test_registry_metadata():
    for name, spec in DATASETS.items():
        assert isinstance(spec.shape, tuple) and all(d > 0 for d in spec.shape)
        assert spec.num_classes is None or spec.num_classes > 0
        assert callable(spec.make_dataset)
        assert callable(spec.visualize)
    assert DATASETS["gaussians"].shape == (2,)
    assert DATASETS["gaussians"].num_classes is None
    assert DATASETS["mnist"].shape == (1, 32, 32)
    assert DATASETS["mnist"].num_classes == 10


def test_gaussians_samples_and_loader():
    spec = DATASETS["gaussians"]
    ds = spec.make_dataset()
    x1, label = ds[0]
    assert x1.shape == spec.shape
    assert int(label) == 0
    loader = DataLoader(ds, batch_size=16)
    xb, lb = next(iter(loader))
    assert xb.shape == (16, 2)
    assert lb.shape == (16,)


def test_gaussians_visualize(tmp_path: Path):
    spec = DATASETS["gaussians"]
    samples = torch.randn(64, 2)
    out = tmp_path / "scatter.png"
    spec.visualize(samples, out)
    assert out.exists() and out.stat().st_size > 0
