import torch

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import MLPModelConfig, build_model
from physics_informed_flow_map.flow_matching.sample import real_reference, sample


def test_sample_shape() -> None:
    cfg = DATASETS["gaussians"]
    model = build_model(cfg.shape, cfg.num_classes, MLPModelConfig(width=16, depth=2))
    out = sample(model, 32, cfg.shape, sampler_steps=5, device=torch.device("cpu"))
    assert out.shape == (32, 2)


def test_real_reference() -> None:
    ds = DATASETS["gaussians"].build()
    ref = real_reference(ds, 100, torch.device("cpu"))
    assert ref.shape == (100, 2)
