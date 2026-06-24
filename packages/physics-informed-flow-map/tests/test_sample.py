import torch

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)


def test_energy_distance_zero_for_same_distribution():
    torch.manual_seed(0)
    x = torch.randn(2000, 2)
    y = torch.randn(2000, 2)
    assert abs(energy_distance(x, x)) < 1e-4
    far = torch.randn(2000, 2) + 50.0
    assert energy_distance(x, far) > energy_distance(x, y)


def test_sample_shape():
    spec = DATASETS["gaussians"]
    model = build_model(spec, mlp_width=16, mlp_depth=2)
    out = sample(model, 32, spec.shape, sampler_steps=5, device=torch.device("cpu"))
    assert out.shape == (32, 2)


def test_real_reference():
    ds = DATASETS["gaussians"].make_dataset()
    ref = real_reference(ds, 100, torch.device("cpu"))
    assert ref.shape == (100, 2)
