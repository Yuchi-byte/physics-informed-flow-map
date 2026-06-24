import torch

from physics_informed_flow_map.flow_matching.datasets import DatasetSpec
from physics_informed_flow_map.flow_matching.models import build_model


def _has_finite_grads(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def _fwd_bwd(model, x):
    b = x.shape[0]
    s = torch.rand(b)
    t_cond = torch.zeros(b)
    x_cond = torch.zeros_like(x)
    v = model.v(s, s, x, t_cond, x_cond)
    assert v.shape == x.shape  # forward: velocity shaped like input
    loss = v.pow(2).mean()
    loss.backward()  # backward
    assert _has_finite_grads(model)


def test_vector_model_forward_backward():
    spec = DatasetSpec(shape=(2,), num_classes=None, make_dataset=lambda: None, visualize=lambda *_: None)
    model = build_model(spec, mlp_width=16, mlp_depth=2)
    _fwd_bwd(model, torch.randn(4, 2))


def test_image_model_forward_backward():
    spec = DatasetSpec(shape=(1, 32, 32), num_classes=10, make_dataset=lambda: None, visualize=lambda *_: None)
    model = build_model(spec, dit_hidden=32, dit_depth=1, num_heads=4)
    _fwd_bwd(model, torch.randn(2, 1, 32, 32))


def test_non_square_image_rejected():
    import pytest
    spec = DatasetSpec(shape=(1, 16, 32), num_classes=None, make_dataset=lambda: None, visualize=lambda *_: None)
    with pytest.raises(ValueError):
        build_model(spec)
