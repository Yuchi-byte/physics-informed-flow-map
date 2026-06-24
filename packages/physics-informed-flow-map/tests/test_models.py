"""Registry-driven model tests: every entry in MODELS does a forward + backward."""

from typing import cast

import pytest
import torch

from physics_informed_flow_map.flow_matching.models import MODELS, build_model

# Tiny knobs so every architecture builds and runs fast on CPU. build_model
# ignores the knobs irrelevant to the chosen model.
_TINY = dict(mlp_width=16, mlp_depth=2, dit_hidden=32, dit_depth=1, num_heads=4)


@pytest.fixture(params=sorted(MODELS))
def model_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def _has_finite_grads(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return len(grads) > 0 and all(bool(torch.isfinite(g).all()) for g in grads)


def test_model_forward_backward(model_name: str) -> None:
    spec = MODELS[model_name]
    model = build_model(spec.sample_shape, spec.num_classes, **_TINY)

    x = torch.randn(4, *spec.sample_shape)
    s = torch.rand(x.shape[0])
    t_cond = torch.zeros(x.shape[0])
    x_cond = torch.zeros_like(x)

    v = model.v(s, s, x, t_cond, x_cond)
    assert v.shape == x.shape  # forward: velocity shaped like input

    loss = v.pow(2).mean()
    loss.backward()  # backward
    assert _has_finite_grads(model)


def test_non_square_image_rejected() -> None:
    with pytest.raises(ValueError):
        build_model((1, 16, 32), None, **_TINY)
