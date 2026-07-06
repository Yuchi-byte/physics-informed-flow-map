"""Registry-driven model tests: every entry in MODELS does a forward + backward."""

from typing import cast

import pytest
import torch

from physics_informed_flow_map.flow_matching.models import (
    MODELS,
    DiTModelConfig,
    MLPModelConfig,
    ModelConfig,
    build_model,
)

# Tiny configs so every architecture builds and runs fast on CPU.
_TINY: dict[str, ModelConfig] = {
    "mlp": MLPModelConfig(width=16, depth=2),
    "dit": DiTModelConfig(hidden=32, depth=1, num_heads=4),
}


@pytest.fixture(params=sorted(MODELS))
def model_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def _has_finite_grads(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return len(grads) > 0 and all(bool(torch.isfinite(g).all()) for g in grads)


def test_model_forward_backward(model_name: str) -> None:
    case = MODELS[model_name]
    model = build_model(case.sample_shape, case.num_classes, _TINY[model_name])

    x = torch.randn(4, *case.sample_shape)
    s = torch.rand(x.shape[0])
    t_cond = torch.zeros(x.shape[0])
    x_cond = torch.zeros_like(x)

    v = model.v(s, s, x, t_cond, x_cond)
    assert v.shape == x.shape  # forward: velocity shaped like input

    loss = v.pow(2).mean()
    loss.backward()  # backward
    assert _has_finite_grads(model)


def test_mlp_rejects_image() -> None:
    with pytest.raises(ValueError):
        build_model((1, 16, 16), None, MLPModelConfig())


def test_dit_rejects_vector() -> None:
    with pytest.raises(ValueError):
        build_model((2,), None, DiTModelConfig(hidden=32, depth=1))


def test_dit_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        build_model((1, 16, 32), 10, DiTModelConfig(hidden=32, depth=1))


def test_dit_preset_param_counts() -> None:
    # The dit_b (768/12/12) and dit_m (512/8/8) yaml presets, asserted at the builder level
    # so a silent yaml/param drift can't change the definitive prior capacity unnoticed.
    from physics_informed_flow_map.flow_matching.models import count_parameters

    dit_b = build_model(
        (1, 64, 64), None, DiTModelConfig(hidden=768, depth=12, num_heads=12, patch_size=4)
    )
    n_b, _ = count_parameters(dit_b)
    assert 100e6 < n_b < 180e6, f"dit_b params {n_b/1e6:.1f}M outside expected band"

    dit_m = build_model(
        (1, 64, 64), None, DiTModelConfig(hidden=512, depth=8, num_heads=8, patch_size=4)
    )
    n_m, _ = count_parameters(dit_m)
    assert 20e6 < n_m < 50e6, f"dit_m params {n_m/1e6:.1f}M outside expected band"
