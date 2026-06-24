"""Hydra-composed DictConfig -> typed pydantic Config, with strict validation."""

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from physics_informed_flow_map.experiment import Config


class _Demo(Config):
    a: int = 1
    b: str = "x"


def test_from_dictconfig_returns_subclass_instance() -> None:
    cfg = OmegaConf.create({"a": 5, "b": "y"})
    out = _Demo.from_dictconfig(cfg)
    assert isinstance(out, _Demo)
    assert out.a == 5
    assert out.b == "y"


def test_from_dictconfig_rejects_unknown_key() -> None:
    cfg = OmegaConf.create({"a": 5, "b": "y", "typo": 1})
    with pytest.raises(ValidationError):
        _Demo.from_dictconfig(cfg)
