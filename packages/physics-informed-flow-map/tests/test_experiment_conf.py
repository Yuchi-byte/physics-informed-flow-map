"""0001 Hydra config groups compose and validate into FlowMatchingConfig."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from physics_informed_flow_map.flow_matching.datasets import GaussiansDatasetConfig
from physics_informed_flow_map.flow_matching.models import DiTModelConfig

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "0001_flow_matching"
CONF = EXP / "conf"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fm_run", EXP / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "variant,dataset_name,model_kind,n_epochs",
    [
        ("gaussians", "gaussians", "mlp", 100),
        ("mnist", "mnist", "dit", 100),
        ("smoke", "gaussians", "mlp", 1),
    ],
)
def test_compose_validates(
    variant: str, dataset_name: str, model_kind: str, n_epochs: int
) -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=[f"experiment={variant}"])
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.dataset.name == dataset_name
    assert cfg.model.kind == model_kind
    assert cfg.training.n_epochs == n_epochs


def test_compose_applies_cli_override() -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(
            config_name="config",
            overrides=["experiment=mnist", "training.n_epochs=80"],
        )
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.training.n_epochs == 80
    assert cfg.dataset.name == "mnist"


def test_incompatible_model_dataset_rejected() -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with pytest.raises(ValidationError):
        cfg_cls(model=DiTModelConfig(), dataset=GaussiansDatasetConfig())
