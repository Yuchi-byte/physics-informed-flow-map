"""0001 Hydra config groups compose and validate into FlowMatchingConfig."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir

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
    "variant,dataset,n_steps",
    [
        ("gaussians", "gaussians", 2000),
        ("mnist", "mnist", 3000),
        ("smoke", "gaussians", 20),
    ],
)
def test_compose_validates(variant: str, dataset: str, n_steps: int) -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=[f"experiment={variant}"])
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.dataset == dataset
    assert cfg.n_steps == n_steps


def test_compose_applies_cli_override() -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(
            config_name="config", overrides=["experiment=mnist", "n_steps=500"]
        )
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.n_steps == 500
    assert cfg.dataset == "mnist"
