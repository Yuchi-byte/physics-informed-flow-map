"""0004 prior groups pin their checkpoint alias and backbone shape.

``prior=<name>`` must resolve the ckpt and the DiT shape together — they have to agree, and
splitting them across the CLI is how you load a 768/12/12 checkpoint into a 256/6/8 model.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "0004_inversion"
CONF = EXP / "conf"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inversion_run", EXP / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Pydantic resolves InversionConfig's forward refs (EvalEntry) via sys.modules; without
    # registering first, model_rebuild() leaves EvalConfig undefined and from_dictconfig raises.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _compose(overrides: list[str]) -> Any:
    cfg_cls = _load_run_module().InversionConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=overrides)
    return cfg_cls.from_dictconfig(dcfg)


@pytest.mark.parametrize(
    "prior,method,ckpt",
    [
        ("flow_matching", "flow_tilt", "checkpoints/0001_flow_matching_openfwi.pt"),
        ("flow_map", "mfm_g", "checkpoints/0002_flow_map_openfwi.pt"),
        ("diffusion", "dps", "checkpoints/0003_diffusion_openfwi.pt"),
    ],
)
def test_prior_group_pins_ckpt_and_shape(prior: str, method: str, ckpt: str) -> None:
    cfg = _compose([f"prior={prior}", f"method={method}"])
    assert cfg.ckpt == ckpt
    assert (cfg.model.hidden, cfg.model.depth, cfg.model.num_heads) == (768, 12, 12)
    assert cfg.model.patch_size == 4  # ModelConfig default survives the group merge


def test_diffusion_prior_defaults_to_dit_denoiser() -> None:
    cfg = _compose(["prior=diffusion", "method=dps"])
    assert cfg.prior.denoiser_kind == "dit"
    assert cfg.prior.num_train_timesteps == 1000


def test_prior_none_pins_no_ckpt() -> None:
    cfg = _compose(["prior=none", "method=classical_fwi"])
    assert cfg.ckpt == ""


def test_smoke_still_runs_an_untrained_prior() -> None:
    # experiment groups load after prior, so smoke's ckpt: "" blanks the alias.
    cfg = _compose(["experiment=smoke"])
    assert cfg.ckpt == ""


def test_flatfault_keeps_its_320_8_backbone() -> None:
    cfg = _compose(["experiment=flatfault"])
    assert (cfg.model.hidden, cfg.model.depth) == (320, 8)
    assert cfg.ckpt == ""


def test_cli_ckpt_override_beats_the_alias() -> None:
    cfg = _compose(["prior=diffusion", "method=dps", "ckpt=runs/legacy/step_9.pt"])
    assert cfg.ckpt == "runs/legacy/step_9.pt"
