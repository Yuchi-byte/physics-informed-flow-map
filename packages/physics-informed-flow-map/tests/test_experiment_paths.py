"""Run dirs are repo-relative, so one command works on RunPod and locally alike.

The repo sits on the RunPod network volume, so a relative ``runs/`` is as durable there
as the old absolute ``/workspace/runs`` was — and it resolves locally too.
"""

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
FRAMEWORKS = ["0001_flow_matching", "0002_flow_map", "0003_diffusion", "0004_inversion"]


def _hydra_node(framework: str) -> Any:
    raw = OmegaConf.load(REPO / "experiments" / framework / "conf" / "config.yaml")
    # resolve=False: ${now:...} has no resolver outside a running Hydra app.
    node = OmegaConf.to_container(raw, resolve=False)
    assert isinstance(node, dict)
    return node["hydra"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_run_dir_is_repo_relative(framework: str) -> None:
    run_dir = _hydra_node(framework)["run"]["dir"]
    assert run_dir.startswith(f"runs/{framework}/"), run_dir


def test_multirun_sweep_dir_is_repo_relative() -> None:
    sweep_dir = _hydra_node("0004_inversion")["sweep"]["dir"]
    assert sweep_dir.startswith("runs/0004_inversion/"), sweep_dir
