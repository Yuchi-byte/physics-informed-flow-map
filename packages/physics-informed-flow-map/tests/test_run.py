"""Harness Run lifecycle against a disabled wandb backend (no network/files)."""

from pathlib import Path

import pytest
import torch
from torchvision.utils import save_image

from physics_informed_flow_map.experiment.run import start_run


@pytest.fixture(autouse=True)
def _disable_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")


def test_run_lifecycle(tmp_path: Path) -> None:
    run = start_run("test_exp", tmp_path, {"lr": 0.1})
    assert run.ckpt_dir == tmp_path / "checkpoints"
    assert run.ckpt_dir.is_dir()

    run.log(step=0, total=1.0, fm_loss=1.0)
    run.log(step=1, total=0.5, fm_loss=0.5)

    model = torch.nn.Linear(2, 2)
    path = run.save_checkpoint(model, 1, dataset="demo")
    assert path.exists()
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["step"] == 1
    assert ckpt["dataset"] == "demo"
    assert "model" in ckpt

    img = tmp_path / "img.png"
    save_image(torch.rand(3, 4, 4), str(img))
    run.log_image("samples", img, step=1)

    run.log_artifact(path, name="demo-model", aliases=["final"])
    run.finish("pass", final_loss=0.5)
