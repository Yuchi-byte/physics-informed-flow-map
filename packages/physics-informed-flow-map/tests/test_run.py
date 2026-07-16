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
    # Scaffolding is lazy: nothing exists until first use.
    assert not run.ckpt_dir.exists()
    assert not (tmp_path / "metrics.jsonl").exists()
    assert not (tmp_path / "steps.jsonl").exists()

    run.log(epoch=0, total=1.0, fm_loss=1.0)
    run.log(epoch=1, total=0.5, fm_loss=0.5)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2

    model = torch.nn.Linear(2, 2)
    path = run.save_checkpoint(model, 1, dataset="demo")
    assert run.ckpt_dir.is_dir()
    assert path.exists()
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["step"] == 1
    assert ckpt["dataset"] == "demo"
    assert "model" in ckpt

    img = tmp_path / "img.png"
    save_image(torch.rand(3, 4, 4), str(img))
    run.log_image("samples", img)

    run.log_artifact(path, name="demo-model", aliases=["final"])
    run.finish(final_loss=0.5)


def test_checkpoint_callback_uploads_final_only(tmp_path: Path) -> None:
    """best checkpoints stay local-only; only final/periodic reach wandb (spec §6.1)."""
    run = start_run("test_exp", tmp_path, {"lr": 0.1})
    uploads: list[tuple[str, list[str], tuple[str, ...]]] = []
    run.log_artifact = lambda path, *, name, aliases, strip_keys=(): uploads.append(  # type: ignore[method-assign]
        (name, aliases, strip_keys)
    )
    cb = run.checkpoint_callback(
        artifact_name="demo-model", ckpt_every_epochs=0, dataset="demo", config={}
    )
    model = torch.nn.Linear(2, 2)

    cb(model, 0, is_best=True)
    assert uploads == []  # best: saved locally, never uploaded
    assert (run.ckpt_dir / "step_0.pt").exists()

    cb(model, 1, is_final=True, train_state={"optimizer": {}})
    assert [(n, a) for n, a, _ in uploads] == [("demo-model", ["final"])]
    assert uploads[0][2] == ("train_state",)  # raw upload is weights-only


def test_log_artifact_strip_keys_uploads_slim_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """strip_keys drops entries from the uploaded copy; the local file keeps them."""
    run = start_run("test_exp", tmp_path, {"lr": 0.1})
    model = torch.nn.Linear(2, 2)
    path = run.save_checkpoint(model, 1, dataset="demo", train_state={"optimizer": {}})

    uploaded_keys: list[set[str]] = []
    import wandb

    def fake_add_file(self: wandb.Artifact, p: str) -> None:
        uploaded_keys.append(set(torch.load(p, weights_only=False).keys()))

    monkeypatch.setattr(wandb.Artifact, "add_file", fake_add_file)
    run.log_artifact(
        path, name="demo-model", aliases=["final"], strip_keys=("train_state",)
    )

    assert uploaded_keys and "train_state" not in uploaded_keys[0]
    assert "model" in uploaded_keys[0]
    local = torch.load(path, weights_only=False)
    assert "train_state" in local  # local file untouched


def test_save_checkpoint_suffix(tmp_path: Path) -> None:
    run = start_run("test_exp", tmp_path, {"lr": 0.1})
    model = torch.nn.Linear(2, 2)
    path = run.save_checkpoint(model, 5, suffix="_ema", dataset="demo")
    assert path == run.ckpt_dir / "step_5_ema.pt"
    assert path.exists()
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["step"] == 5
    assert ckpt["dataset"] == "demo"
    assert "model" in ckpt


def test_make_step_saver_capture(tmp_path: Path) -> None:
    from typing import Any

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = start_run("test_exp", tmp_path, {"lr": 0.1})

    def viz(frames: torch.Tensor, path: Path) -> None:
        fig, ax = plt.subplots()
        ax.imshow(frames[0, 0].numpy())
        fig.savefig(path)
        plt.close(fig)

    cap: dict[str, Any] = {}
    cb = run.make_step_saver("k", viz, total_steps=3, n_frames=3, capture=cap)
    for step in range(3):
        cb(step, torch.rand(2, 1, 4, 4), data_fidelity=1.0 / (step + 1))

    assert cap["steps"] == [0, 1, 2]
    assert cap["frames"].shape == (3, 2, 1, 4, 4)  # (n_frames, B, C, H, W)
    assert cap["total_steps"] == 3

    # capture=None leaves behavior unchanged (renders, does not raise).
    cb2 = run.make_step_saver("k2", viz, total_steps=2, n_frames=2, capture=None)
    cb2(0, torch.rand(2, 1, 4, 4))
    cb2(1, torch.rand(2, 1, 4, 4))
    run.finish()
