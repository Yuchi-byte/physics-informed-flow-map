"""Run lifecycle backed by Weights & Biases.

``start_run`` opens a wandb run (config = resolved experiment config + git/env
metadata) and prepares a local ``checkpoints/`` dir inside the Hydra-provided run
directory. :class:`Run` streams scalars (:meth:`log`), images (:meth:`log_image`),
and model checkpoints/artifacts to it; :meth:`finish` records summary scalars in the
run summary. No local JSON is written — wandb is the single source of truth.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import wandb

DEFAULT_PROJECT = "physics-informed-flow-map"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _env() -> dict[str, Any]:
    """Reproducibility metadata folded into the wandb run config."""
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }


@dataclass
class Run:
    """A live experiment run wrapping a wandb run and a local checkpoint dir."""

    run: Any  # wandb Run handle
    experiment: str
    ckpt_dir: Path
    _last_epoch: int | None = None

    def log(self, **metrics: Any) -> None:
        """Log scalars to wandb on the ``epoch`` axis (set as the step metric in ``start_run``).

        Any ``step`` key is dropped — we plot against ``epoch``, not the raw optimizer step.
        """
        metrics.pop("step", None)
        epoch = metrics.get("epoch")
        if epoch is not None:
            self._last_epoch = int(epoch)
        self.run.log(metrics)

    def update_config(self, **values: Any) -> None:
        """Merge extra static values into the wandb run config (e.g. param counts).

        For one-off run-level facts known after ``start_run`` (where the resolved
        config is pinned) — not time series; use :meth:`log` for those.
        """
        self.run.config.update(values, allow_val_change=True)

    def log_image(
        self,
        key: str,
        path: Path,
        *,
        caption: str | None = None,
    ) -> None:
        """Log an image file under ``key`` to wandb, on the ``epoch`` axis.

        Tags the image with the most recently logged ``epoch`` so it lines up with that
        epoch's scalars. Pass ``caption`` (e.g. ``"epoch 12"``) to label the media.
        """
        payload: dict[str, Any] = {key: wandb.Image(str(path), caption=caption)}
        if self._last_epoch is not None:
            payload["epoch"] = self._last_epoch
        self.run.log(payload)

    def save_checkpoint(
        self, model: torch.nn.Module, step: int, *, suffix: str = "", **meta: Any
    ) -> Path:
        """Save ``model`` state (+ metadata) to ``checkpoints/step_<step><suffix>.pt``."""
        path = self.ckpt_dir / f"step_{step}{suffix}.pt"
        torch.save({"model": model.state_dict(), "step": step, **meta}, path)
        return path

    def log_artifact(self, path: Path, *, name: str, aliases: list[str]) -> None:
        """Upload a checkpoint file as a wandb model artifact under ``aliases``."""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases)

    def checkpoint_callback(
        self,
        *,
        artifact_name: str,
        ckpt_every_epochs: int,
        dataset: str,
        config: dict[str, Any],
    ) -> Callable[..., None]:
        """Build the ``on_checkpoint`` hook ``train_loop`` calls (raw + optional EMA).

        Saves the model locally every time, and uploads it as a wandb artifact when it
        carries an alias: ``final`` (end of run), ``best`` (new best eval metric), or
        ``periodic`` (every ``ckpt_every_epochs`` epochs). An EMA model, when present, is
        saved/uploaded alongside under ``<artifact_name>-ema``. Shared by every framework
        so the cadence/alias logic stays in one place.
        """

        def on_checkpoint(
            model: torch.nn.Module,
            epoch: int,
            *,
            is_best: bool = False,
            is_final: bool = False,
            ema_model: torch.nn.Module | None = None,
        ) -> None:
            aliases: list[str] = []
            if is_final:
                aliases.append("final")
            if is_best:
                aliases.append("best")
            if ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0:
                aliases.append("periodic")
            path = self.save_checkpoint(
                model, epoch, epoch=epoch, dataset=dataset, config=config
            )
            if aliases:
                self.log_artifact(path, name=artifact_name, aliases=aliases)
            if ema_model is not None:
                ema_path = self.save_checkpoint(
                    ema_model,
                    epoch,
                    epoch=epoch,
                    suffix="_ema",
                    dataset=dataset,
                    config=config,
                )
                if aliases:
                    self.log_artifact(
                        ema_path, name=f"{artifact_name}-ema", aliases=aliases
                    )

        return on_checkpoint

    def finish(self, **summary: Any) -> None:
        """Record summary scalars to the wandb run summary and close the run."""
        for key, value in summary.items():
            self.run.summary[key] = value
        extra = " ".join(f"{key}={value}" for key, value in summary.items())
        print(f"[{self.experiment}] {extra}".rstrip())
        self.run.finish()


def start_run(
    experiment: str,
    run_dir: Path,
    config: dict[str, Any],
    *,
    project: str = DEFAULT_PROJECT,
    name: str | None = None,
) -> Run:
    """Open a wandb run and prepare ``run_dir/checkpoints/``.

    ``experiment`` names the run group; ``run_dir`` is the Hydra run directory
    (``HydraConfig.get().runtime.output_dir``); ``config`` is ``Config.dump()``.
    Connectivity is wandb-native via ``WANDB_MODE`` (default online).
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=project,
        name=name,
        group=experiment,
        dir=str(run_dir),
        config={**config, **_env()},
    )
    # Plot every metric against epoch instead of the raw optimizer step.
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    print(f"[{experiment}] run → {run_dir}")
    return Run(run=run, experiment=experiment, ckpt_dir=ckpt_dir)
