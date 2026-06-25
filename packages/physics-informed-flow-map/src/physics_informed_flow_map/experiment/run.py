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
from typing import Any

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

    def log(self, **metrics: Any) -> None:
        """Log scalars at ``metrics['step']`` (if present) to wandb."""
        step = metrics.pop("step", None)
        self.run.log(metrics, step=int(step) if step is not None else None)

    def log_image(self, key: str, path: Path, *, step: int | None = None) -> None:
        """Log an image file under ``key`` to wandb."""
        self.run.log({key: wandb.Image(str(path))}, step=step)

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
    print(f"[{experiment}] run → {run_dir}")
    return Run(run=run, experiment=experiment, ckpt_dir=ckpt_dir)
