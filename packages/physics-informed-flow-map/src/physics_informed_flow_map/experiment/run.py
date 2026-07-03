"""Run lifecycle backed by Weights & Biases with local mirror logging.

``start_run`` opens a wandb run (config = resolved experiment config + git/env
metadata) and prepares a local ``checkpoints/`` dir inside the Hydra-provided run
directory. :class:`Run` streams scalars (:meth:`log`), images (:meth:`log_image`),
and model checkpoints/artifacts to it; :meth:`finish` records summary scalars in the
run summary.

Every metric that goes to wandb is also mirrored to the network-volume run directory:
  - ``config.json``   — full config + env snapshot written at run start
  - ``metrics.jsonl`` — one JSON line per :meth:`log` call (epoch-level scalars)
  - ``steps.jsonl``   — one JSON line per optimizer step via :meth:`log_step`
  - ``summary.json``  — final summary scalars written by :meth:`finish`
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib
import torch
import wandb

matplotlib.use("Agg")

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
    _metrics_fh: Any = field(default=None, repr=False)  # open file handle for metrics.jsonl
    _steps_fh: Any = field(default=None, repr=False)    # open file handle for steps.jsonl

    @property
    def log_dir(self) -> Path:
        """The run directory (parent of ``checkpoints/``); all local logs live here."""
        return self.ckpt_dir.parent

    def _append_jsonl(self, fh: Any, record: dict[str, Any]) -> None:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()

    def log(self, **metrics: Any) -> None:
        """Log scalars to wandb on the ``epoch`` axis (set as the step metric in ``start_run``).

        Any ``step`` key is dropped — we plot against ``epoch``, not the raw optimizer step.
        Also appends to ``metrics.jsonl`` in the local run directory.
        """
        metrics.pop("step", None)
        epoch = metrics.get("epoch")
        if epoch is not None:
            self._last_epoch = int(epoch)
        self.run.log(metrics)
        if self._metrics_fh is not None:
            self._append_jsonl(self._metrics_fh, {"_ts": time.time(), **metrics})

    def log_step(self, **metrics: Any) -> None:
        """Log per-optimizer-step scalars to ``steps.jsonl`` only (not wandb).

        Intended for high-frequency step-level data (loss, grad_norm, lr per batch)
        that would be too noisy to stream to wandb but useful for offline analysis.
        """
        if self._steps_fh is not None:
            self._append_jsonl(self._steps_fh, {"_ts": time.time(), **metrics})

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

        Local checkpoints rotate: each save deletes the superseded periodic pair (and, on
        a new best, the superseded best pair), so disk holds at most the latest periodic,
        the latest best, and the final pair. The full version history lives in the wandb
        artifacts.
        """
        rotate: dict[str, list[Path]] = {"periodic": [], "best": []}

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
            paths = [path]
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
                paths.append(ema_path)

            # Rotate only after the new save landed; a best/final save at the same epoch
            # reuses the periodic save's filename, so never unlink a just-written path.
            stale = rotate["periodic"] + (rotate["best"] if is_best else [])
            for old in stale:
                if old not in paths:
                    old.unlink(missing_ok=True)
            rotate["periodic"] = [] if (is_best or is_final) else paths
            if is_best:
                rotate["best"] = paths

        return on_checkpoint

    def make_step_saver(
        self,
        key: str,
        viz_fn: Callable[[Any, Path], None],
        *,
        total_steps: int,
        n_frames: int = 3,
    ) -> Callable[..., None]:
        """Return an ``on_step(step, x_est, **scalars)`` callback for sampler trajectory logging.

        Saves ``n_frames`` evenly-spaced snapshots across the sampler trajectory, both to disk
        (``log_dir/trajectory/{key}/step_{i:04d}.png``) and to W&B via :meth:`log_image`.
        Any ``**scalars`` (data_fidelity, grad_norm, …) are written to ``steps.jsonl`` on
        every call regardless of the frame schedule.

        Args:
            key: sub-directory name / W&B image key, e.g. ``"dps_guided"``.
            viz_fn: ``(x_est_tensor, out_path)`` — callable that saves a PNG to ``out_path``.
            total_steps: total number of sampler steps (needed to space frames evenly).
            n_frames: how many snapshots to save (default 3: start, mid, end).
        """
        traj_dir = self.log_dir / "trajectory" / key
        traj_dir.mkdir(parents=True, exist_ok=True)
        if n_frames <= 1:
            checkpoints: set[int] = {0}
        else:
            checkpoints = {
                round(i * (total_steps - 1) / (n_frames - 1)) for i in range(n_frames)
            }

        def on_step(step: int, x_est: Any, **scalars: float) -> None:
            if scalars:
                self.log_step(phase=key, step=step, **scalars)
            if step in checkpoints:
                path = traj_dir / f"step_{step:04d}.png"
                viz_fn(x_est, path)
                self.log_image(f"traj/{key}", path, caption=f"step {step}/{total_steps}")

        return on_step

    def finish(self, **summary: Any) -> None:
        """Record summary scalars to the wandb run summary, write ``summary.json``, and close."""
        for key, value in summary.items():
            self.run.summary[key] = value
        extra = " ".join(f"{key}={value}" for key, value in summary.items())
        print(f"[{self.experiment}] {extra}".rstrip())
        # Write summary.json before closing so it's always on disk even if wandb upload fails.
        summary_path = self.log_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"_ts": time.time(), **summary}, default=str, indent=2)
        )
        # Close the step/metrics file handles before finishing wandb.
        for fh in (self._metrics_fh, self._steps_fh):
            if fh is not None:
                fh.close()
        self.run.finish()


def start_run(
    experiment: str,
    run_dir: Path,
    config: dict[str, Any],
    *,
    project: str | None = None,
    name: str | None = None,
) -> Run:
    """Open a wandb run and prepare ``run_dir/checkpoints/``.

    Each ``experiment`` gets its own wandb project by default (override with ``project``).
    ``run_dir`` is the Hydra run directory; ``config`` is ``Config.dump()``. Connectivity is
    wandb-native via ``WANDB_MODE`` (default online).
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    full_config = {**config, **_env()}

    # Write config.json to the run directory (on the network volume) before wandb init so it
    # survives even if the wandb upload fails.
    (run_dir / "config.json").write_text(
        json.dumps({"_ts": time.time(), **full_config}, default=str, indent=2)
    )

    run = wandb.init(
        project=project or f"{DEFAULT_PROJECT}-{experiment}",
        name=name,
        group=experiment,
        dir=str(run_dir),
        config=full_config,
    )
    # Plot every metric against epoch instead of the raw optimizer step.
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    print(f"[{experiment}] run → {run_dir}")

    metrics_fh = open(run_dir / "metrics.jsonl", "a")  # noqa: SIM115
    steps_fh = open(run_dir / "steps.jsonl", "a")      # noqa: SIM115
    return Run(
        run=run,
        experiment=experiment,
        ckpt_dir=ckpt_dir,
        _metrics_fh=metrics_fh,
        _steps_fh=steps_fh,
    )
