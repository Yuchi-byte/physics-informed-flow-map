"""Run lifecycle: a manifest-pinned output directory under the git-ignored ``runs/``.

``start_run`` creates ``runs/<experiment>/<UTC-stamp>/`` and writes ``manifest.json``
(argv, resolved config, git commit + dirty-diff digest, python/torch/cuda/gpu env).
:meth:`Run.log` appends one JSON record per call to ``metrics.jsonl``; :meth:`Run.finish`
writes ``result.json`` with the verdict. Every run is reproducible from its manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _repo_root() -> Path:
    root = _git("rev-parse", "--show-toplevel")
    return Path(root) if root else Path.cwd()


@dataclass
class Run:
    """A live experiment run. Owns its output directory and streams metrics to it."""

    dir: Path
    experiment: str
    _step: int = field(default=0, repr=False)

    def log(self, **metrics: Any) -> None:
        """Append one record (auto step counter + wall time) to ``metrics.jsonl``."""
        record = {"step": self._step, "time": time.time(), **metrics}
        with (self.dir / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        self._step += 1

    def finish(self, verdict: str, **summary: Any) -> None:
        """Write ``result.json`` with the verdict and any summary scalars."""
        result = {"verdict": verdict, **summary}
        (self.dir / "result.json").write_text(json.dumps(result, indent=2))
        print(f"\n[{self.experiment}] verdict={verdict}  →  {self.dir}")


def start_run(experiment_dir: Path, config: dict[str, Any]) -> Run:
    """Create ``runs/<experiment>/<UTC-stamp>/`` and pin a manifest.

    ``experiment_dir`` is the framework directory (``__file__``'s parent); its name
    (e.g. ``0001_mnist_pipeline``) names the run group. ``config`` is the resolved
    config dict (``Config.dump()``).
    """
    experiment = Path(experiment_dir).name
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir = _repo_root() / "runs" / experiment / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    diff = _git("diff", "HEAD")
    manifest = {
        "experiment": experiment,
        "started": stamp,
        "argv": sys.argv,
        "config": config,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_diff_sha256": sha256(diff.encode()).hexdigest() if diff else "",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{experiment}] run → {run_dir}")
    return Run(dir=run_dir, experiment=experiment)
