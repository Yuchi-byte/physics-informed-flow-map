# wandb Tracking + Hydra Configs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the harness's local-file tracking with Weights & Biases (decomposed losses, periodic eval images, checkpoints + artifacts) and adopt Hydra (`@hydra.main` + yaml config groups validated into pydantic) harness-wide.

**Architecture:** `run.py` entry points become `@hydra.main` functions that compose a `DictConfig` from `conf/` and validate it into the existing pydantic `Config` (keeps `extra="forbid"`). The harness `Run` wraps a wandb run; `train()` gains pure default-off callbacks for eval/checkpointing so the loop stays I/O-free. All wandb/disk wiring lives in the entry point.

**Tech Stack:** Python 3.12, PyTorch, mfm (reference package, do not edit), Hydra (`hydra-core>=1.3.2`), OmegaConf, pydantic v2, Weights & Biases (`wandb>=0.23.0`), pytest.

## Global Constraints

- **wandb is the only tracking backend.** Do not write `manifest.json`, `metrics.jsonl`, or `result.json`. git/env/config → `wandb.config`; metrics → `wandb.log`; verdict → `wandb.run.summary` (and a console print).
- **All tests run under `WANDB_MODE=disabled`** (autouse fixture per test file) — no network, no wandb files.
- **Config flow is Hydra → pydantic:** compose a `DictConfig`, then `Config.from_dictconfig(cfg)` validates with `extra="forbid"`. A typo'd key is a `ValidationError`, never a silent no-op.
- **Output dir:** `hydra.run.dir: runs/<framework>/${now:%Y-%m-%dT%H-%M-%SZ}` and `hydra.job.chdir: false`. `start_run` receives the run dir from `HydraConfig.get().runtime.output_dir` — it does **not** compute its own stamp and stays Hydra-free.
- **`start_run` signature:** `start_run(experiment: str, run_dir: Path, config: dict, *, project: str = "physics-informed-flow-map", name: str | None = None) -> Run`.
- **Checkpoints are local** under `run_dir/checkpoints/`. A `final` checkpoint+artifact always saves; `best` only when periodic eval is enabled (it needs a metric). Resume-from-checkpoint is out of scope.
- **Do not edit reference packages** (`packages/mfm-meta-flow-map-main/` etc.). Reuse mfm through its public interface only.
- `wandb>=0.23.0` and `hydra-core>=1.3.2` are already declared in `packages/physics-informed-flow-map/pyproject.toml` — no dependency edits required, only mypy overrides.
- DRY, YAGNI, TDD, frequent commits. mypy strict must stay clean; ruff check + format clean.

---

### Task 1: Hydra→pydantic config bridge

Replace `Config.resolve` (the old OmegaConf-merge entry point) with `Config.from_dictconfig`, which validates a Hydra-composed `DictConfig` into the typed schema. This is the seam every `run.py` will use.

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/config.py`
- Test: `packages/physics-informed-flow-map/tests/test_config.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Config.from_dictconfig(cfg: DictConfig) -> T` (generic over the `Config` subclass — returns the precise subclass instance), and `Config.dump() -> dict[str, Any]` (unchanged).

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_config.py`:

```python
"""Hydra-composed DictConfig -> typed pydantic Config, with strict validation."""

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from physics_informed_flow_map.experiment import Config


class _Demo(Config):
    a: int = 1
    b: str = "x"


def test_from_dictconfig_returns_subclass_instance() -> None:
    cfg = OmegaConf.create({"a": 5, "b": "y"})
    out = _Demo.from_dictconfig(cfg)
    assert isinstance(out, _Demo)
    assert out.a == 5
    assert out.b == "y"


def test_from_dictconfig_rejects_unknown_key() -> None:
    cfg = OmegaConf.create({"a": 5, "b": "y", "typo": 1})
    with pytest.raises(ValidationError):
        _Demo.from_dictconfig(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_config.py -v`
Expected: FAIL with `AttributeError: type object '_Demo' has no attribute 'from_dictconfig'`.

- [ ] **Step 3: Implement `from_dictconfig`, drop `resolve`**

Replace the body of `config.py` with (keep the module docstring; update it to describe Hydra composition):

```python
"""Typed experiment configuration: a pydantic schema validated from Hydra output.

Each experiment framework subclasses :class:`Config`, declaring its knobs as typed
fields. A ``run.py`` entry point composes a Hydra ``DictConfig`` from its ``conf/``
tree, then calls :meth:`Config.from_dictconfig` to validate it into the schema.
Unknown keys are rejected (``extra="forbid"``), so a typo'd override fails loudly
instead of being silently ignored.
"""

from __future__ import annotations

from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="Config")


class Config(BaseModel):
    """Base for experiment configs. Validates strictly; serialises round-trippably."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dictconfig(cls: type[T], cfg: DictConfig) -> T:
        """Validate a Hydra-composed ``DictConfig`` into this typed schema.

        Resolves interpolations, converts to a plain container, then validates.
        ``extra="forbid"`` turns any key not declared on the subclass into a
        ``ValidationError``.
        """
        container = OmegaConf.to_container(cfg, resolve=True)
        return cls.model_validate(container)

    def dump(self) -> dict[str, Any]:
        """JSON-ready dict of the resolved config, pinned into the wandb run."""
        return self.model_dump(mode="json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify mypy clean**

Run: `uv run mypy packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/config.py`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/config.py packages/physics-informed-flow-map/tests/test_config.py
git commit -m "feat(harness): Config.from_dictconfig (Hydra->pydantic), drop resolve"
```

---

### Task 2: wandb-backed harness `Run`

Rewrite `experiment/run.py` so `start_run` opens a wandb run and `Run` streams scalars, images, checkpoints, and artifacts to it. No local json files.

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py`
- Modify: `pyproject.toml` (root — mypy `ignore_missing_imports` overrides)
- Test: `packages/physics-informed-flow-map/tests/test_run.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `start_run(experiment: str, run_dir: Path, config: dict, *, project: str = "physics-informed-flow-map", name: str | None = None) -> Run`
  - `Run.log(**metrics: Any) -> None` (pops `step`, forwards to `wandb.log`)
  - `Run.log_image(key: str, path: Path, *, step: int | None = None) -> None`
  - `Run.save_checkpoint(model: torch.nn.Module, step: int, **meta: Any) -> Path`
  - `Run.log_artifact(path: Path, *, name: str, aliases: list[str]) -> None`
  - `Run.finish(verdict: str, **summary: Any) -> None`
  - `Run.ckpt_dir: Path`

- [ ] **Step 1: Add mypy overrides for wandb + hydra**

In the root `pyproject.toml`, extend the existing override module list (add `wandb.*` and `hydra.*`; hydra is added now to avoid re-touching this list in Task 4):

```toml
[[tool.mypy.overrides]]
module = ["mfm.*", "diffusers.*", "torchvision.*", "matplotlib.*", "torchdiffeq.*", "wandb.*", "hydra.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_run.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_run.py -v`
Expected: FAIL (current `start_run` has signature `(experiment_dir, config)` and `Run` lacks `ckpt_dir`/`log_image`/`save_checkpoint`/`log_artifact`).

- [ ] **Step 4: Rewrite `run.py`**

Replace the entire contents of `experiment/run.py`:

```python
"""Run lifecycle backed by Weights & Biases.

``start_run`` opens a wandb run (config = resolved experiment config + git/env
metadata) and prepares a local ``checkpoints/`` dir inside the Hydra-provided run
directory. :class:`Run` streams scalars (:meth:`log`), images (:meth:`log_image`),
and model checkpoints/artifacts to it; :meth:`finish` records the verdict in the
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
        self, model: torch.nn.Module, step: int, **meta: Any
    ) -> Path:
        """Save ``model`` state (+ metadata) to ``checkpoints/step_<step>.pt``."""
        path = self.ckpt_dir / f"step_{step}.pt"
        torch.save({"model": model.state_dict(), "step": step, **meta}, path)
        return path

    def log_artifact(self, path: Path, *, name: str, aliases: list[str]) -> None:
        """Upload a checkpoint file as a wandb model artifact under ``aliases``."""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self, verdict: str, **summary: Any) -> None:
        """Record the verdict + summary scalars and close the wandb run."""
        self.run.summary["verdict"] = verdict
        for key, value in summary.items():
            self.run.summary[key] = value
        print(f"[{self.experiment}] verdict={verdict}")
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_run.py -v`
Expected: PASS (1 passed).

If `wandb.Image` or `log_artifact` raise under `disabled` mode in this wandb
version, switch the fixture to `monkeypatch.setenv("WANDB_MODE", "offline")` and
add `monkeypatch.setenv("WANDB_DIR", str(tmp_path))` — offline fully supports
images/artifacts and writes only under `tmp_path`. Prefer `disabled` if it works.

- [ ] **Step 6: Verify mypy clean**

Run: `uv run mypy packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py packages/physics-informed-flow-map/tests/test_run.py pyproject.toml
git commit -m "feat(harness): wandb-backed Run (log/image/checkpoint/artifact/finish)"
```

---

### Task 3: `train()` hooks + decomposed loss logging

Add three default-off callbacks to the training loop and log every loss term, not just `fm_loss`. The loop performs no I/O — callbacks own all side effects.

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py`
- Test: `packages/physics-informed-flow-map/tests/test_train.py` (extend; keep the existing test)

**Interfaces:**
- Consumes: nothing from earlier tasks (callbacks are caller-supplied).
- Produces (new signature):
  ```python
  def train(model, loader, *, n_steps, lr, device, num_classes=None,
            log=None,
            eval_every=0, on_eval=None,        # on_eval(model, step) -> float | None
            ckpt_every=0, on_checkpoint=None,  # on_checkpoint(model, step, *, is_best, is_final)
            ) -> list[dict[str, float]]
  ```
  Per-step `log` record contains `step`, `total`, and every key in mfm's `opt_losses` (currently `fm_loss`). `on_eval` fires when `(step+1) % eval_every == 0`; a new minimum return value marks the next checkpoint `is_best=True`. `on_checkpoint` fires on the `ckpt_every` cadence or on a new best, plus exactly once after the loop with `is_final=True`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/physics-informed-flow-map/tests/test_train.py` (the existing imports — `torch`, `DataLoader`, `DATASETS`, `build_model`, `train` — already cover these tests; add no new imports):

```python
def test_train_logs_decomposed_losses() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.make_dataset(), batch_size=32, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, mlp_width=16, mlp_depth=2)

    records: list[dict] = []
    train(
        model,
        loader,
        n_steps=3,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: records.append(r),
    )
    assert len(records) == 3
    assert "total" in records[0]
    assert "fm_loss" in records[0]


def test_train_hooks_fire_on_cadence() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.make_dataset(), batch_size=16, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, mlp_width=16, mlp_depth=2)

    evals: list[int] = []
    ckpts: list[tuple[int, bool, bool]] = []

    def on_eval(m: object, step: int) -> float:
        evals.append(step)
        return 1.0 / len(evals)  # strictly decreasing -> every eval is a new best

    def on_checkpoint(
        m: object, step: int, *, is_best: bool, is_final: bool
    ) -> None:
        ckpts.append((step, is_best, is_final))

    train(
        model,
        loader,
        n_steps=10,
        lr=1e-3,
        device=torch.device("cpu"),
        eval_every=5,
        on_eval=on_eval,
        ckpt_every=0,
        on_checkpoint=on_checkpoint,
    )

    assert evals == [4, 9]  # (step+1) % 5 == 0
    assert [c[0] for c in ckpts if c[1]] == [4, 9]  # best at each eval
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 9  # exactly one final, last step
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: FAIL — `test_train_hooks_fire_on_cadence` errors with `TypeError: train() got an unexpected keyword argument 'eval_every'`.

- [ ] **Step 3: Implement the hooks**

In `train.py`: add `import math` near the top imports, widen the `Callable` import, and replace the `train` function (the `_Cfg`/`_fm_loss_cfg` helpers above it are unchanged):

```python
def train(
    model: BaseModel,
    loader: DataLoader,
    *,
    n_steps: int,
    lr: float,
    device: torch.device,
    num_classes: int | None = None,
    log: Callable[..., None] | None = None,
    eval_every: int = 0,
    on_eval: Callable[[BaseModel, int], float | None] | None = None,
    ckpt_every: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> list[dict[str, float]]:
    label_dim = num_classes or 0
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    model.train()
    data_iter = iter(loader)
    history: list[dict[str, float]] = []
    best_metric = math.inf
    for step in range(n_steps):
        try:
            x1, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1, labels = next(data_iter)
        x1 = x1.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        opt_losses, _ = loss_fn(model, None, x1, labels, step=step)
        total = sum(opt_losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rec: dict[str, float] = {"step": float(step), "total": float(total.item())}
        for name, value in opt_losses.items():
            rec[name] = float(value.item())
        history.append(rec)
        if log is not None:
            log(**rec)

        is_best = False
        if on_eval is not None and eval_every and (step + 1) % eval_every == 0:
            model.eval()
            metric = on_eval(model, step)
            model.train()
            if metric is not None and metric < best_metric:
                best_metric = metric
                is_best = True

        if on_checkpoint is not None and (
            is_best or (ckpt_every and (step + 1) % ckpt_every == 0)
        ):
            on_checkpoint(model, step, is_best=is_best, is_final=False)

    if on_checkpoint is not None:
        on_checkpoint(model, n_steps - 1, is_best=False, is_final=True)
    return history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: PASS (3 passed — the original `test_train_runs_and_logs` plus the two new ones).

- [ ] **Step 5: Verify mypy clean**

Run: `uv run mypy packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py`
Expected: `Success: no issues found`.

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py packages/physics-informed-flow-map/tests/test_train.py
git commit -m "feat(train): default-off eval/checkpoint hooks + decomposed loss logging"
```

---

### Task 4: 0001 Hydra config tree + `@hydra.main` entry point

Convert `0001_flow_matching` to Hydra: a `conf/` tree of yaml config groups, and a `run.py` that composes → validates → runs with the new wandb harness and train hooks.

**Files:**
- Create: `experiments/0001_flow_matching/conf/config.yaml`
- Create: `experiments/0001_flow_matching/conf/experiment/gaussians.yaml`
- Create: `experiments/0001_flow_matching/conf/experiment/mnist.yaml`
- Create: `experiments/0001_flow_matching/conf/experiment/smoke.yaml`
- Modify: `experiments/0001_flow_matching/run.py` (full rewrite)
- Test: `packages/physics-informed-flow-map/tests/test_experiment_conf.py` (create)

**Interfaces:**
- Consumes: `Config.from_dictconfig` (Task 1); `start_run`, `Run` methods (Task 2); `train(..., eval_every, on_eval, ckpt_every, on_checkpoint)` (Task 3); existing `DATASETS`, `build_model`, `sample`, `energy_distance`, `real_reference`, `DatasetSpec.visualize`.
- Produces: `FlowMatchingConfig` (pydantic `Config` subclass) and a `@hydra.main` `main()` in `experiments/0001_flow_matching/run.py`.

- [ ] **Step 1: Create the Hydra config tree**

`experiments/0001_flow_matching/conf/config.yaml`:

```yaml
defaults:
  - _self_
  - experiment: gaussians

seed: 0
dataset: gaussians
n_steps: 2000
batch_size: 256
lr: 0.001
sampler_steps: 100
n_eval_samples: 2048
gate: 0.5
mlp_width: 256
mlp_depth: 4
dit_hidden: 128
dit_depth: 4
eval_every: 0
ckpt_every: 0
artifact_every: 0
n_eval_viz: 64

hydra:
  run:
    dir: runs/0001_flow_matching/${now:%Y-%m-%dT%H-%M-%SZ}
  job:
    chdir: false
```

`experiments/0001_flow_matching/conf/experiment/gaussians.yaml`:

```yaml
# @package _global_
dataset: gaussians
n_steps: 2000
gate: 0.5
```

`experiments/0001_flow_matching/conf/experiment/mnist.yaml`:

```yaml
# @package _global_
dataset: mnist
n_steps: 3000
batch_size: 128
sampler_steps: 50
gate: 240.0
```

`experiments/0001_flow_matching/conf/experiment/smoke.yaml`:

```yaml
# @package _global_
dataset: gaussians
n_steps: 20
n_eval_samples: 256
gate: 1000000000.0
```

- [ ] **Step 2: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_experiment_conf.py`:

```python
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
    [("gaussians", "gaussians", 2000), ("mnist", "mnist", 3000), ("smoke", "gaussians", 20)],
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -v`
Expected: FAIL — the current `run.py` has no `FlowMatchingConfig` with these fields / uses `VARIANTS` + `resolve` (the module exec or `from_dictconfig` call errors).

- [ ] **Step 4: Rewrite `run.py`**

Replace the entire contents of `experiments/0001_flow_matching/run.py`:

```python
"""Train flow matching on swappable datasets (2D Gaussians, MNIST) via Hydra.

    uv run python experiments/0001_flow_matching/run.py                       # gaussians
    uv run python experiments/0001_flow_matching/run.py experiment=mnist
    uv run python experiments/0001_flow_matching/run.py experiment=smoke
    uv run python experiments/0001_flow_matching/run.py experiment=mnist eval_every=500 ckpt_every=1000

Verdict: gaussians → energy distance < gate; mnist → final FM loss < gate.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from mfm.models.base_model import BaseModel
from omegaconf import DictConfig
from pydantic import Field

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)
from physics_informed_flow_map.flow_matching.train import train

EXPERIMENT = "0001_flow_matching"


class FlowMatchingConfig(Config):
    seed: int = 0
    dataset: str = "gaussians"
    n_steps: int = Field(2000, gt=0)
    batch_size: int = 256
    lr: float = 1e-3
    sampler_steps: int = Field(100, gt=0)
    n_eval_samples: int = Field(2048, gt=0)
    gate: float = 0.5
    mlp_width: int = 256
    mlp_depth: int = 4
    dit_hidden: int = 128
    dit_depth: int = 4
    eval_every: int = Field(0, ge=0)
    ckpt_every: int = Field(0, ge=0)
    artifact_every: int = Field(0, ge=0)
    n_eval_viz: int = Field(64, gt=0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = FlowMatchingConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, FlowMatchingConfig)

    spec = DATASETS[cfg.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())

    dataset = spec.make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    model = build_model(
        spec.shape,
        spec.num_classes,
        mlp_width=cfg.mlp_width,
        mlp_depth=cfg.mlp_depth,
        dit_hidden=cfg.dit_hidden,
        dit_depth=cfg.dit_depth,
    ).to(device)

    def on_eval(m: BaseModel, step: int) -> float | None:
        samples = sample(
            m, cfg.n_eval_viz, spec.shape, sampler_steps=cfg.sampler_steps, device=device
        )
        path = run.ckpt_dir.parent / f"samples_{step}.png"
        spec.visualize(samples, path)
        run.log_image("samples", path, step=step)
        if cfg.dataset == "gaussians":
            ref = real_reference(dataset, cfg.n_eval_viz, device)
            return energy_distance(samples, ref)
        return None

    def on_checkpoint(
        m: BaseModel, step: int, *, is_best: bool = False, is_final: bool = False
    ) -> None:
        path = run.save_checkpoint(m, step, dataset=cfg.dataset, config=cfg.dump())
        aliases: list[str] = []
        if is_final:
            aliases.append("final")
        if is_best:
            aliases.append("best")
        if cfg.artifact_every and (step + 1) % cfg.artifact_every == 0:
            aliases.append("periodic")
        if aliases:
            run.log_artifact(path, name=f"{cfg.dataset}-model", aliases=aliases)

    history = train(
        model,
        loader,
        n_steps=cfg.n_steps,
        lr=cfg.lr,
        device=device,
        num_classes=spec.num_classes,
        log=run.log,
        eval_every=cfg.eval_every,
        on_eval=on_eval,
        ckpt_every=cfg.ckpt_every,
        on_checkpoint=on_checkpoint,
    )
    final_loss = history[-1]["total"]

    samples = sample(
        model, cfg.n_eval_samples, spec.shape, sampler_steps=cfg.sampler_steps, device=device
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    spec.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    if cfg.dataset == "gaussians":
        ref = real_reference(dataset, cfg.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        verdict = "pass" if metric < cfg.gate else "fail"
        run.finish(verdict, energy_distance=metric, final_loss=final_loss, gate=cfg.gate)
    else:
        verdict = "pass" if final_loss < cfg.gate else "fail"
        run.finish(verdict, final_loss=final_loss, gate=cfg.gate)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -v`
Expected: PASS (4 passed — 3 parametrized + 1 override).

- [ ] **Step 6: Smoke-run the entry point end-to-end (disabled wandb)**

Run:
```bash
WANDB_MODE=disabled uv run python experiments/0001_flow_matching/run.py experiment=smoke
```
Expected: prints `[0001_flow_matching] run → runs/0001_flow_matching/<stamp>` and
`[0001_flow_matching] verdict=pass`; a `runs/0001_flow_matching/<stamp>/checkpoints/step_19.pt`
exists (final checkpoint always saved). No `manifest.json`/`metrics.jsonl`/`result.json`.

- [ ] **Step 7: Verify mypy clean (enforced scope)**

Run: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests`
Expected: `Success: no issues found`. This matches the pre-commit mypy hook scope.
`experiments/` is intentionally **not** under strict mypy — Hydra's `@hydra.main`
is an untyped decorator, which `disallow_untyped_decorators` (strict) would flag;
the entry point is covered by ruff (lint + format) and the compose/smoke tests
instead.

- [ ] **Step 8: Commit**

```bash
git add experiments/0001_flow_matching/ packages/physics-informed-flow-map/tests/test_experiment_conf.py
git commit -m "feat(0001): Hydra @hydra.main entry point + conf groups; wandb + train hooks"
```

---

### Task 5: Harness-wide docs + scaffolder

Make the Hydra+wandb pattern the documented convention: update the experiments contract, the `new.py` scaffolder template, and the stale CLAUDE.md run examples.

**Files:**
- Modify: `experiments/README.md`
- Modify: `experiments/new.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the pattern established in Task 4 (`conf/` tree + `@hydra.main` run.py).
- Produces: a scaffolder that emits a Hydra-shaped framework, and docs matching it.

- [ ] **Step 1: Rewrite the `new.py` scaffolder templates**

In `experiments/new.py`, replace `RUN_STUB` and `REPORT_STUB`, and update `main()` to also write the `conf/` tree. Replace from the `RUN_STUB = '''...'''` line through the end of the file with:

```python
RUN_STUB = '''\
"""{title}

    uv run python experiments/{dirname}/run.py                    # default
    uv run python experiments/{dirname}/run.py experiment=smoke
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field

from physics_informed_flow_map.experiment import Config, start_run

EXPERIMENT = "{dirname}"


class {cls}(Config):
    seed: int = 0
    # TODO: declare typed knobs here.
    gate: float = Field(0.0)  # verdict threshold, asserted in code


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = {cls}.from_dictconfig(dcfg)
    assert isinstance(cfg, {cls})

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())
    # TODO: do the work; call run.log(**metrics) per step.
    run.finish("todo")


if __name__ == "__main__":
    main()
'''

CONFIG_STUB = '''\
defaults:
  - _self_
  - experiment: default

seed: 0
gate: 0.0

hydra:
  run:
    dir: runs/{dirname}/${{now:%Y-%m-%dT%H-%M-%SZ}}
  job:
    chdir: false
'''

EXPERIMENT_DEFAULT_STUB = """\
# @package _global_
# TODO: default-variant overrides.
"""

EXPERIMENT_SMOKE_STUB = """\
# @package _global_
# Trivial budget for a fast end-to-end plumbing check (no strength claim).
gate: 1000000000.0
"""

REPORT_STUB = """\
# {number} — {title}

Status: open

## Hypothesis

<one sentence to be proven>

## Setup

`run.py [experiment=<variant>] [key=value ...]` — config, loop steps, stack.

## Results

Cite run directories under `runs/{dirname}/`; quote numbers + the verdict from the
wandb run (config / metrics / summary). Checkpoints live in `<run>/checkpoints/`.

## Decision

Adopted / Falsified / Parked; what changes. Mirror the verdict line to
`../JOURNAL.md`.
"""


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        sys.exit('usage: uv run python experiments/new.py "short title"')

    title = " ".join(sys.argv[1:]).strip()
    number = next_number()
    dirname = f"{number:04d}_{slugify(title)}"
    cls = "".join(w.capitalize() for w in slugify(title).split("_")) + "Config"

    target = EXPERIMENTS / dirname
    if target.exists():
        sys.exit(f"refusing to overwrite existing {target}")
    (target / "conf" / "experiment").mkdir(parents=True)

    (target / "run.py").write_text(
        RUN_STUB.format(title=title, dirname=dirname, cls=cls)
    )
    (target / "conf" / "config.yaml").write_text(CONFIG_STUB.format(dirname=dirname))
    (target / "conf" / "experiment" / "default.yaml").write_text(EXPERIMENT_DEFAULT_STUB)
    (target / "conf" / "experiment" / "smoke.yaml").write_text(EXPERIMENT_SMOKE_STUB)
    (target / "report.md").write_text(
        REPORT_STUB.format(number=f"{number:04d}", title=title, dirname=dirname)
    )
    print(f"scaffolded experiments/{dirname}/")
    print(f"  edit experiments/{dirname}/run.py and conf/")


if __name__ == "__main__":
    main()
```

Note the doubled braces `${{now:...}}` in `CONFIG_STUB` — it is a `str.format`
template, so the literal `${now:...}` interpolation must be escaped.

- [ ] **Step 2: Verify the scaffolder produces a composable framework**

Run:
```bash
uv run python experiments/new.py "scaffold smoke check"
ls -R experiments/0002_scaffold_smoke_check
WANDB_MODE=disabled uv run python experiments/0002_scaffold_smoke_check/run.py experiment=smoke
```
Expected: the tree has `run.py`, `report.md`, `conf/config.yaml`,
`conf/experiment/{default,smoke}.yaml`; the run prints `verdict=todo` and exits 0.
Then remove the throwaway: `rm -rf experiments/0002_scaffold_smoke_check`.

- [ ] **Step 3: Rewrite the `experiments/README.md` results + anatomy + running sections**

Replace the "Anatomy of a framework", "Where results land", and "Running" sections so they read:

````markdown
## Anatomy of a framework

- **`run.py`** is a `@hydra.main` entry point. It declares a typed `Config`
  subclass (`physics_informed_flow_map.experiment.Config`), composes a
  `DictConfig` from `conf/`, validates it via `Config.from_dictconfig(cfg)`
  (strict — `extra="forbid"`, a typo'd override is an error), then drives the run.
- **`conf/`** holds the Hydra config: `config.yaml` (base defaults + the `hydra`
  block) and `experiment/*.yaml` config groups (variants, each starting with
  `# @package _global_`). Select a variant with `experiment=<name>`.
- The run lifecycle is owned by the harness: `start_run(experiment, run_dir, config)`
  → `run.log(**metrics)` per step → `run.finish(verdict, **summary)`. Tracking goes
  to Weights & Biases; checkpoints are saved locally.

## Where results land

Tracking (config, metrics, sample images, verdict) goes to **Weights & Biases**.
Local artifacts land in the git-ignored `runs/` at the repo root:

```
runs/<framework>/<UTC-stamp>/        # = hydra.run.dir
├── .hydra/config.yaml               # Hydra's composed-config snapshot
├── checkpoints/step_<N>.pt          # local checkpoints (final always saved)
└── samples*.png                     # eval images (also logged to wandb)
```

The verdict is recorded in the wandb run summary and printed to the console.
wandb captures the git commit + a diff patch natively, so runs stay reproducible.

## Running

```bash
WANDB_MODE=online uv run python experiments/NNNN_slug/run.py [experiment=<variant>] [key=value ...]
```

Examples:

```bash
uv run python experiments/0001_flow_matching/run.py experiment=smoke
uv run python experiments/0001_flow_matching/run.py experiment=mnist eval_every=500 ckpt_every=1000
```
````

- [ ] **Step 4: Refresh the CLAUDE.md run examples**

In `CLAUDE.md`, replace the experiment run example block (currently citing
`experiments/0001_mnist_pipeline/run.py smoke` and `... default n_steps=500`) with:

```bash
uv run python experiments/0001_flow_matching/run.py experiment=smoke        # fast plumbing check
uv run python experiments/0001_flow_matching/run.py experiment=mnist n_steps=500
uv run python experiments/new.py "short title"                              # scaffold a new framework
```

- [ ] **Step 5: Run the full test suite + linters**

Run:
```bash
uv run pytest packages/physics-informed-flow-map/tests -q
uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests
uv run ruff check packages/physics-informed-flow-map experiments
uv run ruff format --check packages/physics-informed-flow-map experiments
```
Expected: all green (pytest: existing + new tests pass, 1 skipped for mnist download; mypy clean; ruff clean).

- [ ] **Step 6: Commit**

```bash
git add experiments/README.md experiments/new.py CLAUDE.md
git commit -m "docs(experiments): Hydra+wandb convention in contract, scaffolder, CLAUDE.md"
```

---

## Notes for the implementer

- **Do not edit `packages/mfm-meta-flow-map-main/`** or any reference package. The
  loss/sampler/model come from mfm's public interface only.
- The pre-commit hooks (`ruff check --fix`, `ruff format`, `mypy`, `pytest`) run on
  commit; keep every commit green. If a hook reformats files, re-stage and re-commit.
- `wandb` calls must never run online in tests — the autouse `WANDB_MODE=disabled`
  fixture is mandatory in `test_run.py`. Other test files don't touch wandb.
- There is intentional duplication between `conf/config.yaml` defaults and the
  pydantic field defaults: the yaml is Hydra's source of overridable keys, pydantic
  is the validation schema. This is the accepted cost of "yaml + pydantic validation"
  (structured-config registration was explicitly out of scope).
```
