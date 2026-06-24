# Hierarchical Configs + Discriminated Unions + Epoch Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `0001_flow_matching`'s model/dataset selection into pydantic discriminated unions over a multi-group Hydra `conf/` tree, and switch the train loop to epoch-based budgets with epoch-cadence hooks + a tqdm bar.

**Architecture:** `models.py`/`datasets.py` each expose a discriminated-union config (`Annotated[A | B, Field(discriminator=...)]`); each variant owns its typed args and (for datasets) its `build()`/`shape`/`num_classes`/`visualize`. `build_model(shape, num_classes, cfg)` dispatches on type. `train()` loops over epochs with tqdm and fires hooks on epoch boundaries. `run.py` composes a nested `FlowMatchingConfig` (model/dataset/training/sampling) with a cross-validator (mlp⇔vector, dit⇔image).

**Tech Stack:** Python 3.12, PyTorch, mfm (reference package — do NOT edit), Hydra (`hydra-core`), OmegaConf, pydantic v2 (discriminated unions, `model_validator`), Weights & Biases, tqdm, pytest.

## Global Constraints

- **Both `model` and `dataset` are pydantic discriminated unions** — `ModelConfig = Annotated[MLPModelConfig | DiTModelConfig, Field(discriminator="kind")]`, `DatasetConfig = Annotated[GaussiansDatasetConfig | MNISTDatasetConfig, Field(discriminator="name")]`. Each variant subclasses the package `Config` (so `extra="forbid"` holds).
- **`requires_download` / `shape` / `num_classes` are `@property` (not fields)** on the dataset configs — intrinsic, not yaml-overridable, still readable.
- **`build_model(shape, num_classes, cfg: ModelConfig) -> BaseModel`** dispatches on `isinstance`; raises `ValueError` for mlp-on-image, dit-on-vector, and non-square dit images.
- **`train()` is epoch-based**: `n_epochs` (not `n_steps`); `eval_every_epochs`/`ckpt_every_epochs` cadence; hooks receive `epoch`; per-epoch `tqdm` bar; each per-batch `log` record carries `step`, `epoch`, `total`, and every mfm `opt_losses` key.
- **wandb images logged with no explicit step** (`run.log_image(key, path)`) — distinct keys avoid collisions; the prior explicit-`step` final-image hack is removed.
- **Multi-group Hydra tree**: `conf/{model,dataset,training,sampling}/*.yaml` + `experiment/*.yaml` using `# @package _global_` + a `defaults` list with `override /group: option`. `hydra.run.dir: runs/0001_flow_matching/${now:%Y-%m-%dT%H-%M-%SZ}`, `hydra.job.chdir: false`.
- **`FlowMatchingConfig` cross-validator** (`@model_validator(mode="after")`): mlp needs `len(dataset.shape)==1`, dit needs `len(dataset.shape)==3`.
- **Each commit keeps the FULL package suite green** (the pre-commit `pytest` hook runs `packages/physics-informed-flow-map/tests -q`). `run.py`'s `main()` is transiently stale across Tasks 1–2 — that's fine: no test executes `main()` (the compose test only imports the module and validates `FlowMatchingConfig`, which stays flat+consistent until Task 3).
- **mypy strict scope** = `packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests` (keep green). `experiments/` is NOT under strict mypy (Hydra's untyped `@hydra.main`). ruff covers BOTH `packages/physics-informed-flow-map` and `experiments` (the pre-commit `ruff check --fix` + `ruff format` may reformat; re-stage and re-commit).
- `hydra-core` and `wandb` are already deps; **add `tqdm`** to `packages/physics-informed-flow-map/pyproject.toml` + `tqdm.*` to the root mypy overrides.
- Commit on `main`. Do NOT edit reference packages (`packages/mfm-meta-flow-map-main/` etc.). Gates stay as upper bounds. DRY, YAGNI, TDD.

---

### Task 1: Discriminated model + dataset config layer (package)

Introduce the model and dataset discriminated unions, move dataset build/metadata onto the config classes, and re-point `build_model`. Update every affected test + call site in one cohesive change so the suite stays green. `train()` and `run.py` are untouched here.

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/models.py` (full rewrite)
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py` (full rewrite)
- Modify: `packages/physics-informed-flow-map/tests/test_models.py` (full rewrite)
- Modify: `packages/physics-informed-flow-map/tests/test_datasets.py` (full rewrite)
- Modify: `packages/physics-informed-flow-map/tests/test_sample.py` (update `build_model` call + `make_dataset`→`build`)
- Modify: `packages/physics-informed-flow-map/tests/test_train.py` (update `build_model` calls + `make_dataset`→`build`; leave `train(n_steps=...)` calls unchanged)

**Interfaces:**
- Consumes: package `Config` base (`physics_informed_flow_map.experiment.Config`), `VelocityMLP`, mfm `DiTMFM`/`SIModelWrapper`/`Linear`/`BaseModel`.
- Produces:
  - `MLPModelConfig(kind="mlp", width, depth)`, `DiTModelConfig(kind="dit", hidden, depth, num_heads, patch_size)`, `ModelConfig` union, `ModelCase`, `MODELS: dict[str, ModelCase]`.
  - `build_model(shape: tuple[int, ...], num_classes: int | None, cfg: ModelConfig) -> BaseModel`.
  - `GaussiansDatasetConfig(name="gaussians", n_modes, radius, std, n_samples)`, `MNISTDatasetConfig(name="mnist", image_size, data_dir)`, `DatasetConfig` union, `DATASETS: dict[str, DatasetConfig]`. Each dataset config: `.shape` / `.num_classes` / `.requires_download` properties, `.build() -> Dataset`, `.visualize(samples, path) -> None`.

- [ ] **Step 1: Rewrite `models.py`**

```python
"""Velocity models + a discriminated-union model config.

Both architectures implement mfm's BaseModel.v interface so mfm's loss/sampler
drive them unchanged. The MLP subclasses BaseModel — it does not modify the mfm
library. ``build_model`` dispatches on the (discriminated-union) model config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
import torch.nn as nn
from pydantic import Field
from torch import Tensor

from mfm.SI import Linear
from mfm.models import DiTMFM
from mfm.models.base_model import BaseModel
from mfm.models.model_wrapper import SIModelWrapper

from physics_informed_flow_map.experiment import Config


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of a scalar time in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class VelocityMLP(BaseModel):  # type: ignore[misc]
    """Time-conditioned MLP velocity field v(t, x) for vector data.

    Ignores t_cond/x_cond/class_labels (unconditional flow matching).
    """

    def __init__(
        self, dim: int, width: int = 256, depth: int = 4, time_dim: int = 128
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def v(
        self,
        s: Tensor,
        t: Tensor,
        x: Tensor,
        t_cond: Tensor,
        x_cond: Tensor,
        **kwargs: object,
    ) -> Tensor:
        temb = self.time_embed(s)
        result: Tensor = self.net(torch.cat([x, temb], dim=-1))
        return result


class MLPModelConfig(Config):
    """Config for the MLP velocity field (vector data)."""

    kind: Literal["mlp"] = "mlp"
    width: int = 256
    depth: int = 4


class DiTModelConfig(Config):
    """Config for the DiT velocity field (image data)."""

    kind: Literal["dit"] = "dit"
    hidden: int = 128
    depth: int = 4
    num_heads: int = 4
    patch_size: int = 4


ModelConfig = Annotated[MLPModelConfig | DiTModelConfig, Field(discriminator="kind")]


def build_model(
    shape: tuple[int, ...], num_classes: int | None, cfg: ModelConfig
) -> BaseModel:
    """Build the velocity model for a per-sample ``shape`` from a model config.

    ``MLPModelConfig`` requires vector data (``len(shape) == 1``);
    ``DiTModelConfig`` requires square images (``len(shape) == 3``, H == W).
    """
    if isinstance(cfg, MLPModelConfig):
        if len(shape) != 1:
            raise ValueError(f"mlp model requires vector data, got {shape}")
        return VelocityMLP(dim=shape[0], width=cfg.width, depth=cfg.depth)
    if isinstance(cfg, DiTModelConfig):
        if len(shape) != 3:
            raise ValueError(f"dit model requires image data, got {shape}")
        c, h, w = shape
        if h != w:
            raise ValueError(f"DiTMFM requires square images, got {shape}")
        dit = DiTMFM(
            learn_loss_weighting=False,
            input_size=h,
            patch_size=cfg.patch_size,
            in_channels=c,
            hidden_size=cfg.hidden,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            label_dim=num_classes or 1,
            encoder_depth=2,
            attn_func="base",
            is_zero_data=True,
            learn_sigma=False,
        )
        return SIModelWrapper(dit, Linear(t_max=1.0), use_parametrization=False)
    raise ValueError(f"unsupported model config {cfg!r}")


@dataclass(frozen=True)
class ModelCase:
    """A model config paired with a representative input contract (for tests/tooling)."""

    config: ModelConfig
    sample_shape: tuple[int, ...]
    num_classes: int | None


MODELS: dict[str, ModelCase] = {
    "mlp": ModelCase(MLPModelConfig(), (2,), None),
    "dit": ModelCase(DiTModelConfig(), (1, 32, 32), 10),
}
```

- [ ] **Step 2: Rewrite `datasets.py`**

```python
"""Dataset configs (discriminated union). Swapping datasets = changing one group.

Each variant owns its build + metadata; the module-level ``_make_*``/``_viz_*``
helpers do the actual work and are delegated to.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from pydantic import Field
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

from physics_informed_flow_map.experiment import Config


def _make_gaussians(
    n_samples: int = 100_000,
    n_modes: int = 8,
    radius: float = 4.0,
    std: float = 0.5,
    seed: int = 0,
) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    angles = 2 * math.pi * torch.arange(n_modes) / n_modes
    centers = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
    )
    idx = torch.randint(0, n_modes, (n_samples,), generator=g)
    x = centers[idx] + std * torch.randn(n_samples, 2, generator=g)
    labels = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x.float(), labels)


def _make_mnist(data_dir: str = "data", image_size: int = 32) -> Dataset:
    transform = T.Compose(
        [T.Resize(image_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )
    return cast(
        Dataset,
        torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        ),
    )


def _viz_scatter(samples: Tensor, path: Path) -> None:
    s = samples.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(s[:, 0], s[:, 1], s=3, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_title("generated samples")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _viz_grid(samples: Tensor, path: Path) -> None:
    s = (samples.detach().cpu().clamp(-1, 1) + 1) / 2
    n = min(64, len(s))
    ncols = 8
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(s[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class GaussiansDatasetConfig(Config):
    """2D mixture-of-Gaussians toy dataset."""

    name: Literal["gaussians"] = "gaussians"
    n_modes: int = 8
    radius: float = 4.0
    std: float = 0.5
    n_samples: int = 100_000

    @property
    def requires_download(self) -> bool:
        return False

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def num_classes(self) -> int | None:
        return None

    def build(self) -> Dataset:
        return _make_gaussians(self.n_samples, self.n_modes, self.radius, self.std)

    def visualize(self, samples: Tensor, path: Path) -> None:
        _viz_scatter(samples, path)


class MNISTDatasetConfig(Config):
    """MNIST digits, resized to a square and normalised to [-1, 1]."""

    name: Literal["mnist"] = "mnist"
    image_size: int = 32
    data_dir: str = "data"

    @property
    def requires_download(self) -> bool:
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, self.image_size, self.image_size)

    @property
    def num_classes(self) -> int | None:
        return 10

    def build(self) -> Dataset:
        return _make_mnist(self.data_dir, self.image_size)

    def visualize(self, samples: Tensor, path: Path) -> None:
        _viz_grid(samples, path)


DatasetConfig = Annotated[
    GaussiansDatasetConfig | MNISTDatasetConfig, Field(discriminator="name")
]


DATASETS: dict[str, DatasetConfig] = {
    "gaussians": GaussiansDatasetConfig(),
    "mnist": MNISTDatasetConfig(),
}
```

- [ ] **Step 3: Rewrite `test_models.py`**

```python
"""Registry-driven model tests: every entry in MODELS does a forward + backward."""

from typing import cast

import pytest
import torch

from physics_informed_flow_map.flow_matching.models import (
    MODELS,
    DiTModelConfig,
    MLPModelConfig,
    ModelConfig,
    build_model,
)

# Tiny configs so every architecture builds and runs fast on CPU.
_TINY: dict[str, ModelConfig] = {
    "mlp": MLPModelConfig(width=16, depth=2),
    "dit": DiTModelConfig(hidden=32, depth=1, num_heads=4),
}


@pytest.fixture(params=sorted(MODELS))
def model_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def _has_finite_grads(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return len(grads) > 0 and all(bool(torch.isfinite(g).all()) for g in grads)


def test_model_forward_backward(model_name: str) -> None:
    case = MODELS[model_name]
    model = build_model(case.sample_shape, case.num_classes, _TINY[model_name])

    x = torch.randn(4, *case.sample_shape)
    s = torch.rand(x.shape[0])
    t_cond = torch.zeros(x.shape[0])
    x_cond = torch.zeros_like(x)

    v = model.v(s, s, x, t_cond, x_cond)
    assert v.shape == x.shape  # forward: velocity shaped like input

    loss = v.pow(2).mean()
    loss.backward()  # backward
    assert _has_finite_grads(model)


def test_mlp_rejects_image() -> None:
    with pytest.raises(ValueError):
        build_model((1, 16, 16), None, MLPModelConfig())


def test_dit_rejects_vector() -> None:
    with pytest.raises(ValueError):
        build_model((2,), None, DiTModelConfig(hidden=32, depth=1))


def test_dit_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        build_model((1, 16, 32), 10, DiTModelConfig(hidden=32, depth=1))
```

- [ ] **Step 4: Rewrite `test_datasets.py`**

```python
"""Registry-driven dataset tests: every entry in DATASETS is exercised."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS


@pytest.fixture(params=sorted(DATASETS))
def dataset_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


def test_registry_metadata(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    assert isinstance(cfg.shape, tuple) and all(d > 0 for d in cfg.shape)
    assert cfg.num_classes is None or cfg.num_classes > 0
    assert isinstance(cfg.requires_download, bool)


def test_build_shapes(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    if cfg.requires_download:
        pytest.skip(f"{dataset_name} requires download; exercised by the live run")
    ds = cfg.build()
    x1, label = ds[0]
    assert x1.shape == cfg.shape
    if cfg.num_classes is None:
        assert int(label) == 0
    else:
        assert 0 <= int(label) < cfg.num_classes
    loader = DataLoader(ds, batch_size=16)
    xb, lb = next(iter(loader))
    assert xb.shape == (16, *cfg.shape)
    assert lb.shape == (16,)


def test_visualize_writes_file(dataset_name: str, tmp_path: Path) -> None:
    cfg = DATASETS[dataset_name]
    samples = torch.randn(16, *cfg.shape)
    out = tmp_path / f"{dataset_name}.png"
    cfg.visualize(samples, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 5: Update `test_sample.py`**

Replace the imports + the two affected lines so it uses the new `build_model`
signature and `.build()`:

```python
import torch

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import MLPModelConfig, build_model
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)


def test_energy_distance_zero_for_same_distribution() -> None:
    torch.manual_seed(0)
    x = torch.randn(2000, 2)
    y = torch.randn(2000, 2)
    assert abs(energy_distance(x, x)) < 1e-4
    far = torch.randn(2000, 2) + 50.0
    assert energy_distance(x, far) > energy_distance(x, y)


def test_sample_shape() -> None:
    cfg = DATASETS["gaussians"]
    model = build_model(cfg.shape, cfg.num_classes, MLPModelConfig(width=16, depth=2))
    out = sample(model, 32, cfg.shape, sampler_steps=5, device=torch.device("cpu"))
    assert out.shape == (32, 2)


def test_real_reference() -> None:
    ds = DATASETS["gaussians"].build()
    ref = real_reference(ds, 100, torch.device("cpu"))
    assert ref.shape == (100, 2)
```

- [ ] **Step 6: Update `test_train.py` call sites only (NOT the `train()` epoch change yet)**

Replace `packages/physics-informed-flow-map/tests/test_train.py` with the version
below: it swaps `build_model(... mlp_width/mlp_depth ...)` → `MLPModelConfig(...)`
and `spec.make_dataset()` → `spec.build()`, but **keeps the step-based
`train(n_steps=...)` calls and step-cadence hooks unchanged** (the epoch switch is
Task 2). The step-based loop only runs `n_steps` iterations, so the full dataset is
fine here.

```python
import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import MLPModelConfig, build_model
from physics_informed_flow_map.flow_matching.train import train


def test_train_runs_and_logs() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    ds = spec.build()
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=64, depth=3))

    logged: list[dict] = []
    history = train(
        model,
        loader,
        n_steps=50,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
    )

    assert len(history) == 50
    assert len(logged) == 50
    assert all("fm_loss" in r for r in history)
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]


def test_train_logs_decomposed_losses() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.build(), batch_size=32, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

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
    loader = DataLoader(spec.build(), batch_size=16, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    evals: list[int] = []
    ckpts: list[tuple[int, bool, bool]] = []

    def on_eval(m: object, step: int) -> float:
        evals.append(step)
        return 1.0 / len(evals)  # strictly decreasing -> every eval is a new best

    def on_checkpoint(m: object, step: int, *, is_best: bool, is_final: bool) -> None:
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


def test_train_checkpoint_cadence_without_eval() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.build(), batch_size=16, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    ckpts: list[tuple[int, bool, bool]] = []

    def on_checkpoint(m: object, step: int, *, is_best: bool, is_final: bool) -> None:
        ckpts.append((step, is_best, is_final))

    train(
        model,
        loader,
        n_steps=10,
        lr=1e-3,
        device=torch.device("cpu"),
        ckpt_every=4,
        on_checkpoint=on_checkpoint,
    )

    # No eval -> never a best; cadence fires at (step+1) % 4 == 0 -> steps 3, 7.
    assert all(not is_best for _, is_best, _ in ckpts)
    assert [step for step, _, is_final in ckpts if not is_final] == [3, 7]
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 9  # exactly one final, last step
```

- [ ] **Step 7: Run the package suite to verify green**

Run: `uv run pytest packages/physics-informed-flow-map/tests -q`
Expected: all pass / 1 skipped (mnist download). `test_models.py` exercises mlp+dit
forward/backward and the three `ValueError` guards; `test_datasets.py` exercises
both configs; `test_train.py` still runs with `n_steps`.

- [ ] **Step 8: Verify mypy clean**

Run: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests`
Expected: `Success: no issues found`.

- [ ] **Step 9: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/models.py \
        packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py \
        packages/physics-informed-flow-map/tests/test_models.py \
        packages/physics-informed-flow-map/tests/test_datasets.py \
        packages/physics-informed-flow-map/tests/test_sample.py \
        packages/physics-informed-flow-map/tests/test_train.py
git commit -m "feat(flow_matching): discriminated-union model + dataset configs"
```

(Note: `run.py`'s `main()` now references the old `build_model`/`DATASETS` API and
is temporarily non-runnable — that is expected and fixed in Task 3. No test runs
`main()`, so the suite stays green.)

---

### Task 2: Epoch-based training + tqdm

Switch `train()` from a fixed step budget to epoch passes with a per-epoch tqdm bar and epoch-cadence hooks. Add the `tqdm` dependency.

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py` (replace the `train` function; keep `_Cfg`/`_fm_loss_cfg`)
- Modify: `packages/physics-informed-flow-map/pyproject.toml` (add `tqdm` dependency)
- Modify: `pyproject.toml` (root — add `tqdm.*` to the mypy override list)
- Modify: `packages/physics-informed-flow-map/tests/test_train.py` (epoch semantics in the `train()` calls + cadence tests)

**Interfaces:**
- Consumes: mfm loss (`_fm_loss_cfg`/`get_consistency_loss_fn`); `build_model`/`DATASETS` (new API from Task 1, already used by the tests).
- Produces: `train(model, loader, *, n_epochs, lr, device, num_classes=None, log=None, eval_every_epochs=0, on_eval=None, ckpt_every_epochs=0, on_checkpoint=None) -> list[dict[str, float]]`. `on_eval(model, epoch) -> float | None`; `on_checkpoint(model, epoch, *, is_best, is_final)`. Per-batch `log` record keys: `step`, `epoch`, `total`, + each mfm `opt_losses` key.

- [ ] **Step 1: Add the `tqdm` dependency + mypy override**

In `packages/physics-informed-flow-map/pyproject.toml`, add `"tqdm>=4.66"` to the
`[project] dependencies` list (alphabetical position near `torchvision`/`transformers`
is fine). Then run `uv sync` to make it importable:

Run: `uv sync`
Expected: resolves with tqdm present (it is already an indirect dependency, so this is fast).

In the root `pyproject.toml`, extend the mypy override module list to include `tqdm.*`:

```toml
[[tool.mypy.overrides]]
module = ["mfm.*", "diffusers.*", "torchvision.*", "matplotlib.*", "torchdiffeq.*", "wandb.*", "hydra.*", "tqdm.*"]
ignore_missing_imports = true
```

- [ ] **Step 2: Rewrite `test_train.py` for epoch semantics**

Replace the whole file. Note the **bounded** datasets: with epochs, the loop passes
over the *entire* dataset each epoch, so the tests build small
`GaussiansDatasetConfig(n_samples=…)` datasets (the old step-based tests used the
full 100k dataset because the loop only ran `n_steps` regardless of size). 6400
samples / batch 128 = 50 batches → 50 optimizer steps in one epoch (matching the
old down-trend budget).

```python
import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import GaussiansDatasetConfig
from physics_informed_flow_map.flow_matching.models import MLPModelConfig, build_model
from physics_informed_flow_map.flow_matching.train import train


def _gaussian_loader(n_samples: int, batch_size: int) -> DataLoader:
    ds = GaussiansDatasetConfig(n_samples=n_samples).build()
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def test_train_runs_and_logs() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(6400, 128)  # 50 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=64, depth=3))

    logged: list[dict] = []
    history = train(
        model,
        loader,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
    )

    assert len(history) == len(logged) == 50
    assert all("fm_loss" in r for r in history)
    assert all("epoch" in r for r in history)
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]


def test_train_logs_decomposed_losses() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(96, 32)  # 3 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    records: list[dict] = []
    train(
        model,
        loader,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: records.append(r),
    )
    assert len(records) == 3
    assert "total" in records[0]
    assert "fm_loss" in records[0]
    assert "epoch" in records[0]


def test_train_hooks_fire_on_epoch_cadence() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    evals: list[int] = []
    ckpts: list[tuple[int, bool, bool]] = []

    def on_eval(m: object, epoch: int) -> float:
        evals.append(epoch)
        return 1.0 / len(evals)  # strictly decreasing -> every eval is a new best

    def on_checkpoint(m: object, epoch: int, *, is_best: bool, is_final: bool) -> None:
        ckpts.append((epoch, is_best, is_final))

    train(
        model,
        _gaussian_loader(64, 16),  # 4 batches/epoch
        n_epochs=4,
        lr=1e-3,
        device=torch.device("cpu"),
        eval_every_epochs=2,
        on_eval=on_eval,
        ckpt_every_epochs=0,
        on_checkpoint=on_checkpoint,
    )

    assert evals == [1, 3]  # (epoch+1) % 2 == 0
    assert [c[0] for c in ckpts if c[1]] == [1, 3]  # best at each eval
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 3  # exactly one final, last epoch


def test_train_checkpoint_cadence_without_eval() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    ckpts: list[tuple[int, bool, bool]] = []

    def on_checkpoint(m: object, epoch: int, *, is_best: bool, is_final: bool) -> None:
        ckpts.append((epoch, is_best, is_final))

    train(
        model,
        _gaussian_loader(64, 16),
        n_epochs=6,
        lr=1e-3,
        device=torch.device("cpu"),
        ckpt_every_epochs=3,
        on_checkpoint=on_checkpoint,
    )

    # No eval -> never a best; cadence fires at (epoch+1) % 3 == 0 -> epochs 2, 5.
    assert all(not is_best for _, is_best, _ in ckpts)
    assert [ep for ep, _, is_final in ckpts if not is_final] == [2, 5]
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 5  # exactly one final, last epoch
```

- [ ] **Step 3: Run the train tests to verify they FAIL**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -q`
Expected: FAIL — every `train(...)` call now passes `n_epochs`, which the current
step-based `train()` rejects (`TypeError: train() got an unexpected keyword
argument 'n_epochs'`).

- [ ] **Step 4: Rewrite the `train()` function in `train.py`**

Add `from tqdm.auto import tqdm` to the imports (top of file, after the existing
imports). `import math` is already present. Replace the entire `train` function
with:

```python
def train(
    model: BaseModel,
    loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    num_classes: int | None = None,
    log: Callable[..., None] | None = None,
    eval_every_epochs: int = 0,
    on_eval: Callable[[BaseModel, int], float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> list[dict[str, float]]:
    label_dim = num_classes or 0
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    model.train()
    history: list[dict[str, float]] = []
    best_metric = math.inf
    step = 0
    for epoch in range(n_epochs):
        for x1, labels in tqdm(
            loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False
        ):
            x1 = x1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            opt_losses, _ = loss_fn(model, None, x1, labels, step=step)
            total = sum(opt_losses.values())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            rec: dict[str, float] = {
                "step": float(step),
                "epoch": float(epoch),
                "total": float(total.item()),
            }
            for name, value in opt_losses.items():
                rec[name] = float(value.item())
            history.append(rec)
            if log is not None:
                log(**rec)
            step += 1

        is_best = False
        if (
            on_eval is not None
            and eval_every_epochs
            and (epoch + 1) % eval_every_epochs == 0
        ):
            model.eval()
            metric = on_eval(model, epoch)
            model.train()
            if metric is not None and metric < best_metric:
                best_metric = metric
                is_best = True

        if on_checkpoint is not None and (
            is_best or (ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0)
        ):
            on_checkpoint(model, epoch, is_best=is_best, is_final=False)

    if on_checkpoint is not None:
        on_checkpoint(model, n_epochs - 1, is_best=False, is_final=True)
    return history
```

- [ ] **Step 5: Run the suite to verify green**

Run: `uv run pytest packages/physics-informed-flow-map/tests -q`
Expected: all pass / 1 skipped. The four `test_train_*` tests pass with epoch
semantics; tqdm output is captured by pytest (pristine on pass).

- [ ] **Step 6: Verify mypy clean**

Run: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py \
        packages/physics-informed-flow-map/tests/test_train.py \
        packages/physics-informed-flow-map/pyproject.toml pyproject.toml uv.lock
git commit -m "feat(train): epoch-based loop with tqdm + epoch-cadence hooks"
```

(Note: `run.py`'s `main()` still calls `train(n_steps=...)` and is non-runnable
until Task 3. Still no test executes `main()`, so the suite stays green.)

---

### Task 3: Hierarchical top config + multi-group Hydra tree + run.py integration

Define the nested `FlowMatchingConfig` (model/dataset/training/sampling) with the
cross-validator, build the multi-group `conf/` tree, rewrite `run.py`'s `main()` to
use everything, and update the compose test. This re-integrates `run.py`.

**Files:**
- Modify: `experiments/0001_flow_matching/run.py` (full rewrite)
- Modify: `experiments/0001_flow_matching/conf/config.yaml` (rewrite)
- Create: `experiments/0001_flow_matching/conf/model/mlp.yaml`, `.../model/dit.yaml`
- Create: `experiments/0001_flow_matching/conf/dataset/gaussians.yaml`, `.../dataset/mnist.yaml`
- Create: `experiments/0001_flow_matching/conf/training/default.yaml`
- Create: `experiments/0001_flow_matching/conf/sampling/default.yaml`
- Modify: `experiments/0001_flow_matching/conf/experiment/gaussians.yaml`, `mnist.yaml`, `smoke.yaml` (rewrite)
- Modify: `packages/physics-informed-flow-map/tests/test_experiment_conf.py`

**Interfaces:**
- Consumes: `ModelConfig`/`MLPModelConfig`/`DiTModelConfig`/`build_model` (Task 1);
  `DatasetConfig`/`GaussiansDatasetConfig` (Task 1); `train(..., n_epochs, eval_every_epochs, ckpt_every_epochs, on_eval, on_checkpoint)` (Task 2); `Config`/`start_run`/`Run` (harness); `sample`/`energy_distance`/`real_reference`.
- Produces: nested `FlowMatchingConfig` (`seed`, `gate`, `model`, `dataset`, `training`, `sampling`) + `TrainingConfig`/`SamplingConfig`, with a `@model_validator(mode="after")` cross-check.

- [ ] **Step 1: Create the multi-group config tree**

`experiments/0001_flow_matching/conf/config.yaml`:

```yaml
defaults:
  - _self_
  - model: mlp
  - dataset: gaussians
  - training: default
  - sampling: default
  - experiment: gaussians

seed: 0
gate: 0.5

hydra:
  run:
    dir: runs/0001_flow_matching/${now:%Y-%m-%dT%H-%M-%SZ}
  job:
    chdir: false
```

`conf/model/mlp.yaml`:

```yaml
kind: mlp
width: 256
depth: 4
```

`conf/model/dit.yaml`:

```yaml
kind: dit
hidden: 128
depth: 4
num_heads: 4
patch_size: 4
```

`conf/dataset/gaussians.yaml`:

```yaml
name: gaussians
n_modes: 8
radius: 4.0
std: 0.5
n_samples: 100000
```

`conf/dataset/mnist.yaml`:

```yaml
name: mnist
image_size: 32
data_dir: data
```

`conf/training/default.yaml`:

```yaml
n_epochs: 10
batch_size: 256
lr: 0.001
eval_every_epochs: 0
ckpt_every_epochs: 0
artifact_every_epochs: 0
```

`conf/sampling/default.yaml`:

```yaml
sampler_steps: 100
n_eval_samples: 2048
n_eval_viz: 64
```

`conf/experiment/gaussians.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: mlp
  - override /dataset: gaussians

gate: 0.5
training:
  n_epochs: 5
```

`conf/experiment/mnist.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: dit
  - override /dataset: mnist

gate: 240.0
training:
  n_epochs: 50
  batch_size: 128
sampling:
  sampler_steps: 50
```

`conf/experiment/smoke.yaml`:

```yaml
# @package _global_
defaults:
  - override /model: mlp
  - override /dataset: gaussians

gate: 1000000000.0
training:
  n_epochs: 1
sampling:
  n_eval_samples: 256
```

- [ ] **Step 2: Update the compose test to the nested shape (write the failing test)**

Replace `packages/physics-informed-flow-map/tests/test_experiment_conf.py`:

```python
"""0001 Hydra config groups compose and validate into FlowMatchingConfig."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from physics_informed_flow_map.flow_matching.datasets import GaussiansDatasetConfig
from physics_informed_flow_map.flow_matching.models import DiTModelConfig

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
    "variant,dataset_name,model_kind,n_epochs",
    [
        ("gaussians", "gaussians", "mlp", 5),
        ("mnist", "mnist", "dit", 50),
        ("smoke", "gaussians", "mlp", 1),
    ],
)
def test_compose_validates(
    variant: str, dataset_name: str, model_kind: str, n_epochs: int
) -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=[f"experiment={variant}"])
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.dataset.name == dataset_name
    assert cfg.model.kind == model_kind
    assert cfg.training.n_epochs == n_epochs


def test_compose_applies_cli_override() -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(
            config_name="config",
            overrides=["experiment=mnist", "training.n_epochs=80"],
        )
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.training.n_epochs == 80
    assert cfg.dataset.name == "mnist"


def test_incompatible_model_dataset_rejected() -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with pytest.raises(ValidationError):
        cfg_cls(model=DiTModelConfig(), dataset=GaussiansDatasetConfig())
```

- [ ] **Step 3: Run the compose test to verify it FAILS**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -q`
Expected: FAIL — the current flat `FlowMatchingConfig` has no `dataset.name` /
`model.kind` / `training.n_epochs` nesting (validation/attribute errors).

- [ ] **Step 4: Rewrite `run.py`**

```python
"""Train flow matching on swappable datasets (2D Gaussians, MNIST) via Hydra.

    uv run python experiments/0001_flow_matching/run.py                          # gaussians
    uv run python experiments/0001_flow_matching/run.py experiment=mnist
    uv run python experiments/0001_flow_matching/run.py experiment=smoke
    uv run python experiments/0001_flow_matching/run.py experiment=mnist training.n_epochs=80 training.eval_every_epochs=5

Verdict: gaussians -> energy distance < gate; mnist -> final FM loss < gate.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from mfm.models.base_model import BaseModel
from omegaconf import DictConfig
from pydantic import Field, model_validator

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    GaussiansDatasetConfig,
)
from physics_informed_flow_map.flow_matching.models import (
    DiTModelConfig,
    MLPModelConfig,
    ModelConfig,
    build_model,
)
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)
from physics_informed_flow_map.flow_matching.train import train

EXPERIMENT = "0001_flow_matching"


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = 1e-3
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(0, ge=0)
    artifact_every_epochs: int = Field(0, ge=0)


class SamplingConfig(Config):
    sampler_steps: int = Field(100, gt=0)
    n_eval_samples: int = Field(2048, gt=0)
    n_eval_viz: int = Field(64, gt=0)


class FlowMatchingConfig(Config):
    seed: int = 0
    gate: float = 0.5
    model: ModelConfig = MLPModelConfig()
    dataset: DatasetConfig = GaussiansDatasetConfig()
    training: TrainingConfig = TrainingConfig()
    sampling: SamplingConfig = SamplingConfig()

    @model_validator(mode="after")
    def _check_model_dataset_compat(self) -> "FlowMatchingConfig":
        ndim = len(self.dataset.shape)
        if isinstance(self.model, MLPModelConfig) and ndim != 1:
            raise ValueError("mlp model needs a vector dataset (e.g. gaussians)")
        if isinstance(self.model, DiTModelConfig) and ndim != 3:
            raise ValueError("dit model needs an image dataset (e.g. mnist)")
        return self


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = FlowMatchingConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, FlowMatchingConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())

    dataset = cfg.dataset.build()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    model = build_model(cfg.dataset.shape, cfg.dataset.num_classes, cfg.model).to(device)

    def on_eval(m: BaseModel, epoch: int) -> float | None:
        s = sample(
            m,
            cfg.sampling.n_eval_viz,
            cfg.dataset.shape,
            sampler_steps=cfg.sampling.sampler_steps,
            device=device,
        )
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p)
        if isinstance(cfg.dataset, GaussiansDatasetConfig):
            ref = real_reference(dataset, cfg.sampling.n_eval_viz, device)
            return energy_distance(s, ref)
        return None

    def on_checkpoint(
        m: BaseModel, epoch: int, *, is_best: bool = False, is_final: bool = False
    ) -> None:
        path = run.save_checkpoint(m, epoch, dataset=cfg.dataset.name, config=cfg.dump())
        aliases: list[str] = []
        if is_final:
            aliases.append("final")
        if is_best:
            aliases.append("best")
        if (
            cfg.training.artifact_every_epochs
            and (epoch + 1) % cfg.training.artifact_every_epochs == 0
        ):
            aliases.append("periodic")
        if aliases:
            run.log_artifact(path, name=f"{cfg.dataset.name}-model", aliases=aliases)

    history = train(
        model,
        loader,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=device,
        num_classes=cfg.dataset.num_classes,
        log=run.log,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
    final_loss = history[-1]["total"]

    samples = sample(
        model,
        cfg.sampling.n_eval_samples,
        cfg.dataset.shape,
        sampler_steps=cfg.sampling.sampler_steps,
        device=device,
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        verdict = "pass" if metric < cfg.gate else "fail"
        run.finish(verdict, energy_distance=metric, final_loss=final_loss, gate=cfg.gate)
    else:
        verdict = "pass" if final_loss < cfg.gate else "fail"
        run.finish(verdict, final_loss=final_loss, gate=cfg.gate)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the compose test to verify it PASSES**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -q`
Expected: PASS (3 parametrized compose + 1 CLI override + 1 incompatible-combo = 5 passed).

- [ ] **Step 6: Smoke-run the entry point end-to-end (disabled wandb)**

Run:
```bash
WANDB_MODE=disabled uv run python experiments/0001_flow_matching/run.py experiment=smoke
```
Expected: prints `[0001_flow_matching] run → runs/0001_flow_matching/<stamp>` and
`[0001_flow_matching] verdict=pass`; a tqdm bar appears for the single epoch; a
`runs/0001_flow_matching/<stamp>/checkpoints/step_0.pt` exists (final checkpoint at
`epoch = n_epochs-1 = 0`). No `manifest.json`/`metrics.jsonl`/`result.json`.

- [ ] **Step 7: Smoke-run an epoch-cadence eval + the dit/mnist path is wired**

Run:
```bash
WANDB_MODE=disabled uv run python experiments/0001_flow_matching/run.py experiment=gaussians training.n_epochs=2 training.eval_every_epochs=1 training.ckpt_every_epochs=1
```
Expected: `verdict=pass`; two tqdm epoch bars; periodic `samples_epoch*.png` written
in the run dir; `verdict=pass` printed. (This exercises the on_eval/on_checkpoint
closures end-to-end.)

- [ ] **Step 8: Run the full suite + mypy (enforced scope) + ruff**

Run:
```bash
uv run pytest packages/physics-informed-flow-map/tests -q
uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests
uv run ruff check packages/physics-informed-flow-map experiments
uv run ruff format --check packages/physics-informed-flow-map experiments
```
Expected: pytest all pass / 1 skipped; mypy clean; ruff clean.

- [ ] **Step 9: Commit**

```bash
git add experiments/0001_flow_matching/ packages/physics-informed-flow-map/tests/test_experiment_conf.py
git commit -m "feat(0001): hierarchical FlowMatchingConfig + multi-group Hydra tree + epoch run"
```

---

## Notes for the implementer

- **Do not edit `packages/mfm-meta-flow-map-main/`** or any reference package.
- The pre-commit hooks (`ruff check --fix`, `ruff format`, `mypy`, `pytest`) run on
  every commit; keep each commit green. If a hook reformats files, re-stage and re-commit.
- Across Tasks 1–2 the entry point `run.py` is intentionally non-runnable (it still
  references the pre-refactor API). This is expected — no test executes `main()`,
  and Task 3 re-integrates it. Do not "fix" `run.py` early in a way that breaks the
  flat compose test still in force during Tasks 1–2.
- `ModelConfig`/`DatasetConfig` are `Annotated[... , Field(discriminator=...)]` type
  aliases; annotate fields/vars with them, but instantiate the concrete classes
  (`MLPModelConfig()`, `GaussiansDatasetConfig()`).
- The intentional duplication between `conf/*/*.yaml` defaults and the pydantic field
  defaults is the accepted cost of "yaml + pydantic validation" (structured-config
  registration is out of scope).
- `new.py` scaffolder + `experiments/README.md` are intentionally left on the flat
  pattern (out of scope per the spec); 0001 is the worked multi-group example.
```
