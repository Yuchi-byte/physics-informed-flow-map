# 0001 Flow-Matching Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `0001_mnist_pipeline` and `0002_pretrained_sample` frameworks with one `0001_flow_matching` framework that trains flow matching on swappable datasets (2D Gaussians, MNIST) through a single generic loop built on `mfm`.

**Architecture:** A reusable flow-matching core lives in the package under `physics_informed_flow_map/flow_matching/` (datasets registry, model builder, training loop, sampling+eval). The experiment dir `experiments/0001_flow_matching/` is thin wiring: config → core → verdict. The core reuses `mfm`'s `Linear` interpolant, `get_consistency_loss_fn` (FM-only config), and `ode_sampler_fn`. A small `VelocityMLP` subclassing `mfm`'s `BaseModel` handles 2D data; `DiTMFM` handles images.

**Tech Stack:** Python 3.12, PyTorch (cu130/auto), `mfm` (workspace dep), pydantic, OmegaConf, torchvision, matplotlib, pytest, mypy. uv workspace.

## Global Constraints

- **Never edit the `mfm` library** (`packages/mfm-meta-flow-map-main/`). All new code lives in `packages/physics-informed-flow-map/` or `experiments/`.
- Pure flow matching = `mfm` loss with `data_fm=True`, `distill_fm=False`, model built with `learn_loss_weighting=False`, and trainer `t_cond_0_rate=1.0`. The off-diagonal consistency/distillation term in mfm's loss is gated by `step > num_warmup_steps` (independent of `distill_fm`), so `num_warmup_steps` must be parked beyond any run length to keep that term OFF.
- The model `.v(...)` is called by `mfm` with extra kwargs (`class_labels=`, `cfg_scale=`); every `.v` signature must accept `**kwargs`.
- Verdicts are asserted in code via a `gate`, never by eye (harness contract, `experiments/README.md`).
- Run artifacts go to the git-ignored `runs/`; the harness (`physics_informed_flow_map.experiment`) owns `start_run` / `Run.log` / `Run.finish`.
- Work happens on branch `flow-matching-framework` (already created).

---

### Task 1: Workspace tooling (mypy + pytest) and remove old frameworks

**Files:**
- Modify: `pyproject.toml` (root) — add `[tool.mypy]`, overrides, and dev deps
- Delete: `experiments/0001_mnist_pipeline/` (3 files), `experiments/0002_pretrained_sample/` (3 files)
- Create: `packages/physics-informed-flow-map/tests/__init__.py` (empty)

**Interfaces:**
- Produces: a working `uv run mypy …` and `uv run pytest …` baseline for later tasks.

- [ ] **Step 1: Add mypy config and dev tools to the root `pyproject.toml`**

Append to `pyproject.toml` (after the `[dependency-groups]` block), and add `"mypy>=1.11"` and `"pytest>=8"` to the `dev` list:

```toml
[tool.mypy]
mypy_path = ["packages/physics-informed-flow-map/src"]
strict = true
disallow_any_generics = false
disallow_any_explicit = false

[[tool.mypy.overrides]]
module = ["mfm.*", "diffusers.*", "torchvision.*", "matplotlib.*", "torchdiffeq.*"]
ignore_missing_imports = true
```

The `dev` group becomes:

```toml
[dependency-groups]
dev = [
  "physics-informed-flow-map",
  "ruff>=0.15.14",
  "mypy>=1.11",
  "pytest>=8",
  "ipykernel>=7.2.0",
  "jupyter>=1.1.1",
  "nbconvert>=7.17.1",
]
```

- [ ] **Step 2: Sync**

Run: `uv sync`
Expected: resolves and installs `mypy` and `pytest`; no errors.

- [ ] **Step 3: Remove the old frameworks and create the tests package**

```bash
git rm -r experiments/0001_mnist_pipeline experiments/0002_pretrained_sample
mkdir -p packages/physics-informed-flow-map/tests
touch packages/physics-informed-flow-map/tests/__init__.py
```

- [ ] **Step 4: Verify mypy is green on the existing package code**

Run: `uv run mypy packages/physics-informed-flow-map/src`
Expected: `Success: no issues found`. If errors appear in `experiment/config.py` or `experiment/run.py`, add the minimal annotations to satisfy strict mode (these files are already fully annotated; typical fixes are narrowing an `Any` from OmegaConf with an explicit cast). Re-run until green.

- [ ] **Step 5: Verify pytest collects (no tests yet)**

Run: `uv run pytest packages/physics-informed-flow-map/tests`
Expected: exit code 5, "no tests ran". This is success for now.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: add mypy+pytest tooling, remove old experiment frameworks"
```

---

### Task 2: Dataset abstraction + registry

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/__init__.py`
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py`
- Test: `packages/physics-informed-flow-map/tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `DatasetSpec` dataclass: `shape: tuple[int, ...]`, `num_classes: int | None`, `make_dataset: Callable[[], Dataset]`, `visualize: Callable[[Tensor, Path], None]`
  - `DATASETS: dict[str, DatasetSpec]` with keys `"gaussians"` (shape `(2,)`, `num_classes=None`) and `"mnist"` (shape `(1, 32, 32)`, `num_classes=10`)
  - Each dataset's `__getitem__` returns `(x1: Tensor, label: Tensor|int)`; gaussians labels are all `0`.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_datasets.py`:

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS


def test_registry_metadata():
    for name, spec in DATASETS.items():
        assert isinstance(spec.shape, tuple) and all(d > 0 for d in spec.shape)
        assert spec.num_classes is None or spec.num_classes > 0
        assert callable(spec.make_dataset)
        assert callable(spec.visualize)
    assert DATASETS["gaussians"].shape == (2,)
    assert DATASETS["gaussians"].num_classes is None
    assert DATASETS["mnist"].shape == (1, 32, 32)
    assert DATASETS["mnist"].num_classes == 10


def test_gaussians_samples_and_loader():
    spec = DATASETS["gaussians"]
    ds = spec.make_dataset()
    x1, label = ds[0]
    assert x1.shape == spec.shape
    assert int(label) == 0
    loader = DataLoader(ds, batch_size=16)
    xb, lb = next(iter(loader))
    assert xb.shape == (16, 2)
    assert lb.shape == (16,)


def test_gaussians_visualize(tmp_path: Path):
    spec = DATASETS["gaussians"]
    samples = torch.randn(64, 2)
    out = tmp_path / "scatter.png"
    spec.visualize(samples, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError: ... flow_matching.datasets`.

- [ ] **Step 3: Create the package `__init__.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/__init__.py`:

```python
"""Flow-matching core: dataset registry, model builder, training, sampling.

Built on the `mfm` package (Linear interpolant, consistency/FM loss, ODE
sampler). This is the surface to which physics-residual losses attach later.
"""
```

- [ ] **Step 4: Implement `datasets.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py`:

```python
"""Dataset abstraction + registry. Swapping datasets = changing one config key."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset


@dataclass
class DatasetSpec:
    shape: tuple[int, ...]
    num_classes: int | None
    make_dataset: Callable[[], Dataset]
    visualize: Callable[[Tensor, Path], None]


def _make_gaussians(
    n_samples: int = 100_000,
    n_modes: int = 8,
    radius: float = 4.0,
    std: float = 0.5,
    seed: int = 0,
) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    angles = 2 * math.pi * torch.arange(n_modes) / n_modes
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    idx = torch.randint(0, n_modes, (n_samples,), generator=g)
    x = centers[idx] + std * torch.randn(n_samples, 2, generator=g)
    labels = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x.float(), labels)


def _make_mnist(data_dir: str = "data", image_size: int = 32) -> Dataset:
    transform = T.Compose(
        [T.Resize(image_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )
    return torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
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
    s = ((samples.detach().cpu().clamp(-1, 1) + 1) / 2)
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


DATASETS: dict[str, DatasetSpec] = {
    "gaussians": DatasetSpec(
        shape=(2,), num_classes=None, make_dataset=_make_gaussians, visualize=_viz_scatter
    ),
    "mnist": DatasetSpec(
        shape=(1, 32, 32), num_classes=10, make_dataset=_make_mnist, visualize=_viz_grid
    ),
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_datasets.py -v`
Expected: 3 passed.

- [ ] **Step 6: Typecheck and commit**

Run: `uv run mypy packages/physics-informed-flow-map/src`
Expected: `Success: no issues found`.

```bash
git add -A
git commit -m "feat(flow_matching): dataset abstraction + registry (gaussians, mnist)"
```

---

### Task 3: Velocity models (MLP for vectors, DiT for images)

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/models.py`
- Test: `packages/physics-informed-flow-map/tests/test_models.py`

**Interfaces:**
- Consumes: `DatasetSpec` (Task 2).
- Produces:
  - `VelocityMLP(BaseModel)` with `.v(s, t, x, t_cond, x_cond, **kwargs) -> Tensor` returning velocity shaped like `x`.
  - `build_model(spec: DatasetSpec, *, mlp_width=256, mlp_depth=4, dit_hidden=128, dit_depth=4, num_heads=4) -> BaseModel` — `VelocityMLP` for `len(shape)==1`, `SIModelWrapper(DiTMFM)` for `len(shape)==3`.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_models.py`:

```python
import torch

from physics_informed_flow_map.flow_matching.datasets import DatasetSpec
from physics_informed_flow_map.flow_matching.models import build_model


def _has_finite_grads(model: torch.nn.Module) -> bool:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def _fwd_bwd(model, x):
    b = x.shape[0]
    s = torch.rand(b)
    t_cond = torch.zeros(b)
    x_cond = torch.zeros_like(x)
    v = model.v(s, s, x, t_cond, x_cond)
    assert v.shape == x.shape  # forward: velocity shaped like input
    loss = v.pow(2).mean()
    loss.backward()  # backward
    assert _has_finite_grads(model)


def test_vector_model_forward_backward():
    spec = DatasetSpec(shape=(2,), num_classes=None, make_dataset=lambda: None, visualize=lambda *_: None)
    model = build_model(spec, mlp_width=16, mlp_depth=2)
    _fwd_bwd(model, torch.randn(4, 2))


def test_image_model_forward_backward():
    spec = DatasetSpec(shape=(1, 32, 32), num_classes=10, make_dataset=lambda: None, visualize=lambda *_: None)
    model = build_model(spec, dit_hidden=32, dit_depth=1, num_heads=4)
    _fwd_bwd(model, torch.randn(2, 1, 32, 32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: ... flow_matching.models`.

- [ ] **Step 3: Implement `models.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/models.py`:

```python
"""Velocity models. A small MLP for low-dim data; mfm's DiT for images.

Both implement mfm's BaseModel.v interface so mfm's loss/sampler drive them
unchanged. The MLP subclasses BaseModel — it does not modify the mfm library.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from mfm.SI import Linear
from mfm.models import DiTMFM
from mfm.models.base_model import BaseModel
from mfm.models.model_wrapper import SIModelWrapper

from physics_informed_flow_map.flow_matching.datasets import DatasetSpec


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


class VelocityMLP(BaseModel):
    """Time-conditioned MLP velocity field v(t, x) for vector data.

    Ignores t_cond/x_cond/class_labels (unconditional flow matching).
    """

    def __init__(self, dim: int, width: int = 256, depth: int = 4, time_dim: int = 128) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def v(self, s: Tensor, t: Tensor, x: Tensor, t_cond: Tensor, x_cond: Tensor, **kwargs: object) -> Tensor:
        temb = self.time_embed(s)
        return self.net(torch.cat([x, temb], dim=-1))


def build_model(
    spec: DatasetSpec,
    *,
    mlp_width: int = 256,
    mlp_depth: int = 4,
    dit_hidden: int = 128,
    dit_depth: int = 4,
    num_heads: int = 4,
) -> BaseModel:
    if len(spec.shape) == 1:
        return VelocityMLP(dim=spec.shape[0], width=mlp_width, depth=mlp_depth)
    if len(spec.shape) == 3:
        c, h, _ = spec.shape
        dit = DiTMFM(
            learn_loss_weighting=False,
            input_size=h,
            patch_size=4,
            in_channels=c,
            hidden_size=dit_hidden,
            depth=dit_depth,
            num_heads=num_heads,
            label_dim=spec.num_classes or 1,
            encoder_depth=2,
            attn_func="base",
            is_zero_data=True,
            learn_sigma=False,
        )
        return SIModelWrapper(dit, Linear(t_max=1.0), use_parametrization=False)
    raise ValueError(f"unsupported sample shape {spec.shape}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_models.py -v`
Expected: 2 passed. (The image test builds a tiny DiT; runs on CPU in a few seconds.)

- [ ] **Step 5: Typecheck and commit**

Run: `uv run mypy packages/physics-informed-flow-map/src`
Expected: `Success: no issues found`.

```bash
git add -A
git commit -m "feat(flow_matching): VelocityMLP + build_model (MLP/DiT by modality)"
```

---

### Task 4: Training loop

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py`
- Test: `packages/physics-informed-flow-map/tests/test_train.py`

**Interfaces:**
- Consumes: a model with `.v` (Task 3), a `DataLoader` yielding `(x1, label)`.
- Produces: `train(model, loader, *, n_steps, lr, device, log=None) -> list[dict[str, float]]` — returns per-step history of `{"step", "fm_loss", "total"}`; calls `log(**rec)` per step if given.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_train.py`:

```python
import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.train import train


def test_train_runs_and_logs():
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    ds = spec.make_dataset()
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = build_model(spec, mlp_width=64, mlp_depth=3)

    logged: list[dict] = []
    history = train(
        model, loader, n_steps=50, lr=1e-3, device=torch.device("cpu"), log=lambda **r: logged.append(r)
    )

    assert len(history) == 50
    assert len(logged) == 50
    assert all("fm_loss" in r for r in history)
    # loss should be finite and trend down over 50 steps on this easy target
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: ... flow_matching.train`.

- [ ] **Step 3: Implement `train.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py`:

```python
"""Generic flow-matching training loop, wrapping mfm's FM loss (pure-FM config)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from mfm.SI import Linear
from mfm.losses.losses import get_consistency_loss_fn
from mfm.models.base_model import BaseModel


class _Cfg:
    """Attribute bag mirroring the Hydra DictConfig mfm's loss expects."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _fm_loss_cfg(label_dim: int) -> _Cfg:
    return _Cfg(
        SI=_Cfg(t_max=1.0),
        trainer=_Cfg(
            t_cond_warmup_steps=0,
            t_cond_0_rate=1.0,  # always condition on pure noise → standard FM
            t_cond_power=1.0,
            # mfm's off-diagonal consistency/distillation term is gated by
            # `step > num_warmup_steps` (independent of distill_fm). Park the
            # warmup beyond any run length so pure FM keeps only the diagonal term.
            num_warmup_steps=10**12,
            anneal_end_step=10**12,
            class_dropout_prob=0.0,
        ),
        model=_Cfg(
            label_dim=label_dim,
            learn_loss_weighting=False,
            model_guidance_class_ws=[],
            model_guidance_x_cond_ws=[],
            init="dmf",
        ),
        loss=_Cfg(
            data_fm=True,
            distill_fm=False,
            distillation_type="mf",
            model_guidance=False,
            model_guidance_base_prob=0.5,
            fm_loss_type="l2",
            distillation_loss_type="l2",
            distill_fm_loss_type="l2",
            distill_teacher_stop_grad=True,
            fm_adaptive_loss_p=None,
            fm_adaptive_loss_c=None,
            distill_adaptive_loss_p=None,
            distill_adaptive_loss_c=None,
        ),
    )


def train(
    model: BaseModel,
    loader: DataLoader,
    *,
    n_steps: int,
    lr: float,
    device: torch.device,
    log: Callable[..., None] | None = None,
) -> list[dict[str, float]]:
    label_dim = 0  # unconditional FM; class_labels are ignored by the velocity models
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    data_iter = iter(loader)
    history: list[dict[str, float]] = []
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

        rec = {
            "step": float(step),
            "fm_loss": float(opt_losses["fm_loss"].item()),
            "total": float(total.item()),
        }
        history.append(rec)
        if log is not None:
            log(**rec)
    return history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: 1 passed.

- [ ] **Step 5: Typecheck and commit**

Run: `uv run mypy packages/physics-informed-flow-map/src`
Expected: `Success: no issues found`.

```bash
git add -A
git commit -m "feat(flow_matching): generic FM training loop over mfm loss"
```

---

### Task 5: Sampling + evaluation metric

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/sample.py`
- Test: `packages/physics-informed-flow-map/tests/test_sample.py`

**Interfaces:**
- Consumes: a model with `.v` (Task 3).
- Produces:
  - `sample(model, n_samples, shape, *, sampler_steps, device) -> Tensor` of shape `(n_samples, *shape)`.
  - `energy_distance(x: Tensor, y: Tensor) -> float`.
  - `real_reference(dataset, n, device) -> Tensor` — `n` random `x1` rows stacked.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_sample.py`:

```python
import torch

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)


def test_energy_distance_zero_for_same_distribution():
    torch.manual_seed(0)
    x = torch.randn(2000, 2)
    y = torch.randn(2000, 2)
    assert abs(energy_distance(x, x)) < 1e-4
    far = torch.randn(2000, 2) + 50.0
    assert energy_distance(x, far) > energy_distance(x, y)


def test_sample_shape():
    spec = DATASETS["gaussians"]
    model = build_model(spec, mlp_width=16, mlp_depth=2)
    out = sample(model, 32, spec.shape, sampler_steps=5, device=torch.device("cpu"))
    assert out.shape == (32, 2)


def test_real_reference():
    ds = DATASETS["gaussians"].make_dataset()
    ref = real_reference(ds, 100, torch.device("cpu"))
    assert ref.shape == (100, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_sample.py -v`
Expected: FAIL — `ModuleNotFoundError: ... flow_matching.sample`.

- [ ] **Step 3: Implement `sample.py`**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/sample.py`:

```python
"""Sampling (mfm ODE sampler from noise) + an energy-distance eval metric."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from mfm.SI.samplers import ode_sampler_fn
from mfm.models.base_model import BaseModel


@torch.no_grad()
def sample(
    model: BaseModel,
    n_samples: int,
    shape: tuple[int, ...],
    *,
    sampler_steps: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    x_noise = torch.randn(n_samples, *shape, device=device)
    t_cond = torch.zeros(n_samples, device=device)
    return ode_sampler_fn(
        model,
        xt_cond=x_noise,
        t_cond=t_cond,
        n_steps=sampler_steps,
        solver="euler",
        eps_start=x_noise,
        v_type="standard",
    )


def _pdist_mean(a: Tensor, b: Tensor) -> Tensor:
    return torch.cdist(a, b).mean()


def energy_distance(x: Tensor, y: Tensor) -> float:
    """Energy distance between two point sets (lower = closer distributions)."""
    x = x.flatten(1) if x.ndim > 2 else x
    y = y.flatten(1) if y.ndim > 2 else y
    val = 2 * _pdist_mean(x, y) - _pdist_mean(x, x) - _pdist_mean(y, y)
    return float(val.item())


def real_reference(dataset: Dataset, n: int, device: torch.device) -> Tensor:
    idx = torch.randperm(len(dataset))[:n]  # type: ignore[arg-type]
    xs = torch.stack([dataset[int(i)][0] for i in idx])
    return xs.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_sample.py -v`
Expected: 3 passed.

- [ ] **Step 5: Typecheck and commit**

Run: `uv run mypy packages/physics-informed-flow-map/src`
Expected: `Success: no issues found`.

```bash
git add -A
git commit -m "feat(flow_matching): ODE sampling + energy-distance eval"
```

---

### Task 6: The `0001_flow_matching` experiment framework + smoke run

**Files:**
- Create: `experiments/0001_flow_matching/run.py`
- Create: `experiments/0001_flow_matching/report.md`

**Interfaces:**
- Consumes: `physics_informed_flow_map.experiment.{Config, start_run}`; `flow_matching.{datasets, models, train, sample}`.
- Produces: a runnable framework: `uv run python experiments/0001_flow_matching/run.py [variant] [key=value...]`.

- [ ] **Step 1: Implement `run.py`**

Create `experiments/0001_flow_matching/run.py`:

```python
"""Train flow matching on swappable datasets (2D Gaussians, MNIST).

    uv run python experiments/0001_flow_matching/run.py gaussians
    uv run python experiments/0001_flow_matching/run.py mnist
    uv run python experiments/0001_flow_matching/run.py smoke

Verdict: gaussians → energy distance < gate; mnist → final FM loss < gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)
from physics_informed_flow_map.flow_matching.train import train


class FlowMatchingConfig(Config):
    seed: int = 0
    dataset: str = "gaussians"
    n_steps: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    sampler_steps: int = 100
    n_eval_samples: int = 2048
    gate: float = 0.5
    mlp_width: int = 256
    mlp_depth: int = 4
    dit_hidden: int = 128
    dit_depth: int = 4


VARIANTS: dict[str, dict[str, object]] = {
    "gaussians": {"dataset": "gaussians", "n_steps": 2000, "gate": 0.5},
    "mnist": {
        "dataset": "mnist",
        "n_steps": 3000,
        "batch_size": 128,
        "sampler_steps": 50,
        "gate": 5.0,
    },
    "smoke": {"dataset": "gaussians", "n_steps": 20, "n_eval_samples": 256, "gate": 1e9},
}


def main() -> None:
    argv = sys.argv[1:]
    has_variant = bool(argv) and "=" not in argv[0]
    variant = argv[0] if has_variant else "gaussians"
    overrides = argv[1:] if has_variant else argv
    if variant not in VARIANTS:
        sys.exit(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    cfg = FlowMatchingConfig.resolve(VARIANTS[variant], overrides)
    assert isinstance(cfg, FlowMatchingConfig)

    spec = DATASETS[cfg.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run = start_run(Path(__file__).parent, cfg.dump())

    dataset = spec.make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    model = build_model(
        spec,
        mlp_width=cfg.mlp_width,
        mlp_depth=cfg.mlp_depth,
        dit_hidden=cfg.dit_hidden,
        dit_depth=cfg.dit_depth,
    ).to(device)

    history = train(model, loader, n_steps=cfg.n_steps, lr=cfg.lr, device=device, log=run.log)
    final_loss = history[-1]["total"]

    samples = sample(model, cfg.n_eval_samples, spec.shape, sampler_steps=cfg.sampler_steps, device=device)
    spec.visualize(samples, run.dir / "samples.png")

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

- [ ] **Step 2: Create `report.md`**

Create `experiments/0001_flow_matching/report.md`:

```markdown
# 0001 — Flow matching on various datasets

Status: open

## Hypothesis

Pure flow matching (mfm FM loss, t_cond=0) trains a velocity field that samples
the target distribution: 2D Gaussians (energy-distance gate) and MNIST (FM-loss
gate), through one generic loop with a swappable dataset registry.

## Setup

`run.py [variant]` — `FlowMatchingConfig` (dataset, steps, lr, model knobs)
drives: dataset registry → `build_model` (MLP for vectors, DiT for images) →
generic FM training (`flow_matching.train`) → ODE sampling
(`flow_matching.sample`). Datasets: `gaussians`, `mnist`. Stack: mfm
(interpolant/loss/sampler).

## Results

Cite run directories under `runs/0001_flow_matching/`; quote `energy_distance` /
`final_loss` from `result.json` and inspect `samples.png`.

## Decision

Adopted / Falsified / Parked. Mirror the verdict line to `../JOURNAL.md`.
```

- [ ] **Step 3: Run the smoke variant end to end**

Run: `uv run python experiments/0001_flow_matching/run.py smoke`
Expected: trains 20 steps, samples, writes `runs/0001_flow_matching/<stamp>/` with `manifest.json`, `metrics.jsonl`, `result.json`, `samples.png`; prints `verdict=pass` (gate is 1e9, plumbing only).

- [ ] **Step 4: Verify the run artifacts**

Run: `cat $(ls -dt runs/0001_flow_matching/*/ | head -1)result.json`
Expected: JSON with `"verdict": "pass"`, an `energy_distance`, `final_loss`, `gate`.

- [ ] **Step 5: Typecheck and commit**

Run: `uv run mypy packages/physics-informed-flow-map/src`
Expected: `Success: no issues found`.

```bash
git add experiments/0001_flow_matching
git commit -m "feat(exp): 0001_flow_matching framework (gaussians, mnist)"
```

---

### Task 7: Real gaussians run, gate calibration, docs

**Files:**
- Modify: `experiments/0001_flow_matching/run.py` (only if the gate needs calibration)
- Modify: `experiments/0001_flow_matching/report.md` (fill Results/Decision)
- Modify: `experiments/JOURNAL.md` (add the verdict line)

**Interfaces:**
- Consumes: the framework (Task 6).

- [ ] **Step 1: Run the real gaussians variant**

Run: `uv run python experiments/0001_flow_matching/run.py gaussians`
Expected: trains 2000 steps (fast on CPU/GPU with the tiny MLP), prints a `verdict` and an `energy_distance`. Note the printed `energy_distance` value `E`.

- [ ] **Step 2: Calibrate the gate (only if needed)**

Inspect `runs/0001_flow_matching/<stamp>/samples.png` — the scatter should show 8 clear modes on a ring. If the samples look correct but `verdict=fail` because `E >= 0.5`, set the `gaussians` variant `gate` in `run.py` to a value comfortably above `E` (e.g. `round(E * 1.5, 2)`) so a correct fit passes with margin and a broken fit (modes missing/merged, larger `E`) fails. If samples look wrong, that is a real failure — debug via `superpowers:systematic-debugging`, do not loosen the gate to mask it. Re-run Step 1 after any change.

- [ ] **Step 3: Fill in the report and journal**

Edit `experiments/0001_flow_matching/report.md` Results section with the run directory and the observed `energy_distance`; set Decision to Adopted if the modes are recovered.

Append to `experiments/JOURNAL.md`:

```markdown
- 0001_flow_matching/gaussians — pass: energy distance <E> < <gate> gate; 8 modes recovered (runs/0001_flow_matching/<stamp>)
```

(Replace `<E>`, `<gate>`, `<stamp>` with actual values.)

- [ ] **Step 4: Full verification gate**

Run: `uv run pytest packages/physics-informed-flow-map/tests -v`
Expected: all tests pass.

Run: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests`
Expected: `Success: no issues found`.

- [ ] **Step 5: Commit**

```bash
git add experiments/0001_flow_matching/report.md experiments/JOURNAL.md experiments/0001_flow_matching/run.py
git commit -m "exp(0001): gaussians flow-matching run + calibrated gate"
```

---

## Self-Review notes

- **Spec coverage:** dataset abstraction+registry (Task 2), model-by-modality (Task 3), generic training over mfm loss (Task 4), ODE sampling + energy distance (Task 5), `0001_flow_matching` framework with variants + verdict (Task 6/7), mypy setup (Task 1), forward/backward tests for network + datasets (Tasks 2–3, plus train/sample), removal of old frameworks (Task 1). All covered.
- **mnist note:** unit tests stay offline (gaussians + synthetic batches); the mnist path is exercised by running `run.py mnist` (GPU), not in CI. This is intentional (network/size).
- **Gate honesty:** Task 7 Step 2 forbids loosening the gate to mask a real failure.
```
