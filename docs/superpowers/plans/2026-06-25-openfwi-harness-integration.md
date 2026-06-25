# OpenFWI Harness Integration (+ remove pass/fail gates) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the pass/fail gate logic from the `0001_flow_matching` harness, then add an `openfwi` dataset so OpenFWI velocity maps train an unconditional flow-matching prior through the existing wandb/Hydra/epoch/EMA loop.

**Architecture:** Part A replaces `Run.finish(verdict, …)` with `Run.finish(**summary)` and strips the `gate` field / verdict computation from the experiment. Part B adds a self-contained `flow_matching/openfwi.py` (a lazy memory-mapped `Dataset` + a colormap visualizer), then registers an `OpenFWIDatasetConfig` in the discriminated dataset union with matching Hydra config groups. OpenFWI rides the existing DiT path and the non-gaussians `else` branch — no `main()` special-casing.

**Tech Stack:** Python 3.12, PyTorch, NumPy (memory-mapped `.npy`), pydantic v2 (discriminated unions, `extra="forbid"`), Hydra, matplotlib (Agg), Weights & Biases, pytest, uv workspace.

## Global Constraints

- **Do NOT edit any reference package** (`packages/mfm-meta-flow-map-main/`, `packages/PhysicsInformedDiffusionModels-main/`, `packages/PBFM/`).
- Work and commit directly on `main` (the user has durably approved this — no feature branch).
- This is a uv workspace: run all Python as `uv run …` from the repo root `/home/markhaoxiang/Projects/physics-informed-flow-map`.
- A pre-commit hook chain runs on `git commit` (ruff check --fix → ruff format → mypy on the package → pytest on the package). Let it run; if it reformats or fails, fix the cause and re-commit. `mypy` is scoped to the package (`packages/physics-informed-flow-map/`), not `experiments/` — do not add type-ignore noise to `experiments/0001_flow_matching/run.py`.
- Tests that touch wandb run under `WANDB_MODE=disabled` (the `test_run.py` autouse fixture already sets this).
- No pass/fail gates anywhere after this work: `Run.finish` takes only `**summary`; no `"verdict"` summary key; no `gate` config field or yaml key.
- OpenFWI normalization is fixed: velocity `[1500.0, 4500.0] m/s → [-1, 1]`, clamped. Native map size is `70×70`; default training resolution is `64` (resize via bilinear + antialias). Unconditional (`num_classes=None`, label `0`).
- OpenFWI default family is `["FlatVel_A"]`; the loader globs **both** layouts: `<family>/model/*.npy` and flat `<family>/vel*.npy`.
- Tests must be hermetic — no dependency on the real 8.8 GB `data/openfwi/` download. Use synthetic `.npy` fixtures under `tmp_path`.

---

### Task 1: Remove pass/fail gates (harness + 0001 experiment)

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py` (the `finish` method + module docstring)
- Modify: `experiments/0001_flow_matching/run.py` (remove `gate` field, verdict logic, docstring line)
- Modify: `experiments/0001_flow_matching/conf/config.yaml` (remove `gate: 0.5`)
- Modify: `experiments/0001_flow_matching/conf/experiment/gaussians.yaml` (remove `gate: 0.5`)
- Modify: `experiments/0001_flow_matching/conf/experiment/mnist.yaml` (remove `gate: 240.0`)
- Modify: `experiments/0001_flow_matching/conf/experiment/smoke.yaml` (remove `gate: 1000000000.0`)
- Modify: `experiments/JOURNAL.md` (format guidance line)
- Test: `packages/physics-informed-flow-map/tests/test_run.py`

**Interfaces:**
- Produces: `Run.finish(self, **summary: Any) -> None` — records each summary scalar to `wandb.run.summary` and prints them; no `verdict` parameter, no `"verdict"` key.

- [ ] **Step 1: Update the harness test to the new signature (failing test)**

In `packages/physics-informed-flow-map/tests/test_run.py`, change the last line of `test_run_lifecycle` from:
```python
    run.finish("pass", final_loss=0.5)
```
to:
```python
    run.finish(final_loss=0.5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_run.py::test_run_lifecycle -v`
Expected: FAIL — `TypeError: finish() missing 1 required positional argument: 'verdict'`.

- [ ] **Step 3: Implement `Run.finish(**summary)`**

In `packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py`, replace the `finish` method:
```python
    def finish(self, **summary: Any) -> None:
        """Record summary scalars to the wandb run summary and close the run."""
        for key, value in summary.items():
            self.run.summary[key] = value
        extra = " ".join(f"{key}={value}" for key, value in summary.items())
        print(f"[{self.experiment}] {extra}".rstrip())
        self.run.finish()
```

Also update the module docstring near the top of the same file: change the sentence
```
and model checkpoints/artifacts to it; :meth:`finish` records the verdict in the
run summary. No local JSON is written — wandb is the single source of truth.
```
to
```
and model checkpoints/artifacts to it; :meth:`finish` records summary scalars in the
run summary. No local JSON is written — wandb is the single source of truth.
```

- [ ] **Step 4: Run the harness test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_run.py -v`
Expected: PASS (`test_run_lifecycle`, `test_save_checkpoint_suffix`).

- [ ] **Step 5: Remove the gate field + verdict logic from the experiment**

In `experiments/0001_flow_matching/run.py`:

(a) Remove the `gate` field from `FlowMatchingConfig`. Delete this line:
```python
    gate: float = 0.5
```

(b) Update the module docstring line (near line 8). Replace:
```
Verdict: gaussians -> energy distance < gate; mnist -> final FM loss < gate.
```
with:
```
Logs energy distance (gaussians) / final FM loss (image datasets) as run summary scalars.
```

(c) Replace the final verdict block. Change:
```python
    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        verdict = "pass" if metric < cfg.gate else "fail"
        run.finish(
            verdict, energy_distance=metric, final_loss=final_loss, gate=cfg.gate
        )
    else:
        verdict = "pass" if final_loss < cfg.gate else "fail"
        run.finish(verdict, final_loss=final_loss, gate=cfg.gate)
```
to:
```python
    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        run.finish(energy_distance=metric, final_loss=final_loss)
    else:
        run.finish(final_loss=final_loss)
```

- [ ] **Step 6: Remove `gate:` from the four config files**

- In `experiments/0001_flow_matching/conf/config.yaml`, delete the line `gate: 0.5` (leaving `seed: 0` and the `hydra:` block).
- In `experiments/0001_flow_matching/conf/experiment/gaussians.yaml`, delete the line `gate: 0.5`.
- In `experiments/0001_flow_matching/conf/experiment/mnist.yaml`, delete the line `gate: 240.0`.
- In `experiments/0001_flow_matching/conf/experiment/smoke.yaml`, delete the line `gate: 1000000000.0`.

- [ ] **Step 7: Run the full package suite to verify config consistency**

Run: `uv run pytest packages/physics-informed-flow-map/tests/ -q`
Expected: PASS. (The compose tests in `test_experiment_conf.py` call `FlowMatchingConfig.from_dictconfig`; because `Config` is `extra="forbid"`, a leftover `gate:` in any composed yaml or a leftover `gate` field would raise `ValidationError` here — green confirms field and yaml removals are consistent.)

- [ ] **Step 8: Update the JOURNAL format guidance**

In `experiments/JOURNAL.md`, change the format line from:
```
Format: `NNNN_slug/variant — verdict: headline (evidence)`
```
to:
```
Format: `NNNN_slug/variant — headline metric (evidence)`
```
Leave the existing historical journal entries unchanged.

- [ ] **Step 9: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py \
        packages/physics-informed-flow-map/tests/test_run.py \
        experiments/0001_flow_matching/run.py \
        experiments/0001_flow_matching/conf/config.yaml \
        experiments/0001_flow_matching/conf/experiment/gaussians.yaml \
        experiments/0001_flow_matching/conf/experiment/mnist.yaml \
        experiments/0001_flow_matching/conf/experiment/smoke.yaml \
        experiments/JOURNAL.md
git commit -m "refactor(0001): remove pass/fail gates; log metrics only"
```

---

### Task 2: OpenFWI dataset module (`flow_matching/openfwi.py`)

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/openfwi.py`
- Test: `packages/physics-informed-flow-map/tests/test_openfwi.py`

**Interfaces:**
- Produces:
  - `OpenFWIVelocityDataset(root: Path, families: list[str], resolution: int = 64)` — a `torch.utils.data.Dataset`; `__len__ -> int`; `__getitem__(idx: int) -> tuple[Tensor, int]` returning `(x, 0)` with `x` shape `(1, resolution, resolution)`, float32, values in `[-1, 1]`. Raises `FileNotFoundError` if a family directory yields no velocity files.
  - `viz_velocity(samples: Tensor, path: Path) -> None` — writes a colormap grid PNG.
  - Module constants `VMIN = 1500.0`, `VMAX = 4500.0`, `NATIVE = 70`.

- [ ] **Step 1: Write the failing tests**

Create `packages/physics-informed-flow-map/tests/test_openfwi.py`:
```python
"""OpenFWI velocity-map dataset: hermetic tests on synthetic .npy fixtures."""

from pathlib import Path

import numpy as np
import pytest
import torch

from physics_informed_flow_map.flow_matching.openfwi import (
    NATIVE,
    VMAX,
    VMIN,
    OpenFWIVelocityDataset,
    viz_velocity,
)


def _write_family(
    root: Path, family: str, layout: str, n_files: int = 2, rows: int = 3
) -> None:
    """Write `n_files` synthetic velocity files for one family, in the given layout.

    layout="model" -> <family>/model/model{i}.npy
    layout="flat"  -> <family>/vel{i}.npy
    Each file holds `rows` maps of shape (1, 70, 70) spanning [VMIN, VMAX].
    """
    if layout == "model":
        out_dir = root / family / "model"
        names = [f"model{i}.npy" for i in range(n_files)]
    else:
        out_dir = root / family
        names = [f"vel{i}.npy" for i in range(n_files)]
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.linspace(VMIN, VMAX, rows * 70 * 70, dtype=np.float32).reshape(
        rows, 1, 70, 70
    )
    for name in names:
        np.save(out_dir / name, arr)


def test_len_counts_all_rows(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A"])
    assert len(ds) == 6  # 2 files * 3 rows


def test_sample_shape_and_range(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model")
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A"], resolution=64)
    x, label = ds[0]
    assert x.shape == (1, 64, 64)
    assert x.dtype == torch.float32
    assert int(label) == 0
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0


def test_normalization_endpoints_native_resolution(tmp_path: Path) -> None:
    # At native resolution (no resize), VMIN -> -1 and VMAX -> +1 exactly.
    _write_family(tmp_path, "FlatVel_A", "model", n_files=1, rows=1)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A"], resolution=NATIVE)
    x, _ = ds[0]
    assert x.shape == (1, NATIVE, NATIVE)
    assert float(x.min()) == pytest.approx(-1.0, abs=1e-4)
    assert float(x.max()) == pytest.approx(1.0, abs=1e-4)


def test_flat_vel_layout_loads(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=2, rows=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatFault_A"])
    assert len(ds) == 6
    x, _ = ds[0]
    assert x.shape == (1, 64, 64)


def test_multiple_families(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=1, rows=3)
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=1, rows=3)
    ds = OpenFWIVelocityDataset(tmp_path, ["FlatVel_A", "FlatFault_A"])
    assert len(ds) == 6


def test_missing_family_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenFWIVelocityDataset(tmp_path, ["DoesNotExist"])


def test_viz_velocity_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "vel.png"
    viz_velocity(torch.randn(16, 1, 64, 64), out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_openfwi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...flow_matching.openfwi'`.

- [ ] **Step 3: Implement the module**

Create `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/openfwi.py`:
```python
"""OpenFWI velocity-map dataset (lazy, memory-mapped) + a colormap visualizer.

Velocity maps are normalised from [1500, 4500] m/s to [-1, 1] and optionally
resized from the native 70x70 to a square training resolution. The dataset is
unconditional: every sample is paired with a dummy label 0.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

VMIN = 1500.0
VMAX = 4500.0
NATIVE = 70


class OpenFWIVelocityDataset(Dataset):
    """Lazy, memory-mapped OpenFWI velocity maps across one or more families."""

    def __init__(
        self, root: Path, families: list[str], resolution: int = 64
    ) -> None:
        self.resolution = resolution
        self.index: list[tuple[Path, int]] = []
        for family in families:
            family_dir = root / family
            files = sorted(family_dir.glob("model/*.npy")) + sorted(
                family_dir.glob("vel*.npy")
            )
            if not files:
                raise FileNotFoundError(
                    f"No OpenFWI velocity files under {family_dir} "
                    f"(expected <family>/model/*.npy or <family>/vel*.npy). "
                    f"Download from the 'ashynf/OpenFWI' HuggingFace dataset."
                )
            for f in files:
                n_rows = int(np.load(f, mmap_mode="r").shape[0])
                self.index.extend((f, i) for i in range(n_rows))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path, row = self.index[idx]
        arr = np.load(path, mmap_mode="r")[row]  # (1, 70, 70) float32
        x = torch.from_numpy(np.ascontiguousarray(arr)).float()
        x = ((x - VMIN) / (VMAX - VMIN) * 2.0 - 1.0).clamp(-1.0, 1.0)
        if self.resolution != NATIVE:
            x = F.interpolate(
                x[None],
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        return x, 0


def viz_velocity(samples: Tensor, path: Path) -> None:
    """Grid of velocity maps with a perceptual colormap (samples are in [-1, 1])."""
    s = samples.detach().cpu().clamp(-1, 1)
    n = min(64, len(s))
    ncols = 8
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(s[i, 0].numpy(), cmap="viridis", vmin=-1, vmax=1)
        ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_openfwi.py -v`
Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/openfwi.py \
        packages/physics-informed-flow-map/tests/test_openfwi.py
git commit -m "feat(flow_matching): lazy memory-mapped OpenFWI velocity dataset + viz"
```

---

### Task 3: Register OpenFWI in the dataset union + Hydra config

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py` (import from `openfwi`, add `OpenFWIDatasetConfig`, extend the union + `DATASETS`)
- Create: `experiments/0001_flow_matching/conf/dataset/openfwi.yaml`
- Create: `experiments/0001_flow_matching/conf/experiment/openfwi.yaml`
- Test: `packages/physics-informed-flow-map/tests/test_experiment_conf.py`

**Interfaces:**
- Consumes: `OpenFWIVelocityDataset`, `viz_velocity` (Task 2); the existing `Config` base, `DatasetConfig` union, and `DATASETS` registry in `datasets.py`.
- Produces: `OpenFWIDatasetConfig` variant (`name="openfwi"`, `data_dir`, `families`, `resolution`) registered in the union and `DATASETS`; Hydra groups `dataset/openfwi` and `experiment/openfwi`.

- [ ] **Step 1: Write the failing compose tests**

In `packages/physics-informed-flow-map/tests/test_experiment_conf.py`:

(a) Add the `openfwi` row to the existing `test_compose_validates` parametrization. Change:
```python
@pytest.mark.parametrize(
    "variant,dataset_name,model_kind,n_epochs",
    [
        ("gaussians", "gaussians", "mlp", 100),
        ("mnist", "mnist", "dit", 100),
        ("smoke", "gaussians", "mlp", 1),
    ],
)
```
to:
```python
@pytest.mark.parametrize(
    "variant,dataset_name,model_kind,n_epochs",
    [
        ("gaussians", "gaussians", "mlp", 100),
        ("mnist", "mnist", "dit", 100),
        ("openfwi", "openfwi", "dit", 100),
        ("smoke", "gaussians", "mlp", 1),
    ],
)
```

(b) Add the `openfwi` row to the existing `test_compose_ema_enabled` parametrization. Change:
```python
@pytest.mark.parametrize(
    "variant,ema_enabled",
    [("gaussians", True), ("mnist", True), ("smoke", False)],
)
```
to:
```python
@pytest.mark.parametrize(
    "variant,ema_enabled",
    [("gaussians", True), ("mnist", True), ("openfwi", True), ("smoke", False)],
)
```

(c) Append a new test for the OpenFWI-specific shape:
```python
def test_compose_openfwi_shape() -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=["experiment=openfwi"])
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.dataset.name == "openfwi"
    assert cfg.dataset.shape == (1, 64, 64)
    assert cfg.dataset.families == ["FlatVel_A"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -v`
Expected: FAIL — the `openfwi` parametrizations and `test_compose_openfwi_shape` error because `experiment=openfwi` cannot be composed (no such config group yet).

- [ ] **Step 3: Add `OpenFWIDatasetConfig` to the union**

In `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py`:

(a) Add the import (after the existing `from physics_informed_flow_map.experiment import Config`):
```python
from physics_informed_flow_map.flow_matching.openfwi import (
    OpenFWIVelocityDataset,
    viz_velocity,
)
```

(b) Add the config class after `MNISTDatasetConfig`:
```python
class OpenFWIDatasetConfig(Config):
    """OpenFWI subsurface velocity maps, normalised to [-1, 1]."""

    name: Literal["openfwi"] = "openfwi"
    data_dir: str = "data/openfwi"
    families: list[str] = ["FlatVel_A"]
    resolution: int = 64

    @property
    def requires_download(self) -> bool:
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, self.resolution, self.resolution)

    @property
    def num_classes(self) -> int | None:
        return None

    def build(self) -> Dataset:
        return OpenFWIVelocityDataset(
            Path(self.data_dir), self.families, self.resolution
        )

    def visualize(self, samples: Tensor, path: Path) -> None:
        viz_velocity(samples, path)
```

(c) Extend the union and registry. Change:
```python
DatasetConfig = Annotated[
    GaussiansDatasetConfig | MNISTDatasetConfig, Field(discriminator="name")
]


DATASETS: dict[str, DatasetConfig] = {
    "gaussians": GaussiansDatasetConfig(),
    "mnist": MNISTDatasetConfig(),
}
```
to:
```python
DatasetConfig = Annotated[
    GaussiansDatasetConfig | MNISTDatasetConfig | OpenFWIDatasetConfig,
    Field(discriminator="name"),
]


DATASETS: dict[str, DatasetConfig] = {
    "gaussians": GaussiansDatasetConfig(),
    "mnist": MNISTDatasetConfig(),
    "openfwi": OpenFWIDatasetConfig(),
}
```

- [ ] **Step 4: Create the Hydra config groups**

Create `experiments/0001_flow_matching/conf/dataset/openfwi.yaml`:
```yaml
name: openfwi
data_dir: data/openfwi
families: [FlatVel_A]
resolution: 64
```

Create `experiments/0001_flow_matching/conf/experiment/openfwi.yaml`:
```yaml
# @package _global_
defaults:
  - override /model: dit
  - override /dataset: openfwi

training:
  n_epochs: 100
  batch_size: 64
  ema:
    enabled: true
sampling:
  sampler_steps: 200
  n_eval_samples: 64
```

- [ ] **Step 5: Run the experiment-conf tests, then the full suite**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -v`
Expected: PASS — the `openfwi` parametrizations and `test_compose_openfwi_shape` now compose and validate (`dataset.name == "openfwi"`, `model.kind == "dit"`, `shape == (1, 64, 64)`, `ema.enabled is True`).

Run: `uv run pytest packages/physics-informed-flow-map/tests/ -q`
Expected: PASS. The registry-driven `test_datasets.py` now also exercises `openfwi`: `test_registry_metadata` (shape/num_classes/requires_download) passes; `test_build_shapes` **skips** it (`requires_download` is `True`); `test_visualize_writes_file` runs `viz_velocity` on a random tensor and passes.

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py \
        packages/physics-informed-flow-map/tests/test_experiment_conf.py \
        experiments/0001_flow_matching/conf/dataset/openfwi.yaml \
        experiments/0001_flow_matching/conf/experiment/openfwi.yaml
git commit -m "feat(0001): register openfwi dataset + Hydra variant"
```

---

## Self-Review

**Spec coverage:**
- Part A — `Run.finish(**summary)`, remove `gate` field + verdict logic, remove `gate:` from 4 yamls, `test_run.py` update, `JOURNAL.md` format → Task 1. ✅
- Part B — `flow_matching/openfwi.py` (`OpenFWIVelocityDataset` lazy mmap, both layouts, normalize+clamp, resize, `FileNotFoundError`, `viz_velocity` colormap) → Task 2. ✅
- Part B — `OpenFWIDatasetConfig` in union + `DATASETS`, `conf/dataset/openfwi.yaml`, `conf/experiment/openfwi.yaml` (DiT, EMA on, `n_eval_samples: 64`), compose tests → Task 3. ✅
- Hermetic tests via synthetic `.npy` fixtures → Task 2 (`_write_family`); registry skip for download-required → Task 3 note. ✅
- "No `main()` special-casing": OpenFWI takes the `else` branch unchanged → confirmed in Task 1 (the `else` already logs `final_loss`); no OpenFWI edit to `run.py`. ✅
- Out-of-scope items (held-out validation, conditional `p(v|d)`, multi-family tuning, auto-download) → not implemented. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✅

**Type consistency:** `OpenFWIVelocityDataset(root, families, resolution=64)` and `__getitem__ -> tuple[Tensor, int]` (Task 2) match `OpenFWIDatasetConfig.build()` and the `(x, label)` harness contract (Task 3). `viz_velocity(samples, path)` (Task 2) matches the `visualize` call (Task 3). `Run.finish(**summary)` (Task 1) matches the `run.finish(energy_distance=…, final_loss=…)` / `run.finish(final_loss=…)` calls (Task 1) and `run.finish(final_loss=0.5)` (test). Constants `VMIN`/`VMAX`/`NATIVE` defined in Task 2 are imported by the Task 2 tests. ✅
