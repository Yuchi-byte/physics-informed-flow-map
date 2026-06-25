# Held-out Validation Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a held-out validation FM loss to `0001_flow_matching`: each dataset gets a disjoint validation split, each eval computes/logs `val_loss`, and `val_loss` selects the best checkpoint uniformly.

**Architecture:** Extract the FM loss into a reusable `make_loss_fn` so `run.py` can score a held-out loader with the same loss. `on_eval` returns that held-out loss, which `train()` logs as `val_loss` and uses for best-tracking via the existing mechanism. Each dataset config gains `build_val()` (gaussians: distinct-seed draw; mnist: test set; openfwi: deterministic `Subset` split).

**Tech Stack:** Python 3.12, PyTorch, pydantic v2 discriminated unions, Hydra, Weights & Biases, pytest, uv workspace.

## Global Constraints

- **Do NOT edit any reference package** (`packages/mfm-meta-flow-map-main/`, `packages/PhysicsInformedDiffusionModels-main/`, `packages/PBFM/`).
- Work and commit directly on `main` (durably approved — no feature branch).
- uv workspace: run all Python as `uv run …` from repo root `/home/markhaoxiang/Projects/physics-informed-flow-map`.
- Pre-commit chain runs on `git commit` (ruff check --fix → ruff format → mypy on the package → pytest on the package). Let it run; fix the cause if it fails. `mypy` is scoped to the package (`packages/physics-informed-flow-map/`), **including tests**, but NOT `experiments/` — do not add type-ignore noise to `experiments/0001_flow_matching/run.py`.
- Tests that touch wandb run under `WANDB_MODE=disabled`. Keep tests hermetic — no real MNIST/OpenFWI download.
- `val_loss` is the held-out FM loss, averaged over the validation split, computed in eval/no-grad mode. It is logged each eval epoch and once at the end, and it drives best-checkpoint selection (lower = better).
- `build()` returns the training split; `build_val()` the disjoint held-out split. OpenFWI `val_fraction` default `0.1`; gaussians `val_samples` default `10000` drawn with a distinct seed (`seed=1`); mnist uses the torchvision test set.
- Physics residuals, FID, and energy-distance-vs-held-out are out of scope.

---

### Task 1: Reusable FM loss + val_loss logging (`flow_matching/train.py`)

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py`
- Test: `packages/physics-informed-flow-map/tests/test_train.py`

**Interfaces:**
- Produces: `make_loss_fn(num_classes: int | None) -> Callable[..., Any]` — the pure-FM consistency loss used for both training and validation; calling it as `loss_fn(model, None, x1, labels, step=<int>)` returns `(opt_losses, aux)` where `opt_losses` is a dict of named loss tensors.
- `train()`'s signature and return type are unchanged; it now logs a `val_loss` record on eval epochs (when `on_eval` returns a metric and `log` is provided).

- [ ] **Step 1: Write the failing tests**

Add to `packages/physics-informed-flow-map/tests/test_train.py`. First add `import math` at the top (below the existing `import torch`), and extend the import from `train`:
```python
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train
```
Then append:
```python
def test_make_loss_fn_produces_finite_loss() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    loss_fn = make_loss_fn(None)
    x1, labels = next(iter(_gaussian_loader(32, 16)))
    opt_losses, _ = loss_fn(model, None, x1, labels, step=0)
    total = sum(opt_losses.values())
    assert math.isfinite(float(total.item()))


def test_train_logs_val_loss_when_on_eval_returns_metric() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    records: list[dict] = []
    train(
        model,
        _gaussian_loader(96, 32),  # 3 batches / epoch
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: records.append(r),
        eval_every_epochs=1,
        on_eval=lambda m, epoch: 0.5,
    )
    assert any(r.get("val_loss") == 0.5 for r in records)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py::test_make_loss_fn_produces_finite_loss packages/physics-informed-flow-map/tests/test_train.py::test_train_logs_val_loss_when_on_eval_returns_metric -v`
Expected: FAIL — `make_loss_fn` is not importable; and `train` does not yet log a `val_loss` record.

- [ ] **Step 3: Extract `make_loss_fn` and use it in `train()`**

In `train.py`, add the factory above the `train` function (after `_fm_loss_cfg`):
```python
def make_loss_fn(num_classes: int | None) -> Callable[..., Any]:
    """The pure-FM consistency loss used for both training and validation."""
    return get_consistency_loss_fn(_fm_loss_cfg(num_classes or 0), Linear(t_max=1.0))
```

Inside `train()`, replace these two lines:
```python
    label_dim = num_classes or 0
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
```
with:
```python
    loss_fn = make_loss_fn(num_classes)
```

- [ ] **Step 4: Log `val_loss` in the eval block**

In `train()`, replace the eval/best-tracking block:
```python
        is_best = False
        if (
            on_eval is not None
            and eval_every_epochs
            and (epoch + 1) % eval_every_epochs == 0
        ):
            eval_model = ema_module() or model
            metric = on_eval(eval_model, epoch)
            model.train()
            if metric is not None and metric < best_metric:
                best_metric = metric
                is_best = True
```
with:
```python
        is_best = False
        if (
            on_eval is not None
            and eval_every_epochs
            and (epoch + 1) % eval_every_epochs == 0
        ):
            eval_model = ema_module() or model
            metric = on_eval(eval_model, epoch)
            model.train()
            if metric is not None:
                if log is not None:
                    log(step=step, epoch=epoch, val_loss=metric)
                if metric < best_metric:
                    best_metric = metric
                    is_best = True
```

- [ ] **Step 5: Run the full `test_train.py` to verify pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: PASS — the two new tests plus all pre-existing tests (including `test_train_hooks_fire_on_epoch_cadence`, which passes no `log`, so best-tracking still keys off the returned metric and is unaffected).

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py \
        packages/physics-informed-flow-map/tests/test_train.py
git commit -m "feat(train): reusable make_loss_fn + log val_loss on eval epochs"
```

---

### Task 2: Held-out splits via `build_val()` (`flow_matching/datasets.py`)

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py`
- Test: `packages/physics-informed-flow-map/tests/test_datasets.py`
- Test: `packages/physics-informed-flow-map/tests/test_openfwi.py`

**Interfaces:**
- Produces: `build_val(self) -> Dataset` on `GaussiansDatasetConfig`, `MNISTDatasetConfig`, and `OpenFWIDatasetConfig` (returns the disjoint held-out split). New fields: `GaussiansDatasetConfig.val_samples: int = 10000`; `OpenFWIDatasetConfig.val_fraction: float = 0.1`. `OpenFWIDatasetConfig.build()` now returns the training `Subset` (not the full dataset).

- [ ] **Step 1: Write the failing tests**

In `tests/test_datasets.py`, change the import line
```python
from physics_informed_flow_map.flow_matching.datasets import DATASETS
```
to
```python
from physics_informed_flow_map.flow_matching.datasets import (
    DATASETS,
    GaussiansDatasetConfig,
)
```
and append:
```python
def test_build_val_shapes(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    if cfg.requires_download:
        pytest.skip(f"{dataset_name} requires download; exercised by the live run")
    ds = cfg.build_val()
    x1, label = ds[0]
    assert x1.shape == cfg.shape
    if cfg.num_classes is None:
        assert int(label) == 0
    else:
        assert 0 <= int(label) < cfg.num_classes


def test_gaussians_build_val_is_distinct() -> None:
    cfg = GaussiansDatasetConfig(n_samples=64, val_samples=32)
    train_ds = cfg.build()
    val_ds = cfg.build_val()
    assert len(val_ds) == 32
    assert not torch.equal(train_ds[0][0], val_ds[0][0])  # distinct seed -> different draw
```

In `tests/test_openfwi.py`, add an import for `Subset` (alongside the existing imports) and append a split test that reuses the existing `_write_family` fixture helper:
```python
from torch.utils.data import Subset

from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig


def test_openfwi_config_split_is_disjoint(tmp_path: Path) -> None:
    _write_family(tmp_path, "FlatVel_A", "model", n_files=2, rows=4)  # 8 maps total
    cfg = OpenFWIDatasetConfig(
        data_dir=str(tmp_path), families=["FlatVel_A"], val_fraction=0.25
    )
    train_ds = cfg.build()
    val_ds = cfg.build_val()
    assert len(train_ds) + len(val_ds) == 8
    assert len(val_ds) == max(1, int(0.25 * 8))  # 2
    assert isinstance(train_ds, Subset) and isinstance(val_ds, Subset)
    assert set(train_ds.indices).isdisjoint(set(val_ds.indices))
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_datasets.py packages/physics-informed-flow-map/tests/test_openfwi.py -v`
Expected: FAIL — `build_val` does not exist; `GaussiansDatasetConfig` has no `val_samples`; `OpenFWIDatasetConfig` has no `val_fraction` and `build()` returns the full dataset (not a `Subset`).

- [ ] **Step 3: Add `Subset` import and the MNIST `train` flag**

In `datasets.py`, change the import:
```python
from torch.utils.data import Dataset, TensorDataset
```
to:
```python
from torch.utils.data import Dataset, Subset, TensorDataset
```

Give `_make_mnist` a `train` flag:
```python
def _make_mnist(
    data_dir: str = "data", image_size: int = 32, train: bool = True
) -> Dataset:
    transform = T.Compose(
        [T.Resize(image_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    )
    return cast(
        Dataset,
        torchvision.datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        ),
    )
```

- [ ] **Step 4: Add `build_val` to the gaussians and mnist configs**

In `GaussiansDatasetConfig`, add the `val_samples` field after `n_samples`:
```python
    n_samples: int = 100_000
    val_samples: int = 10_000
```
and add `build_val` after `build`:
```python
    def build_val(self) -> Dataset:
        return _make_gaussians(
            self.val_samples, self.n_modes, self.radius, self.std, seed=1
        )
```

In `MNISTDatasetConfig`, add `build_val` after `build`:
```python
    def build_val(self) -> Dataset:
        return _make_mnist(self.data_dir, self.image_size, train=False)
```

- [ ] **Step 5: Add the OpenFWI split**

In `OpenFWIDatasetConfig`, add the `val_fraction` field after `resolution`:
```python
    resolution: int = 64
    val_fraction: float = 0.1
```
and replace the `build` method with the split helper + `build`/`build_val`:
```python
    def _split(self) -> tuple[OpenFWIVelocityDataset, list[int], list[int]]:
        full = OpenFWIVelocityDataset(
            Path(self.data_dir), self.families, self.resolution
        )
        n = len(full)
        n_val = max(1, int(self.val_fraction * n))
        perm = torch.randperm(
            n, generator=torch.Generator().manual_seed(0)
        ).tolist()
        return full, perm[n_val:], perm[:n_val]

    def build(self) -> Dataset:
        full, train_idx, _ = self._split()
        return Subset(full, train_idx)

    def build_val(self) -> Dataset:
        full, _, val_idx = self._split()
        return Subset(full, val_idx)
```

- [ ] **Step 6: Run the dataset tests, then the full suite**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_datasets.py packages/physics-informed-flow-map/tests/test_openfwi.py -v`
Expected: PASS — `test_build_val_shapes` passes for `gaussians` and skips `mnist`/`openfwi`; `test_gaussians_build_val_is_distinct` and `test_openfwi_config_split_is_disjoint` pass.

Run: `uv run pytest packages/physics-informed-flow-map/tests/ -q`
Expected: PASS (the existing registry `test_build_shapes` still passes — `Subset` for openfwi is download-skipped; mnist/gaussians unchanged).

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/datasets.py \
        packages/physics-informed-flow-map/tests/test_datasets.py \
        packages/physics-informed-flow-map/tests/test_openfwi.py
git commit -m "feat(datasets): build_val held-out splits for all three datasets"
```

---

### Task 3: Wire val loss through `run.py`

**Files:**
- Modify: `experiments/0001_flow_matching/run.py`

**Interfaces:**
- Consumes: `make_loss_fn` (Task 1); `cfg.dataset.build_val()` (Task 2).
- Produces: a runnable `main()` that builds a val loader, computes `val_loss` per eval and at the end, and includes `val_loss` in the final summary.

This task is integration glue in `experiments/` (not type-checked by mypy and not unit-tested — no test executes `main()`). It is verified by the compose-import suite plus a real end-to-end smoke run (gaussians, CPU, seconds).

- [ ] **Step 1: Import `make_loss_fn`**

In `experiments/0001_flow_matching/run.py`, change:
```python
from physics_informed_flow_map.flow_matching.train import train
```
to:
```python
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train
```

- [ ] **Step 2: Build the val loader + `compute_val_loss` helper**

Immediately after the `model = build_model(...).to(device)` block (and before the `on_eval` definition), insert:
```python
    val_loader = torch.utils.data.DataLoader(
        cfg.dataset.build_val(),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    val_loss_fn = make_loss_fn(cfg.dataset.num_classes)

    @torch.no_grad()
    def compute_val_loss(m: BaseModel) -> float:
        m.eval()
        total, n = 0.0, 0
        for xb, lb in val_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            opt_losses, _ = val_loss_fn(m, None, xb, lb, step=0)
            total += float(sum(opt_losses.values()).item())
            n += 1
        return total / max(n, 1)
```

- [ ] **Step 3: Make `on_eval` return the val loss**

Replace the `on_eval` body's tail. Change:
```python
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p)
        if isinstance(cfg.dataset, GaussiansDatasetConfig):
            # 'best'-tracking metric: deliberately the cheaper n_eval_viz budget,
            # not the final verdict's n_eval_samples.
            ref = real_reference(dataset, cfg.sampling.n_eval_viz, device)
            return energy_distance(s, ref)
        return None
```
to:
```python
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p)
        return compute_val_loss(m)
```

- [ ] **Step 4: Add the final val loss to the summary**

Replace the final summary block. Change:
```python
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        run.finish(energy_distance=metric, final_loss=final_loss)
    else:
        run.finish(final_loss=final_loss)
```
to:
```python
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    final_val_loss = compute_val_loss(eval_model)
    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        run.finish(
            energy_distance=metric, final_loss=final_loss, val_loss=final_val_loss
        )
    else:
        run.finish(final_loss=final_loss, val_loss=final_val_loss)
```

- [ ] **Step 5: Run the full package suite (compose imports `run.py`)**

Run: `uv run pytest packages/physics-informed-flow-map/tests/ -q`
Expected: PASS — the compose tests import the edited `run.py` (so the `make_loss_fn` import and module-level code must be valid).

- [ ] **Step 6: Real end-to-end smoke run**

Run (gaussians, 1 epoch, exercises per-eval AND final `val_loss`):
```bash
WANDB_MODE=disabled uv run python experiments/0001_flow_matching/run.py \
    experiment=smoke training.eval_every_epochs=1
```
Expected: the run completes and prints a final summary line containing `val_loss=` (e.g. `[0001_flow_matching] energy_distance=… final_loss=… val_loss=…`). This confirms `build_val()` → `val_loader` → `compute_val_loss` → `on_eval`/final summary all wire together end-to-end.

- [ ] **Step 7: Commit**

```bash
git add experiments/0001_flow_matching/run.py
git commit -m "feat(0001): compute + log held-out val_loss; drive best checkpoint"
```

---

## Self-Review

**Spec coverage:**
- Section A (`make_loss_fn` factory) → Task 1. ✅
- Section B (log `val_loss` in eval block, best-tracking unchanged) → Task 1. ✅
- Section C (`build_val()` on all three configs; gaussians distinct seed + `val_samples`; mnist test set; openfwi seeded `Subset` split + `val_fraction`; `build()` returns train split) → Task 2. ✅
- Section D (`val_loader`, `compute_val_loss`, `on_eval` returns val loss, final `val_loss` in summary) → Task 3. ✅
- Testing (make_loss_fn finite; train logs val_loss; registry build_val shapes; gaussians distinct; openfwi split disjoint) → Tasks 1–2. ✅
- Out-of-scope (physics residuals, FID, energy-vs-heldout, early stopping) → not implemented. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✅

**Type consistency:** `make_loss_fn(num_classes)` (Task 1) is imported and called in Task 3 with `cfg.dataset.num_classes`. `build_val()` (Task 2) is called in Task 3's `val_loader`. `compute_val_loss(m) -> float` feeds `on_eval`'s `float | None` return and the final `run.finish(val_loss=…)`. The `loss_fn(model, None, x1, labels, step=…)` call shape matches between `train()`, the `make_loss_fn` test, and `compute_val_loss`. `Subset` is imported in both `datasets.py` (Task 2 impl) and `test_openfwi.py` (Task 2 test). ✅
