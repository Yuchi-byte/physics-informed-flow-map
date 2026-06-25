# Configurable EMA of Velocity Weights — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable exponential-moving-average (EMA) of the velocity-model weights to `0001_flow_matching`, sampling/eval from the EMA weights and uploading both raw and EMA checkpoints.

**Architecture:** EMA is implemented with the stdlib `torch.optim.swa_utils.AveragedModel` inside our own `flow_matching.train` loop (no edit to any reference package). `train()` returns the EMA model alongside the history; the experiment `run.py` samples from it for the final eval and saves/uploads both the raw and EMA checkpoints. A nested `training.ema` pydantic config (off by default) drives it, enabled in the tuned `mnist`/`gaussians` Hydra variants.

**Tech Stack:** Python 3.12, PyTorch (`torch.optim.swa_utils`), pydantic v2, Hydra, Weights & Biases, pytest, uv workspace.

## Global Constraints

- **Do NOT edit any reference package** (`packages/mfm-meta-flow-map-main/`, `packages/PhysicsInformedDiffusionModels-main/`, `packages/PBFM/`). EMA uses only stdlib `torch.optim.swa_utils`.
- EMA `decay` is constrained to the open interval `(0, 1)`; `warmup_steps >= 0`; `enabled` defaults to `False`.
- `train()` changes return type from `list[dict[str, float]]` to `tuple[list[dict[str, float]], BaseModel | None]`. All in-repo callers (tests + `run.py`) are updated within this plan. `run.py`'s `main()` is transiently inconsistent between Task 1 and Task 3, but **no test executes `main()`** (the compose tests only import the module and validate config), so the suite stays green throughout.
- Tests that touch wandb run under `WANDB_MODE=disabled` (the `test_run.py` autouse fixture already sets this).
- `final_loss` reported to the verdict stays the **raw training loss** (`history[-1]["total"]`). Only the *sampled* outputs (gaussians energy distance, `samples.png`) switch to the EMA model when enabled.
- Artifact naming: raw checkpoint → wandb artifact `{dataset}-model`, EMA checkpoint → `{dataset}-model-ema`, both with the **same aliases**. Local files: `step_<step>.pt` (raw) and `step_<step>_ema.pt` (EMA).
- EMA is **off** in the base config, **on** in the `mnist` and `gaussians` experiment variants.
- `mypy` in pre-commit is scoped to the package (`packages/physics-informed-flow-map/`), not `experiments/`. Do not add type-ignore noise to `experiments/0001_flow_matching/run.py` for the untyped `@hydra.main` decorator.

---

### Task 1: EMA in the training loop (`flow_matching/train.py`)

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py`
- Test: `packages/physics-informed-flow-map/tests/test_train.py`

**Interfaces:**
- Consumes: `mfm.models.base_model.BaseModel` (already imported), `torch.optim.swa_utils.{AveragedModel, get_ema_multi_avg_fn}`.
- Produces: new `train()` signature
  ```python
  def train(
      model: BaseModel, loader: DataLoader, *, n_epochs: int, lr: float,
      device: torch.device, num_classes: int | None = None,
      log: Callable[..., None] | None = None,
      ema_enabled: bool = False, ema_decay: float = 0.999, ema_warmup_steps: int = 0,
      eval_every_epochs: int = 0, on_eval: Callable[[BaseModel, int], float | None] | None = None,
      ckpt_every_epochs: int = 0, on_checkpoint: Callable[..., None] | None = None,
  ) -> tuple[list[dict[str, float]], BaseModel | None]:
  ```
  The `on_checkpoint` callback is now invoked with an extra keyword: `on_checkpoint(model, epoch, *, is_best, is_final, ema_model=<BaseModel | None>)`. Returns `(history, ema_model)` where `ema_model` is the averaged `BaseModel` if at least one EMA update happened, else `None`.

- [ ] **Step 1: Update existing test callers and hook signatures**

In `tests/test_train.py`:
- In `test_train_runs_and_logs`, change `history = train(` to `history, _ = train(`.
- In `test_train_hooks_fire_on_epoch_cadence`, change the `on_checkpoint` signature to accept the new kwarg:
  ```python
  def on_checkpoint(
      m: object, epoch: int, *, is_best: bool, is_final: bool, ema_model: object = None
  ) -> None:
      ckpts.append((epoch, is_best, is_final))
  ```
- In `test_train_checkpoint_cadence_without_eval`, change the `on_checkpoint` signature the same way:
  ```python
  def on_checkpoint(
      m: object, epoch: int, *, is_best: bool, is_final: bool, ema_model: object = None
  ) -> None:
      ckpts.append((epoch, is_best, is_final))
  ```
- Leave `test_train_logs_decomposed_losses` as-is (it does not capture the return value; discarding the tuple is fine).

- [ ] **Step 2: Write the failing EMA tests**

Append to `tests/test_train.py`:

```python
def test_train_ema_disabled_returns_none() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    history, ema_model = train(
        model,
        _gaussian_loader(96, 32),  # 3 batches -> 3 steps
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    assert ema_model is None
    assert len(history) == 3


def test_train_ema_enabled_returns_distinct_module() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    _, ema_model = train(
        model,
        _gaussian_loader(96, 32),  # 3 steps -> EMA lags the raw weights
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        ema_enabled=True,
        ema_decay=0.9,
    )
    assert ema_model is not None
    assert ema_model is not model
    raw = dict(model.named_parameters())
    differs = any(
        not torch.equal(p.detach(), raw[name].detach())
        for name, p in ema_model.named_parameters()
        if name in raw
    )
    assert differs


def test_train_ema_warmup_beyond_run_returns_none() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    _, ema_model = train(
        model,
        _gaussian_loader(96, 32),  # 3 steps total
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        ema_enabled=True,
        ema_warmup_steps=100,  # never reached -> EMA never updates
    )
    assert ema_model is None
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: the three new tests FAIL (`train()` still returns a list, so tuple-unpacking raises or `ema_model` is undefined), while `test_train_runs_and_logs` now also fails on the unpack until the implementation lands.

- [ ] **Step 4: Implement EMA in `train.py`**

Change the imports near the top of `train.py`:
```python
from typing import Any, Callable, cast
```
and add, below the existing `torch` import:
```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
```

Replace the `train(...)` function body (everything from `def train(` to the final `return history`) with:

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
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[[BaseModel, int], float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> tuple[list[dict[str, float]], BaseModel | None]:
    label_dim = num_classes or 0
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    model.train()

    ema: AveragedModel | None = None
    if ema_enabled:
        ema = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True
        )

    def ema_module() -> BaseModel | None:
        """The averaged model once at least one EMA update has happened, else None."""
        if ema is not None and int(ema.n_averaged.item()) > 0:
            return cast(BaseModel, ema.module)
        return None

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

            if ema is not None and step >= ema_warmup_steps:
                ema.update_parameters(model)

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
            eval_model = ema_module() or model
            metric = on_eval(eval_model, epoch)
            model.train()
            if metric is not None and metric < best_metric:
                best_metric = metric
                is_best = True

        if on_checkpoint is not None and (
            is_best or (ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0)
        ):
            on_checkpoint(
                model, epoch, is_best=is_best, is_final=False, ema_model=ema_module()
            )

    if on_checkpoint is not None:
        on_checkpoint(
            model, n_epochs - 1, is_best=False, is_final=True, ema_model=ema_module()
        )
    return history, ema_module()
```

Note `eval_model = ema_module() or model`: `ema_module()` returns either a `BaseModel` (truthy) or `None`, so the raw `model` is used as the fallback. `sample()` (called inside `on_eval`) sets the model it receives to `.eval()`; the unconditional `model.train()` afterward restores the raw model's training mode.

- [ ] **Step 5: Run all `test_train.py` tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_train.py -v`
Expected: PASS for all tests (the 4 pre-existing tests + the 3 new EMA tests).

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/train.py \
        packages/physics-informed-flow-map/tests/test_train.py
git commit -m "feat(train): EMA of velocity weights via stdlib AveragedModel"
```

---

### Task 2: Checkpoint suffix in the harness (`experiment/run.py`)

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py`
- Test: `packages/physics-informed-flow-map/tests/test_run.py`

**Interfaces:**
- Produces: `Run.save_checkpoint(self, model, step, *, suffix: str = "", **meta) -> Path` writing to `checkpoints/step_<step><suffix>.pt`. Existing callers passing only `**meta` keywords are unaffected.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_run.py::test_save_checkpoint_suffix -v`
Expected: FAIL with `TypeError` (`save_checkpoint()` got an unexpected keyword argument `suffix`).

- [ ] **Step 3: Implement the `suffix` parameter**

In `run.py`, replace the `save_checkpoint` method:

```python
    def save_checkpoint(
        self, model: torch.nn.Module, step: int, *, suffix: str = "", **meta: Any
    ) -> Path:
        """Save ``model`` state (+ metadata) to ``checkpoints/step_<step><suffix>.pt``."""
        path = self.ckpt_dir / f"step_{step}{suffix}.pt"
        torch.save({"model": model.state_dict(), "step": step, **meta}, path)
        return path
```

- [ ] **Step 4: Run the test (and the rest of the file) to verify pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_run.py -v`
Expected: PASS (`test_run_lifecycle` and `test_save_checkpoint_suffix`).

- [ ] **Step 5: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py \
        packages/physics-informed-flow-map/tests/test_run.py
git commit -m "feat(harness): save_checkpoint suffix for EMA checkpoints"
```

---

### Task 3: EMA config + `run.py` wiring (`experiments/0001_flow_matching/run.py`)

**Files:**
- Modify: `experiments/0001_flow_matching/run.py`
- Test: `packages/physics-informed-flow-map/tests/test_experiment_conf.py`

**Interfaces:**
- Consumes: `train()` returning `(history, ema_model)` (Task 1); `Run.save_checkpoint(..., suffix=...)` (Task 2).
- Produces: module-level `EmaConfig` and an `ema: EmaConfig` field on `TrainingConfig`, both importable from the run module via the existing `_load_run_module()` helper. This task re-integrates `main()` so it is runnable again.

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_experiment_conf.py`:

```python
def test_ema_config_defaults() -> None:
    mod = _load_run_module()
    tcfg = mod.TrainingConfig()
    assert tcfg.ema.enabled is False
    assert tcfg.ema.decay == 0.999
    assert tcfg.ema.warmup_steps == 0


def test_ema_decay_out_of_range_rejected() -> None:
    mod = _load_run_module()
    with pytest.raises(ValidationError):
        mod.EmaConfig(decay=1.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py::test_ema_config_defaults packages/physics-informed-flow-map/tests/test_experiment_conf.py::test_ema_decay_out_of_range_rejected -v`
Expected: FAIL with `AttributeError` (`module 'fm_run' has no attribute 'EmaConfig'` / `TrainingConfig` has no `ema`).

- [ ] **Step 3: Add `EmaConfig` and the `ema` field**

In `experiments/0001_flow_matching/run.py`, add the `EmaConfig` class immediately above `class TrainingConfig` and add the `ema` field to `TrainingConfig`:

```python
class EmaConfig(Config):
    enabled: bool = False
    decay: float = Field(0.999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = 1e-3
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(0, ge=0)
    artifact_every_epochs: int = Field(0, ge=0)
    ema: EmaConfig = EmaConfig()
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py::test_ema_config_defaults packages/physics-informed-flow-map/tests/test_experiment_conf.py::test_ema_decay_out_of_range_rejected -v`
Expected: PASS.

- [ ] **Step 5: Wire EMA through `main()`**

In `experiments/0001_flow_matching/run.py`, replace the `on_checkpoint` closure with one that accepts and uploads the EMA checkpoint:

```python
    def on_checkpoint(
        m: BaseModel,
        epoch: int,
        *,
        is_best: bool = False,
        is_final: bool = False,
        ema_model: BaseModel | None = None,
    ) -> None:
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
        path = run.save_checkpoint(
            m, epoch, dataset=cfg.dataset.name, config=cfg.dump()
        )
        if aliases:
            run.log_artifact(path, name=f"{cfg.dataset.name}-model", aliases=aliases)
        if ema_model is not None:
            ema_path = run.save_checkpoint(
                ema_model,
                epoch,
                suffix="_ema",
                dataset=cfg.dataset.name,
                config=cfg.dump(),
            )
            if aliases:
                run.log_artifact(
                    ema_path, name=f"{cfg.dataset.name}-model-ema", aliases=aliases
                )
```

Then replace the `train(...)` call and the final-sampling block. Change:

```python
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
```

to:

```python
    history, ema_model = train(
        model,
        loader,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=device,
        num_classes=cfg.dataset.num_classes,
        log=run.log,
        ema_enabled=cfg.training.ema.enabled,
        ema_decay=cfg.training.ema.decay,
        ema_warmup_steps=cfg.training.ema.warmup_steps,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
    final_loss = history[-1]["total"]
    eval_model = ema_model if ema_model is not None else model

    samples = sample(
        eval_model,
        cfg.sampling.n_eval_samples,
        cfg.dataset.shape,
        sampler_steps=cfg.sampling.sampler_steps,
        device=device,
    )
```

Leave the verdict block below unchanged — it already consumes `samples` and `final_loss`.

- [ ] **Step 6: Run the full package test suite to verify nothing regressed**

Run: `uv run pytest packages/physics-informed-flow-map/tests/ -q`
Expected: PASS (all tests, including the import-and-validate compose tests which exercise the edited `run.py` module).

- [ ] **Step 7: Commit**

```bash
git add experiments/0001_flow_matching/run.py \
        packages/physics-informed-flow-map/tests/test_experiment_conf.py
git commit -m "feat(0001): wire EMA config through run.py; sample + upload EMA weights"
```

---

### Task 4: Hydra config — enable EMA in tuned variants

**Files:**
- Modify: `experiments/0001_flow_matching/conf/training/default.yaml`
- Modify: `experiments/0001_flow_matching/conf/experiment/gaussians.yaml`
- Modify: `experiments/0001_flow_matching/conf/experiment/mnist.yaml`
- Test: `packages/physics-informed-flow-map/tests/test_experiment_conf.py`

**Interfaces:**
- Consumes: `TrainingConfig.ema` (Task 3) so the composed `ema:` block validates under `extra="forbid"`.

- [ ] **Step 1: Write the failing compose test**

Append to `tests/test_experiment_conf.py`:

```python
@pytest.mark.parametrize(
    "variant,ema_enabled",
    [("gaussians", True), ("mnist", True), ("smoke", False)],
)
def test_compose_ema_enabled(variant: str, ema_enabled: bool) -> None:
    cfg_cls = _load_run_module().FlowMatchingConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=[f"experiment={variant}"])
    cfg = cfg_cls.from_dictconfig(dcfg)
    assert cfg.training.ema.enabled is ema_enabled
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py::test_compose_ema_enabled -v`
Expected: FAIL — for `gaussians`/`mnist` the composed `ema.enabled` is still `False` (the variants do not yet set it).

- [ ] **Step 3: Add the EMA block to the base training config**

Replace the contents of `experiments/0001_flow_matching/conf/training/default.yaml` with:

```yaml
n_epochs: 10
batch_size: 256
lr: 0.001
eval_every_epochs: 0
ckpt_every_epochs: 0
artifact_every_epochs: 0
ema:
  enabled: false
  decay: 0.999
  warmup_steps: 0
```

- [ ] **Step 4: Enable EMA in the gaussians variant**

Replace the contents of `experiments/0001_flow_matching/conf/experiment/gaussians.yaml` with:

```yaml
# @package _global_
defaults:
  - override /model: mlp
  - override /dataset: gaussians

gate: 0.5
training:
  n_epochs: 100
  ema:
    enabled: true
sampling:
  sampler_steps: 200
```

- [ ] **Step 5: Enable EMA in the mnist variant**

Replace the contents of `experiments/0001_flow_matching/conf/experiment/mnist.yaml` with:

```yaml
# @package _global_
defaults:
  - override /model: dit
  - override /dataset: mnist

gate: 240.0
training:
  n_epochs: 100
  batch_size: 128
  ema:
    enabled: true
sampling:
  sampler_steps: 200
```

- [ ] **Step 6: Run the compose test (and the full suite) to verify pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_conf.py -v`
Expected: PASS — `test_compose_ema_enabled` confirms `gaussians`/`mnist` → `True`, `smoke` → `False`; the existing `test_compose_validates` parametrization still passes (deep-merge keeps `n_epochs`/`batch_size`).

Then run the whole suite once: `uv run pytest packages/physics-informed-flow-map/tests/ -q` — expected PASS.

- [ ] **Step 7: Commit**

```bash
git add experiments/0001_flow_matching/conf/training/default.yaml \
        experiments/0001_flow_matching/conf/experiment/gaussians.yaml \
        experiments/0001_flow_matching/conf/experiment/mnist.yaml \
        packages/physics-informed-flow-map/tests/test_experiment_conf.py
git commit -m "feat(0001): enable EMA in mnist + gaussians variants"
```

---

## Self-Review

**Spec coverage:**
- Section A (EMA in `train.py`, tuple return, `ema_module`, eval/checkpoint hooks) → Task 1. ✅
- Section B (`save_checkpoint` suffix) → Task 2. ✅
- Section C (`EmaConfig`, `TrainingConfig.ema`, `main()` wiring: both artifacts, `eval_model`, `final_loss` stays raw) → Task 3. ✅
- Section D (Hydra yaml: base off, mnist/gaussians on) → Task 4. ✅
- Testing section (train tuple-unpack updates + 3 EMA tests; suffix test; EmaConfig defaults/validation; compose ema assertions) → Tasks 1–4. ✅
- Migration note (transient stale `main()`, no test executes it) → captured in Global Constraints and Task 1/3 ordering. ✅
- Out-of-scope (resume-from-checkpoint, scaffolder, bias-correction) → not implemented. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✅

**Type consistency:** `train()` returns `tuple[list[dict[str, float]], BaseModel | None]` (Task 1) and Task 3 unpacks `history, ema_model`. `ema_module()` returns `BaseModel | None`; `on_checkpoint(..., ema_model=BaseModel | None)` matches run.py's closure signature. `save_checkpoint(..., suffix="")` (Task 2) is called with `suffix="_ema"` (Task 3). `EmaConfig.{enabled,decay,warmup_steps}` (Task 3) match the yaml keys (Task 4). ✅
