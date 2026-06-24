# Hierarchical Configs + Discriminated Unions + Epoch Training — Design

**Date:** 2026-06-25
**Status:** Approved for planning

## Goal

Three coupled changes to `0001_flow_matching` and the package's model/dataset layer:

1. **Epoch-based training + tqdm.** Replace the fixed `n_steps` budget with
   `n_epochs` (full passes over the dataset), with a per-epoch tqdm progress bar.
   Eval/checkpoint hooks fire on an **epoch** cadence.
2. **Hierarchical config.** Split the flat `FlowMatchingConfig` into nested
   sub-configs: `model`, `dataset`, `training`, `sampling` — backed by a
   multi-group Hydra `conf/` tree (`conf/model/*.yaml`, `conf/dataset/*.yaml`, …).
3. **Discriminated-union model & dataset selection.** `model` and `dataset` become
   pydantic discriminated unions (`Annotated[A | B, Field(discriminator=...)]`),
   so each variant carries its own typed args. Mirrors the user's
   `diffusion_co_design/rware/schema.py` pattern.

## Motivation

The current config is flat (`mlp_width`, `dit_hidden`, … all siblings), so adding
a model means widening one struct and threading more positional knobs through
`build_model`. Discriminated unions make each model/dataset variant own its args
and make invalid combinations unrepresentable (validated at parse time). Epoch
training matches how these models actually converge (full-dataset passes) and the
tqdm bar gives live feedback on the now-longer runs.

## Architecture

### A. Discriminated model config — `flow_matching/models.py`

```python
from typing import Annotated, Literal
from pydantic import Field
from physics_informed_flow_map.experiment import Config

class MLPModelConfig(Config):
    kind: Literal["mlp"] = "mlp"
    width: int = 256
    depth: int = 4

class DiTModelConfig(Config):
    kind: Literal["dit"] = "dit"
    hidden: int = 128
    depth: int = 4
    num_heads: int = 4
    patch_size: int = 4

ModelConfig = Annotated[MLPModelConfig | DiTModelConfig, Field(discriminator="kind")]
```

`build_model` dispatches on the config type (replaces the keyword-knob signature):

```python
def build_model(
    shape: tuple[int, ...], num_classes: int | None, cfg: ModelConfig
) -> BaseModel:
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
        dit = DiTMFM(..., input_size=h, patch_size=cfg.patch_size, in_channels=c,
                     hidden_size=cfg.hidden, depth=cfg.depth, num_heads=cfg.num_heads,
                     label_dim=num_classes or 1, ...)  # other args unchanged from today
        return SIModelWrapper(dit, Linear(t_max=1.0), use_parametrization=False)
    raise ValueError(f"unsupported model config {cfg!r}")  # unreachable (union exhaustive)
```

`VelocityMLP` / `TimeEmbedding` are unchanged.

**Test registry** (replaces the old `ModelSpec`/`MODELS`):

```python
@dataclass(frozen=True)
class ModelCase:
    config: ModelConfig
    sample_shape: tuple[int, ...]
    num_classes: int | None

MODELS: dict[str, ModelCase] = {
    "mlp": ModelCase(MLPModelConfig(), (2,), None),
    "dit": ModelCase(DiTModelConfig(), (1, 32, 32), 10),
}
```

### B. Discriminated dataset config — `flow_matching/datasets.py`

Each dataset config owns its build + metadata (replaces `DatasetSpec`/`DATASETS`).
The module-level `_make_gaussians`, `_make_mnist`, `_viz_scatter`, `_viz_grid`
helpers stay and are delegated to.

```python
class GaussiansDatasetConfig(Config):
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

`requires_download`/`shape`/`num_classes` are computed properties (not fields), so
they are intrinsic to the variant (not yaml-overridable — `extra="forbid"` rejects
them as input) yet still readable by the tests. MNIST's `shape` tracks `image_size`.

### C. Top config + cross-validation — `experiments/0001_flow_matching/run.py`

```python
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
```

`build_model(cfg.dataset.shape, cfg.dataset.num_classes, cfg.model)`; the
DataLoader uses `cfg.training.batch_size`; sampling uses `cfg.sampling.*`. The
monitored-metric branch checks `isinstance(cfg.dataset, GaussiansDatasetConfig)`.

### D. Epoch-based training + tqdm — `flow_matching/train.py`

```python
from tqdm.auto import tqdm

def train(model, loader, *, n_epochs, lr, device, num_classes=None,
          log=None,
          eval_every_epochs=0, on_eval=None,        # on_eval(model, epoch) -> float | None
          ckpt_every_epochs=0, on_checkpoint=None,  # on_checkpoint(model, epoch, *, is_best, is_final)
          ) -> list[dict[str, float]]:
    ...
    step = 0
    best_metric = math.inf
    for epoch in range(n_epochs):
        for x1, labels in tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False):
            # ... one optimizer step (unchanged loss math) ...
            rec = {"step": float(step), "epoch": float(epoch), "total": ...}
            for name, value in opt_losses.items():
                rec[name] = float(value.item())
            history.append(rec)
            if log is not None:
                log(**rec)
            step += 1

        is_best = False
        if on_eval is not None and eval_every_epochs and (epoch + 1) % eval_every_epochs == 0:
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

- Hooks receive `epoch` (cadence + filenames). Each per-batch `log` record gains an
  `epoch` field alongside `step`/`total`/loss terms.
- wandb stepping: scalars carry the global `step`; sample images are logged with
  **no explicit step** (`run.log_image(key, path)`), so they attach to wandb's
  current step under their own key — no monotonic-step collisions, and the prior
  explicit-`step` final-image hack is removed (distinct keys `samples` /
  `samples_final` make it unnecessary).
- The data-iterator-cycling logic is gone (we iterate the loader directly each
  epoch). `drop_last=True` stays in the entry point so batch shapes are uniform.

### E. Hydra multi-group tree — `experiments/0001_flow_matching/conf/`

```
conf/
  config.yaml
  model/{mlp,dit}.yaml
  dataset/{gaussians,mnist}.yaml
  training/default.yaml
  sampling/default.yaml
  experiment/{gaussians,mnist,smoke}.yaml
```

`config.yaml`:

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

Group files carry just their node's fields (Hydra places `conf/model/mlp.yaml`
under the `model` package automatically), e.g. `conf/model/dit.yaml`:

```yaml
kind: dit
hidden: 128
depth: 4
num_heads: 4
patch_size: 4
```

Experiment files use the standard Hydra experiment pattern (`# @package _global_`
+ a `defaults` list that overrides the group selections), e.g.
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

`smoke.yaml` selects mlp+gaussians, `training.n_epochs: 1`,
`sampling.n_eval_samples: 256`, `gate: 1000000000.0`. `gaussians.yaml` selects
mlp+gaussians with `gate: 0.5` and a modest `training.n_epochs` (e.g. 5).

CLI: `run.py experiment=mnist training.n_epochs=80`, or mix groups directly:
`run.py model=dit dataset=mnist training.eval_every_epochs=5`.

### F. Dependencies / tooling

- Add `tqdm` to `packages/physics-informed-flow-map/pyproject.toml` deps; add
  `tqdm.*` to the root mypy `ignore_missing_imports` overrides.

## Testing

Run under `WANDB_MODE=disabled` where wandb is touched (only the harness test).

- **`test_models.py`** — parametrize over `MODELS` (`ModelCase`); build each via
  `build_model(case.sample_shape, case.num_classes, tiny_config[kind])` and do a
  forward + backward. Keep a `test_*` asserting mlp on an image shape and dit on a
  vector shape each raise `ValueError`.
- **`test_datasets.py`** — parametrize over `DATASETS` (`DatasetConfig`); assert
  `cfg.shape`/`cfg.num_classes` metadata, `cfg.build()` shapes (skip
  `requires_download`), and `cfg.visualize` writes a file (synthetic tensors).
- **`test_train.py`** — switch to `n_epochs`; assert the per-batch `log` record has
  `epoch` + `total` + `fm_loss`; assert epoch-cadence hooks: with `n_epochs=4,
  eval_every_epochs=2` a strictly-decreasing `on_eval` marks `is_best`, and
  `on_checkpoint` fires on cadence plus exactly one `is_final=True`; add a
  `ckpt_every_epochs`-only (no eval) cadence test.
- **`test_sample.py`** — update `build_model` call sites to the new signature.
- **`test_experiment_conf.py`** — update compose assertions to the nested shape
  (`cfg.dataset.name`, `cfg.model.kind`, `cfg.training.n_epochs`); cover an
  invalid combo (`model=dit dataset=gaussians`) raising `ValidationError` through
  the cross-validator; keep a CLI-override case (`training.n_epochs=...`).
- **`test_config.py`** — unchanged (base `Config.from_dictconfig`).

## Migration / compatibility

- The `0001` run-dir layout, wandb backend, and verdict logic are unchanged. Only
  the config surface and the train loop's budget unit change.
- Gates stay as upper bounds; with epoch training they remain valid (more training
  ⇒ lower loss / energy distance). Recalibrate only if a real run shows otherwise.
- `experiments/new.py` and `experiments/README.md` describe a *flat* `Config`
  today; this design does not require the scaffolder to emit nested groups —
  0001 is the reference for the multi-group pattern. (A scaffolder update can be a
  separate follow-up; flag it but keep it out of this scope to limit churn.)

## Out of scope

- Resume-from-checkpoint (still just save).
- Physics-residual scalars / loss terms (future work).
- `new.py` scaffolder emitting the multi-group + discriminated-union shape
  (0001 stands as the worked example; revisit when a second framework needs it).
- A `time_dim` knob on the MLP config (kept at the `VelocityMLP` default — YAGNI).
```
