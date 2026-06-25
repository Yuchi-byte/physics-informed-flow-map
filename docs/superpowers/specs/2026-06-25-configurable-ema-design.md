# Configurable EMA of Velocity Weights — Design

**Date:** 2026-06-25
**Status:** Approved for planning

## Goal

Add a configurable exponential-moving-average (EMA) of the velocity-model weights
to `0001_flow_matching`. When enabled, eval images / final samples / the verdict
all sample from the **EMA** weights (the quality benefit), and **both** the raw
last-step weights and the EMA weights are checkpointed locally and uploaded as
wandb artifacts. EMA is on by default for the tuned `mnist` and `gaussians`
variants and off in the base config.

## Motivation

Flow-matching/diffusion gradients are noisy (each batch sees only a few
noise levels/timesteps), so the iterate wobbles near convergence and sampling
chains amplify that into malformed strokes (visible in the current MNIST grid).
EMA averages the weight trajectory into a flatter, more stable point and is the
standard trick behind essentially every strong diffusion/FM result (DDPM, EDM,
mfm itself). mfm's own EMA is a PyTorch-Lightning callback bound to its Trainer,
which our plain training loop does not use — so we implement EMA with the stdlib
`torch.optim.swa_utils.AveragedModel` (no edit to any reference package).

## Architecture

### A. EMA in the training loop — `flow_matching/train.py`

Use `torch.optim.swa_utils.AveragedModel` with the foreach EMA averaging fn:

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
```

`train()` gains EMA params and returns the EMA model alongside the history:

```python
def train(model, loader, *, n_epochs, lr, device, num_classes=None,
          log=None,
          ema_enabled=False, ema_decay=0.999, ema_warmup_steps=0,
          eval_every_epochs=0, on_eval=None,
          ckpt_every_epochs=0, on_checkpoint=None,
          ) -> tuple[list[dict[str, float]], BaseModel | None]:
```

Behaviour:
- After `model = model.to(device)`, create the EMA wrapper iff `ema_enabled`:
  `ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)`.
- Per optimizer step (after `optimizer.step()`), if `ema is not None and step >= ema_warmup_steps`: `ema.update_parameters(model)`.
- "EMA ready" ⇔ `ema is not None and int(ema.n_averaged.item()) > 0` (i.e. at least one
  update has happened). Before that — or when disabled — fall back to the raw model.
- Hooks:
  - `eval_model = ema.module if ema_ready else model`; call `on_eval(eval_model, epoch)`,
    then `model.train()` unconditionally (restores train mode if `eval_model` was the
    raw model and `sample()` flipped it to eval).
  - `on_checkpoint(model, epoch, *, is_best, is_final, ema_model=<ema.module if ema_ready else None>)`.
- Return `(history, ema.module if ema_ready else None)`. With `ema_warmup_steps`
  beyond the run length, EMA never updates → returns `None` (caller falls back to raw).

`ema.module` is a deepcopy of the model with averaged params; its `state_dict()`
keys match the original, so it reloads via `build_model(...)` like any checkpoint,
and it exposes `.v(...)` for `sample()`.

### B. Harness checkpoint suffix — `experiment/run.py`

`Run.save_checkpoint` gains a `suffix` so EMA weights get a distinct file:

```python
def save_checkpoint(self, model, step, *, suffix="", **meta) -> Path:
    path = self.ckpt_dir / f"step_{step}{suffix}.pt"
    torch.save({"model": model.state_dict(), "step": step, **meta}, path)
    return path
```

(`log_artifact`, `log`, `finish`, etc. unchanged.)

### C. Config + run.py wiring — `experiments/0001_flow_matching/run.py`

New nested EMA sub-config and `training.ema`:

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

`main()` changes:
- `on_checkpoint` saves+uploads the raw checkpoint as today, and additionally
  (when `ema_model is not None`) saves `step_N_ema.pt` and uploads it under
  `{dataset}-model-ema` with the same aliases:

  ```python
  def on_checkpoint(m, epoch, *, is_best=False, is_final=False, ema_model=None):
      aliases = [...]              # final / best / periodic, as today
      path = run.save_checkpoint(m, epoch, dataset=cfg.dataset.name, config=cfg.dump())
      if aliases:
          run.log_artifact(path, name=f"{cfg.dataset.name}-model", aliases=aliases)
      if ema_model is not None:
          ep = run.save_checkpoint(ema_model, epoch, suffix="_ema",
                                   dataset=cfg.dataset.name, config=cfg.dump())
          if aliases:
              run.log_artifact(ep, name=f"{cfg.dataset.name}-model-ema", aliases=aliases)
  ```

- Pass EMA config into `train` and use the returned EMA model for the final eval:

  ```python
  history, ema_model = train(
      model, loader, n_epochs=..., lr=..., device=device,
      num_classes=cfg.dataset.num_classes, log=run.log,
      ema_enabled=cfg.training.ema.enabled,
      ema_decay=cfg.training.ema.decay,
      ema_warmup_steps=cfg.training.ema.warmup_steps,
      eval_every_epochs=..., on_eval=on_eval,
      ckpt_every_epochs=..., on_checkpoint=on_checkpoint,
  )
  final_loss = history[-1]["total"]
  eval_model = ema_model if ema_model is not None else model
  samples = sample(eval_model, cfg.sampling.n_eval_samples, cfg.dataset.shape, ...)
  ```

  `final_loss` stays the raw training loss (the mnist gate is loss-based and EMA
  doesn't change the training loss); the verdict's *sampled* metric (gaussians
  energy distance) and `samples.png` come from `eval_model` (EMA when enabled).
  The periodic `on_eval` already samples from whatever model `train` hands it.

### D. Hydra config — `experiments/0001_flow_matching/conf/`

`conf/training/default.yaml` gains the EMA block (off):

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

The tuned variants enable it (Hydra deep-merges the partial block):

```yaml
# conf/experiment/mnist.yaml  (training block)
training:
  n_epochs: 100
  batch_size: 128
  ema:
    enabled: true
```

```yaml
# conf/experiment/gaussians.yaml  (training block)
training:
  n_epochs: 100
  ema:
    enabled: true
```

CLI: `run.py experiment=mnist training.ema.decay=0.9999`,
or `run.py experiment=mnist training.ema.enabled=false` to disable.

## Testing

Run under `WANDB_MODE=disabled` where wandb is touched.

- **`test_train.py`** — update every `train(...)` call to unpack `history, _ =
  train(...)` (or `history, ema = ...`). Add:
  - EMA disabled (default) → second return is `None`.
  - EMA enabled, a few epochs → second return is an `nn.Module` (not the same object
    as `model`), and its parameters differ from the raw model's after training
    (averaging actually happened).
  - `ema_warmup_steps` greater than the run's total steps → second return is `None`
    (never warmed up).
- **`test_run.py`** — `save_checkpoint(model, step, suffix="_ema")` writes
  `step_<step>_ema.pt` and the file exists with the expected keys.
- **`test_experiment_conf.py`** — assert `cfg.training.ema.enabled is True` for the
  `mnist` and `gaussians` variants and `False` for a base/`smoke` compose; keep the
  existing variant/CLI-override coverage. (The `extra="forbid"` nesting must accept
  the `ema` block.)
- Existing `test_models.py`/`test_datasets.py`/`test_sample.py`/`test_config.py`
  unaffected.

## Migration / compatibility

- `train()`'s return type changes from `list[...]` to `tuple[list[...], BaseModel |
  None]` — all in-repo callers (tests + run.py) are updated in the same plan. As in
  prior refactors, `run.py`'s `main()` is transiently inconsistent between the
  train-change task and the run.py-integration task, but no test executes `main()`,
  so the suite stays green throughout.
- Gates unchanged. EMA defaults to off in base; only the tuned variants enable it.

## Out of scope

- Resume-from-checkpoint (still save-only; EMA checkpoints are saved/uploaded but no
  `--resume` path).
- EMA for the harness/scaffolder beyond 0001 (0001 is the worked example).
- Bias-correction of the EMA average (warmup-seeded averaging is sufficient at our
  run lengths; the stdlib `AveragedModel` semantics are kept as-is).
