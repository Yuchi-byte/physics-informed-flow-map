# wandb Experiment Tracking + Hydra Configs — Design

**Date:** 2026-06-24
**Status:** Approved for planning

## Goal

Two coupled changes to the experiment harness:

1. **Replace** the local-file tracking (`manifest.json` / `metrics.jsonl` /
   `result.json`) with **Weights & Biases** as the single tracking backend, and
   add three capabilities the reference packages (MFM, PBFM, PIDM) have and we
   lack: **decomposed loss logging**, **periodic eval + sample images during
   training**, and **checkpointing** (local + wandb artifacts).
2. **Adopt Hydra** for configuration harness-wide: each framework's `run.py`
   becomes a `@hydra.main` entry point composing yaml config groups, which we
   then validate into the existing pydantic `Config` (keeping `extra="forbid"`).

Resume-from-checkpoint is explicitly **out of scope** — we *save* checkpoints but
add no `--resume` path in this work.

## Motivation

Both physics reference packages treat the physics residual/constraint magnitude
as a first-class logged metric, separate from the data loss. Our loop logs only
the FM `total`. When PIDM-style residual terms land later, per-term logging must
already exist or we are blind to whether the constraint is satisfied. This design
makes loss logging decomposed-by-default and gives every run live curves, mid-run
sample images, and saved model weights. Hydra brings composable yaml config
groups and `key=value` CLI overrides (matching the mfm reference impl) while the
pydantic layer keeps strict typed validation.

## Key finding: dataset/model interfaces already suffice

No changes are required to `DatasetSpec` or the model layer:

- **Periodic eval images** ← `DatasetSpec.visualize(samples, path)` already writes
  a figure we can hand to `wandb.Image`.
- **Checkpointing** ← models are `mfm` `BaseModel` (an `nn.Module`); `state_dict()`
  is all we need to save, and `build_model(shape, num_classes, **knobs)` rebuilds
  any architecture from config for a future resume.

The work lives in the **harness** (`experiment/`), the **training loop**
(`flow_matching/train.py`), the experiment **entry points** + their new `conf/`
trees, and the experiments **contract** (`README.md`, `new.py` scaffolder).

## Architecture

### A. Config & CLI: Hydra → pydantic

**Per-framework layout** (`new.py` scaffolds this; `0001_flow_matching` is the
first to adopt it):

```
experiments/0001_flow_matching/
  conf/
    config.yaml              # base defaults (all Config fields) + hydra settings
    experiment/
      gaussians.yaml         # # @package _global_  — field overrides
      mnist.yaml
      smoke.yaml
  run.py                     # @hydra.main entry point
```

- `conf/config.yaml` holds every `FlowMatchingConfig` field at its default, a
  `defaults:` list selecting the default variant, and the hydra block:

  ```yaml
  defaults:
    - _self_
    - experiment: gaussians
  hydra:
    run:
      dir: runs/0001_flow_matching/${now:%Y-%m-%dT%H-%M-%SZ}
    job:
      chdir: false           # cwd stays at repo root
  ```

- Variant files start with `# @package _global_` so their keys override base
  fields at the top level (the standard Hydra "experiment" pattern), e.g.
  `conf/experiment/mnist.yaml`:

  ```yaml
  # @package _global_
  dataset: mnist
  n_steps: 3000
  batch_size: 128
  sampler_steps: 50
  gate: 240.0
  ```

- The config stays **flat** (no nested model/dataset groups) — `dataset` is a
  string key, mirroring today's `FlowMatchingConfig`. So `conf/` needs only
  `config.yaml` + `experiment/*.yaml`.

- `run.py` composes then validates:

  ```python
  @hydra.main(version_base=None, config_path="conf", config_name="config")
  def main(dcfg: DictConfig) -> None:
      cfg = FlowMatchingConfig.from_dictconfig(dcfg)   # strict; typo -> error
      run_dir = Path(HydraConfig.get().runtime.output_dir)
      run = start_run("0001_flow_matching", run_dir, cfg.dump())
      ...
  ```

- **CLI:** `uv run python experiments/0001_flow_matching/run.py experiment=mnist eval_every=500 ckpt_every=1000`.
  Variant selection is `experiment=<name>`; field overrides are `key=value`.

**Harness `Config` base** (`experiment/config.py`): keep the pydantic schema and
`dump()`; **replace** `resolve(variant, overrides)` (the old OmegaConf merge, now
Hydra's job) with:

```python
@classmethod
def from_dictconfig(cls, cfg: DictConfig) -> "Config":
    container = OmegaConf.to_container(cfg, resolve=True)
    return cls.model_validate(container)        # extra="forbid" still applies
```

### B. Harness: `experiment/run.py` (wandb wrapper)

`start_run(experiment, run_dir, config, *, project="physics-informed-flow-map", name=None) -> Run`

- `experiment` (str) and `run_dir` (Path, from Hydra's `runtime.output_dir`) are
  passed in by `run.py`, so `start_run` stays Hydra-free and unit-testable.
- Builds the same environment dict the old manifest held (git commit, python /
  torch / cuda versions, GPU name) and calls
  `wandb.init(project=…, name=name, group=experiment, dir=str(run_dir), config={**config, **env})`.
  wandb additionally captures the git SHA + a `diff.patch` natively, so
  reproducibility is preserved/strengthened — nothing is lost by dropping
  `manifest.json`.
- Creates `run_dir/checkpoints/` (binaries do not belong in wandb config); exposes
  it as `Run.ckpt_dir`. Hydra's own `.hydra/config.yaml` dump also lands in
  `run_dir` as a free config snapshot.
- Connectivity is wandb-native via `WANDB_MODE` (default online); `project`
  overridable via the arg or `WANDB_PROJECT`. Tests force `WANDB_MODE=disabled`.

`Run` API:

| method | behaviour |
|---|---|
| `log(**metrics)` | `wandb.log(metrics, step=metrics.pop("step", None))` |
| `log_image(key, path, *, step=None)` | `wandb.log({key: wandb.Image(str(path))}, step=step)` |
| `save_checkpoint(model, step, **meta) -> Path` | `torch.save({"model": state_dict, "step": step, **meta}, ckpt_dir/f"step_{step}.pt")`; returns the path |
| `log_artifact(path, *, name, aliases)` | `wandb.Artifact(name, type="model")` + add file + `wandb.run.log_artifact(art, aliases=aliases)` |
| `finish(verdict, **summary)` | set `wandb.run.summary["verdict"]=verdict` and each summary scalar; **print** `[<exp>] verdict=<v>`; `wandb.finish()` |

The console print preserves the at-a-glance pass/fail that `result.json` gave,
without opening the dashboard.

### C. Training loop: `flow_matching/train.py`

`train()` gains three optional, default-off hooks:

```python
def train(model, loader, *, n_steps, lr, device, num_classes=None,
          log=None,
          eval_every=0,  on_eval=None,        # on_eval(model, step) -> float | None
          ckpt_every=0,  on_checkpoint=None):  # on_checkpoint(model, step, *, is_best, is_final)
```

- **(1) Decomposed losses.** Per step, `log` emits *every* key in mfm's
  `opt_losses` dict (currently just `fm_loss`) plus `total`, instead of the
  hand-picked subset. New residual terms flow through automatically.
- **(2) Periodic eval.** When `eval_every > 0` and `step % eval_every == 0`, call
  `metric = on_eval(model, step)`. `on_eval` logs a sample image and returns a
  scalar **monitored metric (lower is better)** or `None`. The loop tracks the
  running best; a new low sets `is_best=True` for the next checkpoint call.
- **(3) Checkpointing.** Call `on_checkpoint(model, step, is_best=…, is_final=…)`
  when `ckpt_every > 0 and step % ckpt_every == 0`, **or** when a new best metric
  was just seen. After the loop, always call it once with `is_final=True`.

`train` stays I/O-free and unit-testable: it only invokes callbacks. All wandb /
disk wiring is supplied by the caller.

### D. Entry point: `experiments/0001_flow_matching/run.py`

New `FlowMatchingConfig` knobs (defaults are the opt-in-off value):

| knob | default | meaning |
|---|---|---|
| `eval_every` | `0` | steps between periodic eval+image (0 = off) |
| `ckpt_every` | `0` | steps between local checkpoints (0 = off) |
| `artifact_every` | `0` | steps between wandb artifact uploads (0 = off; coarser than `ckpt_every`) |
| `n_eval_viz` | `64` | sample count for the periodic eval image |

Closures wired in `main()`:

```python
def on_eval(m, step):
    s = sample(m, cfg.n_eval_viz, spec.shape, sampler_steps=cfg.sampler_steps, device=device)
    p = run.ckpt_dir.parent / f"samples_{step}.png"
    spec.visualize(s, p)
    run.log_image("samples", p, step=step)
    return monitored_metric(m)        # gaussians: energy distance; else None

def on_checkpoint(m, step, *, is_best=False, is_final=False):
    path = run.save_checkpoint(m, step, dataset=cfg.dataset, config=cfg.dump())
    aliases = []
    if is_final: aliases.append("final")
    if is_best:  aliases.append("best")
    if cfg.artifact_every and step % cfg.artifact_every == 0:
        aliases.append("periodic")
    if aliases:
        run.log_artifact(path, name=f"{cfg.dataset}-model", aliases=aliases)
```

### Artifact / checkpoint cadence summary

| trigger | local checkpoint | wandb artifact (alias) | default |
|---|---|---|---|
| every step | — | — | scalar log only |
| `eval_every` | — | — (image + metric) | off |
| `ckpt_every` | yes | — | off |
| `artifact_every` | yes | `periodic` | off |
| new best metric | yes | `best` | active only when `eval_every>0` |
| end of run (`is_final`) | yes (`final`-tagged) | `final` | **always** |

`final` (local + artifact) always fires so every run leaves a reusable model;
`best` only materialises when periodic eval is enabled (it needs a metric stream).
Variant yaml files do **not** preset the cadences — the user opts in per run.

The existing post-train verdict eval in `run.py` (energy distance / final loss)
is unchanged; its final `samples.png` is now also pushed via `run.log_image`.

### E. Scope: harness-wide

- **`experiments/README.md`** — rewrite "Anatomy of a framework", "Where results
  land", and "Running": config via Hydra (`conf/` + `@hydra.main`), variants =
  config groups selected with `experiment=<name>`, overrides as `key=value`,
  output dir = `runs/<framework>/<stamp>/` via `hydra.run.dir`, tracking = wandb
  (config / metrics / images / verdict) + local `checkpoints/`. Drop the
  `manifest.json` / `metrics.jsonl` / `result.json` block.
- **`experiments/new.py`** — scaffold the `conf/` tree (`config.yaml` with the
  hydra block + a `default`/`smoke` experiment group) and a `@hydra.main` `run.py`
  template instead of the `VARIANTS` + `sys.argv` template.
- **`CLAUDE.md`** — refresh the experiment run examples to the Hydra syntax
  (`run.py experiment=smoke`, `run.py experiment=mnist n_steps=500`); the current
  examples are already stale (they cite a removed `0001_mnist_pipeline`).

### F. Dependencies / tooling

- Add `wandb` and `hydra-core` to `packages/physics-informed-flow-map/pyproject.toml`
  `[project]` dependencies; add `wandb.*` and `hydra.*` to the mypy
  `ignore_missing_imports` override list.

## Testing

All tests run under `WANDB_MODE=disabled` (autouse env fixture), so no network or
wandb files are touched.

- **Config validation** (`tests/test_config.py`, new): `Config.from_dictconfig`
  on an `OmegaConf.create({...})` returns a validated instance; an unknown key
  raises `ValidationError` (proves `extra="forbid"` survives the Hydra→pydantic
  hop). No Hydra runtime needed.
- **Hydra compose** (`tests/test_experiment_conf.py`, new): use
  `hydra.initialize` + `compose(config_name="config", overrides=["experiment=mnist"])`
  for 0001 and assert the composed cfg validates into `FlowMatchingConfig` with
  the expected variant values. Guards the yaml config groups end-to-end.
- **Harness** (`tests/test_run.py`, new): `start_run(exp, tmp_path, cfg)` →
  `log` → `log_image` → `save_checkpoint` → `log_artifact` → `finish(verdict)`
  runs without error; the checkpoint file exists with the expected keys. Hydra-free
  (run_dir passed explicitly).
- **Training hooks** (`tests/test_train.py`, extend): with a tiny MLP, assert
  `on_eval` fires `floor(n_steps/eval_every)` times, `on_checkpoint` fires on the
  `ckpt_every` cadence plus exactly one final `is_final=True` call, and a
  monotonically-decreasing `on_eval` return marks `is_best=True` appropriately.
- Existing `test_datasets.py` / `test_models.py` / `test_sample.py` are unaffected.

## Out of scope

- Resume-from-checkpoint (`--resume`) — checkpoints are saved but not reloaded.
- Physics-specific scalars (residual mean/median, constraint-violation rate) —
  deferred until a physics-residual term exists, so the metric schema can be
  designed around it.
- EMA weights and reward-model evaluation (MFM-specific, overkill at this stage).
- Nested Hydra config groups (per-model / per-dataset yaml) — the config is flat
  today; revisit if a framework needs swappable sub-configs.
```
