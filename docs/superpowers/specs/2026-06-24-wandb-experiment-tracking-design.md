# wandb Experiment Tracking — Design

**Date:** 2026-06-24
**Status:** Approved for planning

## Goal

Replace the harness's local-file tracking (`manifest.json` / `metrics.jsonl` /
`result.json`) with **Weights & Biases** as the single experiment-tracking
backend, and add three capabilities the reference packages (MFM, PBFM, PIDM) all
have and we lack: **decomposed loss logging**, **periodic eval + sample images
during training**, and **checkpointing** (local + wandb artifacts).

Resume-from-checkpoint is explicitly **out of scope** — we *save* checkpoints but
add no `--resume` path in this work.

## Motivation

Both physics reference packages treat the physics residual/constraint magnitude
as a first-class logged metric, separate from the data loss. Our loop logs only
the FM `total`. When PIDM-style residual terms land later, per-term logging must
already exist or we are blind to whether the constraint is satisfied. This design
makes loss logging decomposed-by-default and gives every run live curves, mid-run
sample images, and saved model weights.

## Key finding: dataset/model interfaces already suffice

No changes are required to `DatasetSpec` or the model layer:

- **Periodic eval images** ← `DatasetSpec.visualize(samples, path)` already writes
  a figure we can hand to `wandb.Image`.
- **Checkpointing** ← models are `mfm` `BaseModel` (an `nn.Module`); `state_dict()`
  is all we need to save, and `build_model(shape, num_classes, **knobs)` rebuilds
  any architecture from config for a future resume.

All the work lives in the **harness** (`experiment/run.py`), the **training loop**
(`flow_matching/train.py`), and the experiment **entry point**
(`experiments/0001_flow_matching/run.py`).

## Architecture

### A. Harness: `experiment/run.py` (wandb wrapper)

`start_run(experiment_dir, config, *, project="physics-informed-flow-map", name=None) -> Run`

- Builds the same environment dict the old manifest held (git commit, python /
  torch / cuda versions, GPU name) and passes it into
  `wandb.init(project=…, name=…, config={**config, **env})`. wandb additionally
  captures the git SHA and a `diff.patch` natively, so reproducibility is
  preserved/strengthened — nothing is lost by dropping `manifest.json`.
- Creates `runs/<experiment>/<UTC-stamp>/checkpoints/` locally (binaries do not
  belong in wandb config); exposes it as `Run.ckpt_dir`.
- Connectivity is wandb-native via the `WANDB_MODE` env var (default online).
  `project` is overridable via the arg or `WANDB_PROJECT`. Tests force
  `WANDB_MODE=disabled` (no network, no files written).

`Run` API:

| method | behaviour |
|---|---|
| `log(**metrics)` | `wandb.log(metrics, step=metrics.pop("step", None))` |
| `log_image(key, path, *, step=None)` | `wandb.log({key: wandb.Image(str(path))}, step=step)` |
| `save_checkpoint(model, step, **meta) -> Path` | `torch.save({"model": state_dict, "step": step, **meta}, ckpt_dir/f"step_{step}.pt")`; returns the path |
| `log_artifact(path, *, name, aliases)` | wraps `wandb.Artifact(name, type="model")`, adds the file, `wandb.run.log_artifact(art, aliases=aliases)` |
| `finish(verdict, **summary)` | set `wandb.run.summary["verdict"]=verdict` and each summary scalar; **print** `[<exp>] verdict=<v>` to console; `wandb.finish()` |

The console print preserves the at-a-glance pass/fail that `result.json` used to
give without opening the dashboard.

### B. Training loop: `flow_matching/train.py`

`train()` gains three optional, default-off hooks:

```python
def train(model, loader, *, n_steps, lr, device, num_classes=None,
          log=None,
          eval_every=0,  on_eval=None,        # on_eval(model, step) -> float | None
          ckpt_every=0,  on_checkpoint=None):  # on_checkpoint(model, step, *, is_best, is_final)
```

- **(1) Decomposed losses.** Per step, `log` emits *every* key in mfm's
  `opt_losses` dict (currently just `fm_loss`) plus `total`, instead of the
  hand-picked `fm_loss`/`total`. New residual terms flow through automatically.
- **(2) Periodic eval.** When `eval_every > 0` and `step % eval_every == 0`,
  call `metric = on_eval(model, step)`. `on_eval` logs a sample image and returns
  a scalar **monitored metric (lower is better)** or `None`. The loop tracks the
  running best; a new low sets `is_best=True` for the next checkpoint call.
- **(3) Checkpointing.** Call `on_checkpoint(model, step, is_best=…, is_final=…)`
  when `ckpt_every > 0 and step % ckpt_every == 0`, **or** when a new best metric
  was just seen. After the loop, always call it once with `is_final=True`.

`train` stays I/O-free and unit-testable: it only invokes callbacks. All wandb /
disk wiring is supplied by the caller.

### C. Entry point: `experiments/0001_flow_matching/run.py`

New `FlowMatchingConfig` knobs (all default to the opt-in-off value):

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
Variants (`gaussians` / `mnist` / `smoke`) do **not** preset the cadences — the
user opts in per run, e.g.
`uv run python experiments/0001_flow_matching/run.py mnist eval_every=500 ckpt_every=1000 artifact_every=4000`.

The existing post-train verdict eval in `run.py` (energy distance / final loss)
is unchanged; its final `samples.png` is now also pushed via `run.log_image`.

### D. Dependencies / tooling / docs

- Add `wandb` to `packages/physics-informed-flow-map/pyproject.toml` `[project]`
  dependencies; add `wandb.*` to the mypy `ignore_missing_imports` override list.
- `experiments/README.md` "Where results land" section: rewrite to describe wandb
  (config/metrics/images/verdict) + local `checkpoints/`; drop the
  `manifest.json` / `metrics.jsonl` / `result.json` block.
- Update the `experiment/__init__.py` / `run.py` module docstrings.

## Testing

All tests run under `WANDB_MODE=disabled` (set via a fixture / autouse env), so
no network or wandb files are touched.

- **Harness** (`tests/test_run.py`, new): `start_run` → `log` → `log_image` →
  `save_checkpoint` → `log_artifact` → `finish(verdict)` runs without error; the
  checkpoint file exists on disk with the expected keys.
- **Training hooks** (`tests/test_train.py`, extend): with a tiny MLP, assert
  `on_eval` fires `floor(n_steps/eval_every)` times, `on_checkpoint` fires on the
  `ckpt_every` cadence plus exactly one final `is_final=True` call, and that a
  monotonically-decreasing `on_eval` return marks `is_best=True` appropriately.
- Existing `test_datasets.py` / `test_models.py` / `test_sample.py` are unaffected
  (interfaces unchanged); `test_train.py`'s existing call keeps working because the
  new params default off.

## Out of scope

- Resume-from-checkpoint (`--resume`) — checkpoints are saved but not reloaded.
- Physics-specific scalars (residual mean/median, constraint-violation rate) —
  deferred until a physics-residual term exists, so the metric schema can be
  designed around it.
- EMA weights and reward-model evaluation (MFM-specific, overkill at this stage).
```
