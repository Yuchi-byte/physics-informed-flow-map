# Held-out Validation Loss — Design

**Date:** 2026-06-25
**Status:** Approved for planning

## Goal

Give `0001_flow_matching` a held-out validation signal. Each dataset gains a
disjoint validation split; each evaluation computes the mean **held-out FM loss**
(`val_loss`) over that split, logs it, and uses it to select the **best**
checkpoint — uniformly across `gaussians`, `mnist`, and `openfwi`.

## Motivation

Today the framework holds out nothing: the `mnist`/`openfwi` runs have no eval
signal beyond training loss, and `gaussians` references reals drawn from its own
training set. For flow-matching/diffusion the standard held-out signals are
sample-quality metrics against held-out reals and, more cheaply, a held-out loss as
an overfitting tripwire. We add the held-out **val FM loss** (a genuine
overfitting/early-quality signal that costs one forward pass per eval) and make it
the **best**-checkpoint criterion, which currently never fires for the image
datasets.

Deliberately **not** in this feature: physics residuals (FWI residuals need the
seismic data `d` and the wave-equation forward operator, neither of which we have —
they belong to the later conditional `p(v|d)` phase); FID (its ImageNet Inception
backbone is meaningless for velocity maps); and energy-distance-vs-held-out (the user
scoped this to val-loss only). The existing `gaussians` energy distance stays as a
final-summary readout.

## Architecture

### A. Reusable FM loss — `flow_matching/train.py`

Extract the loss construction currently inline in `train()` into a module-level
factory so `run.py` can compute the validation loss with the *same* loss:

```python
def make_loss_fn(num_classes: int | None) -> Callable[..., Any]:
    """The pure-FM consistency loss used for both training and validation."""
    return get_consistency_loss_fn(_fm_loss_cfg(num_classes or 0), Linear(t_max=1.0))
```

`train()` replaces its inline `loss_fn = get_consistency_loss_fn(...)` with
`loss_fn = make_loss_fn(num_classes)` — no behavior change.

### B. Log val loss in the eval block — `flow_matching/train.py`

`train()`'s eval block already calls `metric = on_eval(eval_model, epoch)` and uses
`metric` for best-tracking. The only change: when `metric is not None`, also log it
as `val_loss` at the current step. Best-tracking is unchanged (it already keys off
the returned metric), so the returned val loss now drives **best** uniformly.

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

(The framework's `on_eval` metric is the held-out FM loss by construction, so naming
the logged key `val_loss` is accurate. `train()`'s signature and return type are
otherwise unchanged.)

### C. Held-out splits — `flow_matching/datasets.py`

Add `build_val(self) -> Dataset` to each of the three configs. `build()` keeps
returning the **training** split; `build_val()` returns the disjoint held-out set.

**Gaussians** — a fresh i.i.d. draw with a distinct seed (the distribution is
infinite/synthetic, so an independent sample *is* the held-out set). Add a
`val_samples: int = 10000` field. The existing module helper `_make_gaussians`
already takes a `seed`; `build()` keeps the default `seed=0`, `build_val()` uses a
fixed distinct seed:

```python
    def build(self) -> Dataset:
        return _make_gaussians(self.n_samples, self.n_modes, self.radius, self.std)

    def build_val(self) -> Dataset:
        return _make_gaussians(
            self.val_samples, self.n_modes, self.radius, self.std, seed=1
        )
```

**MNIST** — the canonical torchvision test set. Give `_make_mnist` a `train` flag:

```python
def _make_mnist(data_dir: str = "data", image_size: int = 32, train: bool = True) -> Dataset:
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
```python
    def build(self) -> Dataset:
        return _make_mnist(self.data_dir, self.image_size, train=True)

    def build_val(self) -> Dataset:
        return _make_mnist(self.data_dir, self.image_size, train=False)
```

**OpenFWI** — a deterministic seeded split of the velocity maps. Add
`val_fraction: float = 0.1`. A private helper builds the full dataset once and
returns the index partition; `build()` wraps the train indices, `build_val()` the
val indices, in `torch.utils.data.Subset`:

```python
    def _split(self) -> tuple[Dataset, list[int], list[int]]:
        full = OpenFWIVelocityDataset(Path(self.data_dir), self.families, self.resolution)
        n = len(full)
        n_val = max(1, int(self.val_fraction * n))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
        return full, perm[n_val:], perm[:n_val]

    def build(self) -> Dataset:
        full, train_idx, _ = self._split()
        return Subset(full, train_idx)

    def build_val(self) -> Dataset:
        full, _, val_idx = self._split()
        return Subset(full, val_idx)
```

(Building `full` once per call is cheap — the index is just globbing plus per-file
mmap shape reads. The split is deterministic, so the two calls partition the same
permutation.)

`requires_download` is unchanged per dataset; `build_val()` shares it (gaussians
False, mnist/openfwi True).

### D. Compute + return val loss — `experiments/0001_flow_matching/run.py`

`main()` builds the val loader and the val loss once, and a `compute_val_loss`
helper averages the FM loss over the held-out split (eval mode, no grad):

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

`on_eval` keeps sampling/visualizing, and now **returns the val loss** (dropping the
per-epoch gaussians energy-distance branch):

```python
    def on_eval(m: BaseModel, epoch: int) -> float | None:
        s = sample(
            m, cfg.sampling.n_eval_viz, cfg.dataset.shape,
            sampler_steps=cfg.sampling.sampler_steps, device=device,
        )
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p)
        return compute_val_loss(m)
```

The final summary computes a final `val_loss` (from `eval_model`) and includes it;
the `gaussians` energy distance stays in the final summary as before:

```python
    final_val_loss = compute_val_loss(eval_model)
    ...
    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        run.finish(energy_distance=metric, final_loss=final_loss, val_loss=final_val_loss)
    else:
        run.finish(final_loss=final_loss, val_loss=final_val_loss)
```

`real_reference`/`energy_distance` imports stay (still used in the final block).

## Testing

Run wandb-touching tests under `WANDB_MODE=disabled`. Keep tests hermetic — no real
MNIST/OpenFWI download.

- **`test_train.py`**
  - `make_loss_fn(num_classes)` returns a callable that, given a model and a gaussian
    batch, produces a finite scalar `sum(opt_losses.values())` (smoke that the
    extracted factory works for both `num_classes=None` and an int).
  - `train()` logs a `val_loss` record when `on_eval` returns a metric: pass a small
    gaussian loader, `eval_every_epochs=1`, an `on_eval` returning a constant (e.g.
    `0.5`), and a `log` capturing records; assert at least one captured record has
    `val_loss == 0.5`. The existing `test_train_hooks_fire_on_epoch_cadence` (which
    passes no `log`) continues to pass unchanged — best-tracking still keys off the
    returned metric.
- **`test_datasets.py`** — a registry-driven `test_build_val_shapes` mirroring
  `test_build_shapes`: skip `requires_download` datasets, else assert `build_val()[0]`
  has shape `cfg.shape` and a valid label. (Covers `gaussians`; `mnist`/`openfwi`
  skip.)
- **`test_openfwi.py`** — using the existing synthetic-fixture helper, an
  `OpenFWIDatasetConfig(data_dir=tmp, families=[...], val_fraction=0.25)` test:
  `len(build()) + len(build_val()) == total`, the two index sets are disjoint, and
  `len(build_val()) == max(1, int(0.25 * total))`.
- **`test_datasets.py`** (or `test_openfwi.py`) — `GaussiansDatasetConfig().build_val()`
  yields `val_samples` items and its first sample differs from `build()[0]` (distinct
  seed → independent draw).

The `run.py` `compute_val_loss`/`on_eval` glue is thin and exercised indirectly: the
`make_loss_fn` test covers the loss, and the compose tests import `run.py`. No test
executes `main()`.

## Migration / compatibility

- `make_loss_fn` is additive; `train()`'s public signature and return type are
  unchanged. The only `train()` behavior change is an extra `val_loss` log record on
  eval epochs and that the returned metric now also gets logged.
- `build_val()` is additive on the dataset interface. `build()`'s semantics are
  unchanged for `gaussians`/`mnist`; for `openfwi`, `build()` now returns the ~90%
  train split instead of the full set (correct for held-out eval).
- The per-epoch `gaussians` energy distance (previously returned by `on_eval` and
  used only for best-tracking, never logged per epoch) is replaced by `val_loss` as
  the best signal; the final-summary energy distance is retained.

## Out of scope (deferred)

- Physics-residual validation (needs seismic `d` + the wave-equation forward
  operator — the conditional `p(v|d)` phase).
- FID / Inception-based metrics.
- Energy-distance / MMD against the held-out split (scoped out: val-loss only).
- Early stopping (val loss is logged and drives best-checkpoint selection, but
  training still runs the full `n_epochs`).
