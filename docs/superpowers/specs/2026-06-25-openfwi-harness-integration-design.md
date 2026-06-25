# OpenFWI Harness Integration (+ remove pass/fail gates) — Design

**Date:** 2026-06-25
**Status:** Approved for planning

## Goal

Two coupled changes to the `0001_flow_matching` framework:

- **Part A — remove pass/fail gates.** Experiments are research runs, not tests; a
  binary pass/fail an experiment can't really "fail" adds noise. Replace the
  gate/verdict logic with plain metric logging.
- **Part B — integrate OpenFWI.** Add an `openfwi` dataset to the discriminated
  dataset union so OpenFWI velocity maps train an unconditional flow-matching prior
  through the existing wandb/Hydra/epoch/EMA harness, exactly like `mnist`.

Part A lands first (it unblocks a gateless OpenFWI run and is a small shared-harness
cleanup).

## Motivation

OpenFWI velocity maps are the natural physically-meaningful image domain for this
repo's research goal: a flow-matching prior `p(v)` over subsurface velocity models,
which the later physics-informed phase will steer into posterior sampling `p(v|d)`
via PDE-residual losses. This spec delivers the *unconditional prior* milestone —
the OpenFWI analogue of the existing mnist run. Held-out validation and a real eval
metric are deliberately a **separate** follow-up plan.

The dataset is already staged on disk at `data/openfwi/` (gitignored): 10 families,
~470k velocity maps, each `(500, 1, 70, 70)` float32, velocity `1500–4500 m/s`, in two
on-disk layouts — Vel/Style families use `<Family>/model/*.npy`, Fault families use
flat `<Family>/vel*.npy`. This integration trains on **FlatVel_A only** by default,
with `families` configurable.

## Part A — Remove pass/fail gates

### Harness — `experiment/run.py`

`Run.finish` drops the `verdict` parameter and just records + prints summary scalars:

```python
def finish(self, **summary: Any) -> None:
    """Record summary scalars to the wandb run summary and close the run."""
    for key, value in summary.items():
        self.run.summary[key] = value
    extra = " ".join(f"{key}={value}" for key, value in summary.items())
    print(f"[{self.experiment}] {extra}".rstrip())
    self.run.finish()
```

Update the module docstring (line ~6) so it no longer says `finish` "records the
verdict". No `"verdict"` summary key is written.

### Experiment — `experiments/0001_flow_matching/run.py`

- Remove the `gate: float = 0.5` field from `FlowMatchingConfig`.
- Remove the `verdict = "pass"/"fail"` computation. The final block becomes:

  ```python
  if isinstance(cfg.dataset, GaussiansDatasetConfig):
      ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
      metric = energy_distance(samples, ref)
      run.finish(energy_distance=metric, final_loss=final_loss)
  else:
      run.finish(final_loss=final_loss)
  ```

- Update the module docstring (line ~8) to drop the "Verdict: … < gate" sentence;
  describe it as logging energy distance (gaussians) / final FM loss (image datasets).
- OpenFWI needs no special-casing in `main()`: it is not a `GaussiansDatasetConfig`,
  so it takes the `else` branch (logs `final_loss`); sampling and visualization
  already dispatch through `cfg.dataset.shape` / `cfg.dataset.visualize`.

### Config — `conf/`

Remove the `gate:` line from `config.yaml`, `experiment/gaussians.yaml`,
`experiment/mnist.yaml`, and `experiment/smoke.yaml`.

### Tests & docs

- `tests/test_run.py`: change `run.finish("pass", final_loss=0.5)` to
  `run.finish(final_loss=0.5)`.
- `experiments/JOURNAL.md`: change the format guidance from
  `NNNN_slug/variant — verdict: headline (evidence)` to
  `NNNN_slug/variant — headline metric (evidence)`. Existing historical lines may
  stay as-is (they record past findings).
- The closed `report.md` for 0001 is a historical artifact; leave it unchanged.

## Part B — OpenFWI dataset integration

### New module — `flow_matching/openfwi.py`

Isolates the heavier loading logic from the config-focused `datasets.py`.

Constants: `VMIN = 1500.0`, `VMAX = 4500.0`, `NATIVE = 70`.

```python
class OpenFWIVelocityDataset(Dataset):
    """Lazy, memory-mapped OpenFWI velocity maps across one or more families."""

    def __init__(self, root: Path, families: list[str], resolution: int = 64) -> None:
        self.resolution = resolution
        self.index: list[tuple[Path, int]] = []
        for fam in families:
            fam_dir = root / fam
            files = sorted(fam_dir.glob("model/*.npy")) + sorted(fam_dir.glob("vel*.npy"))
            if not files:
                raise FileNotFoundError(
                    f"No OpenFWI velocity files under {fam_dir} "
                    f"(expected <family>/model/*.npy or <family>/vel*.npy). "
                    f"Download with huggingface_hub from 'ashynf/OpenFWI'."
                )
            for f in files:
                n = int(np.load(f, mmap_mode="r").shape[0])  # 500 per file
                self.index.extend((f, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        path, row = self.index[idx]
        arr = np.load(path, mmap_mode="r")[row]          # (1, 70, 70) float32
        x = torch.from_numpy(np.ascontiguousarray(arr)).float()
        x = ((x - VMIN) / (VMAX - VMIN) * 2.0 - 1.0).clamp(-1.0, 1.0)
        if self.resolution != NATIVE:
            x = F.interpolate(
                x[None], size=self.resolution, mode="bilinear",
                align_corners=False, antialias=True,
            )[0]
        return x, 0
```

Notes:
- Globbing **both** `model/*.npy` and `vel*.npy` makes it layout-agnostic across all
  families. For FlatVel_A only the `model/*.npy` branch matches.
- The index stores `(file, row)` pairs and uses `mmap_mode="r"` so nothing is held in
  RAM beyond the small index; per-access resize of a 70×70 map is microseconds.
- Returns `(x, 0)` to satisfy the harness's `(x, label)` contract (unconditional).

### Visualization — `flow_matching/openfwi.py`

```python
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

### Config — `flow_matching/datasets.py`

Add the variant (delegating to the new module) and register it in the union:

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
        return OpenFWIVelocityDataset(Path(self.data_dir), self.families, self.resolution)

    def visualize(self, samples: Tensor, path: Path) -> None:
        viz_velocity(samples, path)
```

```python
DatasetConfig = Annotated[
    GaussiansDatasetConfig | MNISTDatasetConfig | OpenFWIDatasetConfig,
    Field(discriminator="name"),
]
DATASETS["openfwi"] = OpenFWIDatasetConfig()
```

The existing `FlowMatchingConfig` cross-validator already routes a 3-D square shape
`(1, 64, 64)` to the DiT model and rejects MLP — no validator change needed.

### Hydra config — `conf/`

`conf/dataset/openfwi.yaml`:
```yaml
name: openfwi
data_dir: data/openfwi
families: [FlatVel_A]
resolution: 64
```

`conf/experiment/openfwi.yaml`:
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

`n_eval_samples: 64` (vs the 2048 default) because OpenFWI takes the `else` branch and
only the visualized grid (≤64) uses the final samples — no need to generate thousands of
64×64 maps. The DiT backbone is reused as-is (`conf/model/dit.yaml`: hidden 256 / depth 6
/ heads 8 / patch_size 4 → 16×16 = 256 tokens at 64×64). Per-variant model tuning for
*good* samples is a runtime concern, not part of this spec.

## Testing

Hermetic — no dependency on the 8.8 GB real download. New `tests/test_openfwi.py` writes
tiny synthetic fixtures and asserts behavior:

- A `_write_family(tmp_path, family, layout, n_files=2, rows=3)` helper saves
  `(rows, 1, 70, 70)` float32 arrays (values spanning ~1500–4500) as `.npy`, either
  under `<family>/model/model{i}.npy` or flat `<family>/vel{i}.npy`.
- `len(build())` equals `n_files * rows` for the selected families.
- A sample is shape `(1, 64, 64)`, dtype float32, with values in `[-1, 1]`.
- Normalization is correct: a fixture pixel set to `VMIN` maps to `-1`, `VMAX` to `+1`
  (within tolerance), and out-of-range values clamp.
- Both layouts load (`model/*.npy` and flat `vel*.npy`).
- `resolution=70` returns native `(1, 70, 70)` (no resize); `resolution=64` resizes.
- `build()` on a missing/empty family dir raises `FileNotFoundError` with the download hint.
- `viz_velocity` writes a PNG file that exists and is non-empty.

`tests/test_experiment_conf.py`:
- Add an `openfwi` case to the compose validation asserting `dataset.name == "openfwi"`,
  `model.kind == "dit"`, `dataset.shape == (1, 64, 64)`, `training.ema.enabled is True`.

Gate-removal test impact:
- `tests/test_run.py` updated to the new `finish(**summary)` signature (above).
- No test constructs `FlowMatchingConfig(gate=...)`, and the compose tests do not assert
  on `gate`, so removing the field + yaml keys leaves them green. (The plan must still
  run the full suite to confirm.)

Run wandb-touching tests under `WANDB_MODE=disabled` (existing fixtures already do).

## Migration / compatibility

- `Run.finish`'s signature changes from `finish(verdict, **summary)` to
  `finish(**summary)`; the only in-repo callers are `run.py` (updated in Part A) and
  `test_run.py`. As in prior refactors, the experiment `run.py` is internally consistent
  within Part A's task and no test executes its `main()`.
- Adding `OpenFWIDatasetConfig` to the union is backward compatible: existing
  `gaussians`/`mnist`/`smoke` composes are unaffected (discriminated by `name`).

## Out of scope (deferred)

- **Held-out validation + real eval metric** — its own follow-up plan (next). This is
  why OpenFWI logs only `final_loss` for now and gaussians keeps its existing
  same-dataset energy distance.
- **Conditional `p(v|d)` / physics-residual losses** — the later physics-informed phase.
- **Multi-family training and per-variant model tuning** for *good* OpenFWI samples —
  `families`/`resolution`/model size are configurable, but tuning is a runtime activity.
- **Auto-download of OpenFWI** — `build()` reads from disk and fails loudly if absent.
