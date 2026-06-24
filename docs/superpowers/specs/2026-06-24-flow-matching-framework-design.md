# Design: 0001 flow-matching framework (various datasets)

Date: 2026-06-24
Status: approved

## Goal

Replace the current `0001_mnist_pipeline` and `0002_pretrained_sample` experiment
frameworks with a single new `0001_flow_matching` framework whose goal is to
**train flow matching on various datasets** through one generic training loop,
with a dataset abstraction that lets us swap datasets via config. First datasets:
2D toy Gaussians and MNIST.

The flow-matching core is built on the `mfm` package (reuse its interpolant,
loss, and samplers). **The `mfm` library must not be edited** without explicit
prior agreement; all new code lives in our package and the experiment dir.

## Key facts about `mfm` (verified)

- `mfm.losses.get_consistency_loss_fn` and `mfm.SI.ode_sampler_fn` are
  **shape-agnostic** when `learn_loss_weighting=False` — the only 4D-image
  assumption lives in the loss-weighting network, which is then bypassed. So the
  same loss/sampler drive both 2D vectors and images.
- Any model only needs to implement `mfm.models.base_model.BaseModel.v(s, t, x,
  t_cond, x_cond, ...)`. `DiTMFM` already does this for images; for 2D data we
  add a small MLP subclassing `BaseModel` — in our package, not in `mfm`.
- **Pure flow matching** = mfm's loss with `data_fm=True`, `distill_fm=False`,
  `learn_loss_weighting=False`, and `t_cond_0_rate=1.0` (always condition on pure
  noise → standard noise→data flow matching). Sampling integrates the ODE from
  noise with `t_cond=0`.

## Where code lives

Two layers, mirroring catan's lib/experiment split. The FM core is reusable and
is the surface to which physics-residual losses attach later.

```
packages/physics-informed-flow-map/src/physics_informed_flow_map/flow_matching/
├── __init__.py
├── datasets.py     # DatasetSpec + DATASETS registry (gaussians, mnist)
├── models.py       # build_model(spec, cfg); VelocityMLP(BaseModel)
├── train.py        # generic FM training loop (wraps mfm's loss)
└── sample.py       # sampling (mfm ode_sampler_fn) + eval metrics

experiments/0001_flow_matching/
├── run.py          # FlowMatchingConfig + VARIANTS + main (thin wiring)
└── report.md
```

## Component: dataset abstraction (the swap point)

Each dataset is a plain `torch.utils.data.Dataset` returning `(x1, label)`,
paired with a small spec:

```python
@dataclass
class DatasetSpec:
    shape: tuple[int, ...]          # (2,) for gaussians, (1, 32, 32) for mnist
    num_classes: int | None          # None = unconditional
    make_dataset: Callable[..., torch.utils.data.Dataset]
    visualize: Callable[[Tensor, Path], None]   # scatter (2D) / image grid (mnist)

DATASETS: dict[str, DatasetSpec] = {"gaussians": ..., "mnist": ...}
```

- **gaussians**: a synthetic 2D mixture (e.g. 8 Gaussians on a ring). Samples
  shape `(2,)`. Unconditional (`num_classes=None`). `visualize` = scatter plot
  of generated vs. (optionally) true samples.
- **mnist**: torchvision MNIST resized to `32×32`, normalised to `[-1, 1]`.
  Samples shape `(1, 32, 32)`, `num_classes=10`. `visualize` = image grid.

Swapping datasets = changing one config field. Adding a dataset = one registry
entry. The training loop only sees `shape` / `num_classes` + a `DataLoader`.

## Component: model selection by modality

`build_model(spec, cfg)` branches on `len(spec.shape)`:

- **vector** (`(D,)`) → `VelocityMLP`: a small time-conditioned MLP subclassing
  `BaseModel`, implementing `.v(s, t, x, t_cond, x_cond, class_labels=None)`. A
  sinusoidal/Fourier time embedding is concatenated with `x`; an MLP outputs the
  velocity of shape `(D,)`. mfm's loss/sampler drive it unchanged.
- **image** (`C, H, W`) → `mfm.DiTMFM` sized from the shape (in_channels,
  input_size, patch size, etc.), wrapped in `SIModelWrapper`, as in the prior
  MNIST run.

## Component: training, sampling, verdict

- **Train** (`train.py`): reuse `get_consistency_loss_fn` with the pure-FM config
  above; Adam + grad clip; `run.log(step=, fm_loss=, total=)` per step.
- **Sample** (`sample.py`): `ode_sampler_fn` from noise with `t_cond=0`.
- **Eval / verdict** (asserted in code, per the harness contract):
  - **gaussians**: quantitative — **energy distance** between generated and true
    samples below a threshold `gate`. A real distributional correctness check.
  - **mnist**: final FM loss below a threshold `gate`, plus a saved sample grid
    for eyeballing.

## Component: the experiment framework

`experiments/0001_flow_matching/run.py`:

```python
class FlowMatchingConfig(Config):
    seed: int = 0
    dataset: str = "gaussians"     # key into DATASETS
    n_steps: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    sampler_steps: int = 100       # ODE integration steps
    n_eval_samples: int = 2048
    gate: float = 0.05             # dataset-appropriate threshold (overridden per variant)
    # flat model knobs (only the relevant ones apply per modality):
    mlp_width: int = 256
    mlp_depth: int = 4
    dit_hidden: int = 128
    dit_depth: int = 4
```

`VARIANTS` (exact budgets/gates finalised in the plan):
- `gaussians` — CPU-fast, real energy-distance gate.
- `mnist` — GPU, `dataset=mnist`, gate on final FM loss.
- `smoke` — tiny budget, loose gate, plumbing only.

`main` resolves the config, `start_run(...)`, trains, samples, evaluates, writes
the verdict, and calls the spec's `visualize` into the run dir.

Run: `uv run python experiments/0001_flow_matching/run.py gaussians`.

## mypy

At the workspace-root `pyproject.toml`:

```toml
[tool.mypy]
mypy_path = ["packages/physics-informed-flow-map/src"]
strict = true
disallow_any_generics = false
disallow_any_explicit = false

[[tool.mypy.overrides]]
module = ["mfm.*", "diffusers.*", "torchvision.*", "matplotlib.*"]
ignore_missing_imports = true
```

- `mypy` added to the root `dev` group.
- Our package code is typed; untyped reused deps (notably `mfm`) are silenced.
- Command: `uv run mypy packages/physics-informed-flow-map/src packages/physics-informed-flow-map/tests`.
- The digit-prefixed experiment dir is not importable, so mypy stays focused on
  the package for now (matches catan's separate-experiments approach).

## Tests

Package-level pytest under `packages/physics-informed-flow-map/tests/`
(`pytest` added to the root `dev` group). Fast plumbing — correct shapes +
gradients flow — not quality checks.

- **`test_datasets.py`** — for each entry in `DATASETS`: building the dataset
  yields `(x1, label)` with `x1.shape == spec.shape`, labels in
  `[0, num_classes)` (or absent when unconditional), and a `DataLoader` batches
  cleanly.
- **`test_models.py`** — for each modality (vector + image): `build_model`
  produces a model whose `.v(...)` returns a velocity of the input shape
  (**forward**), and a scalar loss `.backward()` populates non-`None` grads on
  the parameters (**backward**). Tiny shapes/configs so it runs on CPU in
  seconds.
- Command: `uv run pytest packages/physics-informed-flow-map/tests`.

## Out of scope (YAGNI)

- PDE / physics datasets and physics-residual losses (the eventual research;
  this framework is the clean foundation they attach to).
- Class-conditional sampling / CFG for MNIST (train conditionally if labels
  exist, but sampling stays unconditional for now).
- FID or other heavy generative metrics.

## Removal

Delete `experiments/0001_mnist_pipeline/` and `experiments/0002_pretrained_sample/`
(tracked in git, recoverable). The new framework is `0001_flow_matching`.
```
