# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class of
related runs sharing machinery, not a single run. Prefer adding a *variant* to an
existing framework over scaffolding a new number.

## Layout

```
experiments/
├── README.md                 # this contract
├── JOURNAL.md                # one verdict line per concluded finding
├── new.py                    # scaffold the next framework
└── NNNN_slug/
    ├── run.py                # entry point: @hydra.main + Config subclass
    ├── conf/
    │   ├── config.yaml       # base defaults + hydra block
    │   └── experiment/       # config groups (default, smoke, …)
    ├── <helpers>.py          # framework-specific machinery
    └── report.md             # Hypothesis → Setup → Results → Decision
```

## Anatomy of a framework

- **`run.py`** is a `@hydra.main` entry point. It declares a typed `Config`
  subclass (`physics_informed_flow_map.experiment.Config`), composes a
  `DictConfig` from `conf/`, validates it via `Config.from_dictconfig(cfg)`
  (strict — `extra="forbid"`, a typo'd override is an error), then drives the run.
- **`conf/`** holds the Hydra config: `config.yaml` (base defaults + the `hydra`
  block) and `experiment/*.yaml` config groups (variants, each starting with
  `# @package _global_`). Select a variant with `experiment=<name>`.
- The run lifecycle is owned by the harness: `start_run(experiment, run_dir, config)`
  → `run.log(**metrics)` per step → `run.finish(verdict, **summary)`. Tracking goes
  to Weights & Biases; checkpoints are saved locally.

## Where results land

Tracking (config, metrics, sample images, verdict) goes to **Weights & Biases**.
Local artifacts land in the git-ignored `runs/` at the repo root:

```
runs/<framework>/<UTC-stamp>/        # = hydra.run.dir
├── .hydra/config.yaml               # Hydra's composed-config snapshot
├── checkpoints/step_<N>.pt          # local checkpoints (final always saved)
└── samples*.png                     # eval images (also logged to wandb)
```

The verdict is recorded in the wandb run summary and printed to the console.
wandb captures the git commit + a diff patch natively, so runs stay reproducible.

## Running

```bash
WANDB_MODE=online uv run python experiments/NNNN_slug/run.py [experiment=<variant>] [key=value ...]
```

First runs (or any machine without a wandb login) should use `WANDB_MODE=disabled`
for a no-tracking plumbing check, or `WANDB_MODE=offline` to record locally and
`wandb sync` later; `online` requires `wandb login`.

Examples:

```bash
uv run python experiments/0001_flow_matching/run.py experiment=smoke
uv run python experiments/0001_flow_matching/run.py experiment=mnist eval_every=500 ckpt_every=1000
```

## Conventions

- **Verdicts are asserted in code**, never by eye — gate on a threshold and pass
  it to `run.finish`.
- Every framework ships a `smoke` variant with trivial budgets for a fast
  end-to-end plumbing check (no strength claim).
- Record findings in `report.md` (cites run directories) and mirror the one-line
  verdict to `JOURNAL.md`. Don't journal in package docs.

## New framework

```bash
uv run python experiments/new.py "short title of the idea"
```
