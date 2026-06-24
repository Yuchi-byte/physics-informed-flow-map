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
    ├── run.py                # entry point: Config + VARIANTS + main()
    ├── <helpers>.py          # framework-specific machinery
    └── report.md             # Hypothesis → Setup → Results → Decision
```

## Anatomy of a framework

- **`run.py`** declares a typed `Config` (subclass of
  `physics_informed_flow_map.experiment.Config`), a `VARIANTS` dict of named
  presets, and a `main()` that resolves the config and drives the run.
- Config resolution: `Config.resolve(variant, overrides)` merges
  `defaults <- variant <- key=value CLI args`, then validates strictly
  (`extra="forbid"` — a typo'd override is an error, not a silent no-op).
- The run lifecycle is owned by the harness: `start_run(...)` →
  `run.log(**metrics)` per step → `run.finish(verdict, **summary)`.

## Where results land

Everything a run *produces* goes to the git-ignored `runs/` at the repo root:

```
runs/<framework>/<UTC-stamp>/
├── manifest.json    # argv, resolved config, git commit + dirty-diff digest, env
├── metrics.jsonl    # append-only, one record per run.log call
├── result.json      # {"verdict": ..., ...summary}
└── <checkpoints, samples, plots — framework-specific>
```

Every run is reproducible from its manifest (git commit + config).

## Running

```bash
uv run python experiments/NNNN_slug/run.py [variant] [key=value ...]
```

Examples:

```bash
uv run python experiments/0001_mnist_pipeline/run.py smoke
uv run python experiments/0001_mnist_pipeline/run.py default n_steps=500 lr=5e-4
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
