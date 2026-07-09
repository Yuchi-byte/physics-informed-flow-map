# 0005_analysis — cross-framework post-hoc analyses

Standalone analyses that *consume* trained checkpoints and run outputs from the other
frameworks (0001–0004) instead of training anything. Unlike the training frameworks there
is no `run.py`/`conf/`: each analysis is one self-contained argparse script, because the
interesting configuration (which checkpoints, which sweep grid) is pinned in the script and
`config.json` for auditability, not swept via Hydra.

## Conventions

- **One script per analysis, prefixed by topic**:
  - `prior_*.py` — prior/training quality (sample quality, NFE trade-offs, coverage).
  - `inversion_*.py` — inference-time steering comparisons (methods, misfits, budgets).
  - Future topics follow the same pattern (`data_*`, `physics_*`, `cost_*`, …).
- **Runs land in** `runs/0005_analysis/<script-stem>_<UTC-stamp>/` via the shared
  `start_run` harness: `config.json` (pinned checkpoints + grids + dataset fingerprint),
  `summary.json` (final scalars), plus the analysis's figures and `results.md`. The
  harness creates `checkpoints/` and the jsonl mirrors lazily, so analysis runs stay
  free of empty scaffolding. Everything also syncs to wandb (project
  `physics-informed-flow-map-0005_analysis`).
- **Every script has a `--smoke` flag** for a fast plumbing check (tiny budgets, no
  strength claim), mirroring the training frameworks' `experiment=smoke` variants.
- Findings go to `experiments/JOURNAL.md` as usual, citing the run directory.

## Scripts

- `prior_quality_vs_nfe.py` — sample quality (per-family energy distance vs the held-out
  split) as a function of NFE for the 0001/0002/0003 priors; isolates the flow-map
  off-diagonal's few-step value against the same checkpoint's ODE sampler, with a
  real-vs-real noise floor. Note: DDPM NFE grids must divide `num_train_timesteps`
  (diffusers "leading" spacing truncates the reverse chain for non-divisor counts).
