# 0004 — Inversion (camp-A steering of a trained prior)

Status: open

## Hypothesis

A prior trained offline by one of the training frameworks (`0001_flow_matching`,
`0002_flow_map`, `0003_baselines`) can be steered at inference by the wave equation to
recover a held-out OpenFWI velocity map, and the inference scheme — flow tilting (DPS
Tweedie), Meta-Flow-Map steering (the flow map's own posterior), or diffusion DPS — is the
lever worth comparing. All physics stays at inference; no prior is retrained here.

## Setup

This framework consolidates every inversion entry point. The prior and the inference scheme
are independent config groups:

- `prior=` — `flow_matching` | `flow_map` | `diffusion` (which prior to load + how).
- `method=` — `flow_tilt` | `unguided` | `mfm_g` | `mfm_gf` | `dps`.

Compatibility (validated in `InversionConfig`):

| method      | valid priors                       |
|-------------|------------------------------------|
| `unguided`  | flow_matching, flow_map, diffusion |
| `flow_tilt` | flow_matching, flow_map            |
| `mfm_g`     | flow_map                           |
| `mfm_gf`    | flow_map                           |
| `dps`       | diffusion                          |

- `run.py` (`experiment=single`) inverts one held-out target → a `true | v_hat | error`
  figure + scalars, sharing scoring/figure via `inversion.single_target`.
- `eval.py` (`experiment=eval`) scores several inverters (`evaluation.methods`) on the same
  held-out maps at a matched solve budget via `inversion.Evaluator`, writing
  `runs/0004_inversion/<ts>/results.md`.
- `experiment=smoke` — trivial budgets, untrained prior, CPU plumbing (no GPU/ckpt).

Caveat: `d_obs` is the same noiseless forward operator used inside guidance (an "inverse
crime"), so recovery is optimistic and data-fitting methods are flattered.

## Results

(pending — point `ckpt=` at trained priors and run `experiment=single` / `experiment=eval`.)

## Decision

Status: open
