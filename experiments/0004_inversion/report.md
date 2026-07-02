# 0004 — Inversion (camp-A steering of a trained prior)

Status: open

## Hypothesis

A prior trained offline by one of the training frameworks (`0001_flow_matching`,
`0002_flow_map`, `0003_diffusion`) can be steered at inference by the wave equation to
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

## Method comparison: flow_tilt vs MFM-G vs FMRG-E

All three methods use the MFM prior's *marginal* velocity for the base ODE step. Their
differences are entirely in how they steer that step with the seismic misfit.

### Velocity calls on the MFM prior

| Quantity | flow_tilt | MFM-G | FMRG-E |
|---|---|---|---|
| Base drift call | `v(t, t, xt, 0, 0)` | `v(t, t, xt, 0, 0)` | `v(t, t, xt, 0, 0)` |
| Guidance call | — | `v(0, 1, ε, t, xt)` × mc_samples | — |
| Tweedie x1_hat | `xt + (1-t)·v` | replaced by posterior samples | `xt + (1-t)·v` |

The `t_cond=0, x_cond=0` is not a hack: the `SIModelWrapper` parametrization multiplies the
x_cond term by `t_cond`, so at `t_cond=0` the conditioning is exactly zeroed out regardless
of x_cond. This recovers the marginal velocity field, identical to the `get_unconditional_drift_ode`
call in `mfm.utils.steering`.

### Computation per outer step

| | flow_tilt | MFM-G (mc=K) | FMRG-E (n_opt=N) |
|---|---|---|---|
| Forward wave solves | 1 | K | N |
| Model forward passes | 1 | 1 + K | 1 |
| Model backward passes | 1 (through v) | 0 | 0 |
| Backprop through wave eq | yes (via F) | yes (IWAE score) | yes (inner loop) |

Notes:
- flow_tilt: single Tweedie estimate x1_hat, then one `F(x1_hat)` and backprop through F. No
  backprop through the model. Gradient applied at constant `guidance_strength` (no time
  weighting).
- MFM-G: draws K posterior samples `x1^k ~ p(x1|xt)` by running fresh noise through the
  conditional map `v(0,1,ε,t,xt)`. At each outer step, `(t, xt)` update while `(s=0, u=1)`
  and fresh ε are always re-sampled. Reward evaluated on each posterior sample; IWAE
  importance weights correct for sampling bias. Does NOT backprop through the model.
- FMRG-E: N inner gradient-descent steps on x1_hat in x1-space, each costing one wave solve
  + backprop through F. No backprop through the model. Correction backprojected to xt with
  time-dependent weight `wt = (1-t_cur)·t_next` (MFM convention). This weight peaks near
  t≈0.5 and decays to zero at both endpoints; it comes from the FMRG optimal-control
  derivation.

### Why FMRG-E for FWI

FMRG-E is the natural choice for measurement-based inverse problems. The wave-equation
gradient `J_F^T(F(x1_hat)-d_obs)` evaluated in x1-space is a physical gradient (reverse-time
migration direction) that points toward geophysically plausible corrections. For neural-network
rewards, gradients can point adversarially off-manifold, which is when FMRG-J (backprop
through the flow map's Jacobian to stay on-manifold) becomes preferable. FWI is in the same
category as SR/deblur/inpainting, for which the FMRG authors exclusively use FMRG-E in their
examples. With n_opt=1, FMRG-E is essentially DPS with the FMRG time-dependent weight
instead of constant guidance_strength; increasing n_opt adds more measurement consistency
per step.

### Update rule (MFM time convention, t=0 noise, t=1 data)

```
# Base velocity (marginal, unconditional):
v = prior.v(t_cur, t_cur, xt, zeros, zeros)

# Tweedie estimate (mean of marginal posterior p(x1|xt)):
x1_hat = xt + (1 - t_cur) * v

# FMRG-E inner loop (n_opt steps on x1_hat):
x1_opt = x1_hat - α·∇_{x1} ‖F(x1) - d_obs‖²   # repeated n_opt times

# Correction weight (optimal-control result):
wt = (1 - t_cur) * t_next

# Euler step with backprojected correction:
xt_next = xt + dt * v + wt * (x1_opt - x1_hat)
          ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
          prior advance  measurement correction
```

Contrast with flow_tilt:
```
# Same velocity call, same Tweedie:
v = prior.v(t, t, xt, zeros, zeros)
x1_hat = xt + (1-t) * v
loss = ‖F(x1_hat) - d_obs‖²
xt_next = xt + dt*v - guidance_strength * ∇_{xt} loss   # grad through F only, constant weight
```

The structural difference: FMRG-E optimizes in x1-space then backprojects with wt; flow_tilt
differentiates the loss w.r.t. xt directly and applies with constant strength.

## Results

(pending — point `ckpt=` at trained priors and run `experiment=single` / `experiment=eval`.)

## Decision

Status: open
