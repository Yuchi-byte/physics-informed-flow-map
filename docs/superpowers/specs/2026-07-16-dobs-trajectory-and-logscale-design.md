# d_obs trajectory grid + log-scale seismic plotting

**Status:** approved, not yet implemented
**Date:** 2026-07-16
**Framework:** `experiments/0004_inversion`

## Problem

Two visualization gaps in the inversion runs:

1. The trajectory PNG shows, per sample, the noisy state `x_t` and its Tweedie prediction at
   `n_frames` points along the trajectory — all in **model space** (velocity maps). There is no
   **data-space** view of how the Tweedie prediction's simulated seismic evolves toward the
   observed `d_obs` as steering proceeds.
2. Seismic gathers are plotted on a symmetric linear scale (`RdBu_r`, `±vabs`). Low-amplitude
   arrivals are visually swamped by the high-amplitude direct wave, so structure in the coda is
   hard to see.

## Findings (current code)

- **Tweedie frames are not persisted.** `Run.make_step_saver`
  (`packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py:271`)
  collects `n_frames` evenly-spaced estimate snapshots (`est_frames`) and the noisy states
  (`xt_frames`) for the whole batch, renders one composite grid at the last checkpoint, then
  discards them. `reconstructions.npz` saves only the final velocity maps. So the new figure
  must **capture** sample-0's estimate frames during inversion.
- **The forward operator is cheap relative to the run.** `seismic_forward`
  (`inversion/bridge.py:34`) maps a normalized `(B,1,res,res)` velocity to
  `(B, n_src, n_rec, nt)` seismic via `simulate` (`physics/forward.py`, `n_sources=5`,
  `n_receivers=70`, `nt=1001`). Re-simulating `n_frames` (~5) captured frames post-inversion is
  negligible against the ~2000 solves a guided pass already does.
- **Column-0 d_obs is already in hand.** The observed `d_obs = forward(v_true)` is loaded in
  `invert_and_report` (`inversion/single_target.py:117`); the grid reuses it for the true-velocity
  column rather than recomputing.
- **Two seismic plotters exist**, both linear/`RdBu_r`/`±vabs`: `_plot_seismic`
  (`single_target.py:271`, the observed-`d_obs` figure) and `plot_dobs_compare`
  (`single_target.py:229`, the true/inverted/residual fit figure).
- **`d_obs.png` has one reference:** `experiments/0004_inversion/run.py:637`. No report or eval
  code reads it.

## Decisions

| Decision | Choice |
|---|---|
| Frame d_obs source | Recompute post-inversion from captured sample-0 frames (frames are not persisted today, so capture is added). Sampler-agnostic. |
| Method scope | All trajectory-producing methods (every `make_step_saver` call site). |
| Frame count | Reuse `cfg.n_frames`; the grid's checkpoints match the existing trajectory PNG exactly. |
| Log-scale figures | Observed `d_obs` figure and the new grid emit both scales. `plot_dobs_compare` stays linear-only. |
| Log implementation | `matplotlib.colors.SymLogNorm` (seismic is signed). |
| `d_obs.png` | Replaced by `d_obs_linear.png` + `d_obs_log.png`; update the one reference. |
| Velocity row | Show maps only (viridis, shared scale); no error annotation. |

## Components

### 1. Shared seismic-imshow helper — `single_target.py`

```python
def _seismic_imshow(ax, gather, *, scale, vabs, cmap="RdBu_r"):
    """imshow one (n_rec, nt) gather transposed (time down), linear or symmetric-log.
    gather: 2-D numpy array. scale: "linear" | "log". Returns the AxesImage."""
```

- `scale="linear"`: `imshow(gather.T, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)` — the
  current behavior verbatim.
- `scale="log"`: same, but `norm=SymLogNorm(linthresh, vmin=-vabs, vmax=vabs)` and no
  `vmin/vmax` kwargs. `linthresh = max(vabs * 1e-2, 1e-12)` — the ±linear band below which
  SymLogNorm is linear; above it, log. This keeps the diverging symmetric look while lifting the
  low-amplitude coda.

Both `_plot_seismic` and `plot_dobs_trajectory` route every gather through this helper. The
colorbar label notes the scale (`"amplitude (symlog)"` vs `"amplitude"`).

### 2. `_plot_seismic` gains `scale` — `single_target.py`

Signature `_plot_seismic(d_obs, gidx, out_png, *, scale="linear")`. Body swaps its inline
`imshow` for `_seismic_imshow(..., scale=scale, vabs=vabs)`. Title suffix `" · log"` when
`scale="log"`.

### 3. New figure — `plot_dobs_trajectory` — `single_target.py`

```python
def plot_dobs_trajectory(
    v_true,            # (70,70) m/s tensor
    frames_norm,       # (n_frames, 1, res, res) sample-0 Tweedie maps, normalized [-1,1]
    frame_steps,       # list[int], the trajectory step index of each frame
    d_obs_true,        # (n_src, n_rec, nt) tensor — reused for column 0
    forward_fn,        # frames_norm -> (n_frames, n_src, n_rec, nt); seismic_forward
    out_png, *, scale, title, total_steps,
): ...
```

- Recompute `d_frames = forward_fn(frames_norm)` once (`n_frames` solves). Reuse `d_obs_true`
  for column 0.
- Grid `gridspec` of `(1 + n_src)` rows × `(1 + n_frames)` cols.
  - **Row 0:** col 0 = `v_true` (viridis); cols `1..n_frames` = `to_mps_native(frames_norm)`
    each (viridis, shared `vmin/vmax` = true map's range). Titles: col 0 `"true v"`, others
    `f"step {frame_steps[j]}"`.
  - **Rows `1..n_src`:** row `s` = source `s`. Col 0 = `d_obs_true[s-1]`; cols `1..n_frames` =
    `d_frames[j, s-1]`. All via `_seismic_imshow` with one shared `vabs` = 99th percentile of
    `|stack(d_obs_true, d_frames)|` (so columns are directly comparable). Left-column ylabel
    `f"source {s}"`.
- One colorbar for the velocity row, one for the seismic block. `suptitle = title` (+ `" · log"`
  when logscale).
- **`scale` affects only the seismic rows.** The velocity row is always viridis on a positive
  linear scale — velocity is not signed and log there would be meaningless.
- Callable twice (linear, log) → two PNGs.

### 4. Capture hook — `Run.make_step_saver`

Add `capture: dict | None = None`. At `last_checkpoint`, after building `frames`, if `capture`
is not None:

```python
capture["frames"] = torch.stack([est_frames[s] for s in sorted(est_frames)])  # (n_frames, B, C?, H, W)
capture["steps"] = sorted(est_frames)
capture["total_steps"] = total_steps
```

Only the estimate frames (Tweedie/iterate), not the interleaved `x_t` — the grid is
Tweedie-vs-data. No behavior change when `capture is None`.

### 5. Wiring — `run.py`

- One shared `traj_cap: dict = {}` created near the other per-run capture dicts.
- Pass `capture=traj_cap` to **every** `make_step_saver(...)` call (all sampler branches +
  the FWI branch).
- Rename `obs_out`: drop the single `d_obs.png`; pass the run dir / stem so `invert_and_report`
  writes both scales.
- Pass `traj_cap`, a new `out_dobs_traj_png` stem, and `dobs_scales=("linear","log")` into
  `invert_and_report`.

### 6. Wiring — `invert_and_report` (`single_target.py`)

New params: `out_obs_png` stays but is emitted per scale; add `traj_capture=None`,
`out_dobs_traj_png=None`, `dobs_scales=("linear",)`, `forward_fn=seismic_forward`.

- Observed figure: for each `scale in dobs_scales`, call `_plot_seismic(d_obs, gidx,
  f"{stem}_{scale}.png", scale=scale)`.
- After the guided pass, if `traj_capture` has `"frames"`:
  - `frames_norm = traj_capture["frames"][:, 0]` → `(n_frames, C?, H, W)`; ensure a channel dim
    (`[:, None]` when 3-D, for the FWI native-grid path).
  - For each `scale in dobs_scales`, call `plot_dobs_trajectory(v_true, frames_norm,
    traj_capture["steps"], d_obs, forward_fn, f"{traj_stem}_{scale}.png", scale=scale,
    title=f"{cmp_label} · {label} · sample 0", total_steps=traj_capture["total_steps"])`.
  - Log each PNG to wandb (`traj/dobs_{scale}`).
- If `traj_capture` is empty (e.g. `guidance=0` unguided control, `n_frames=0`): skip silently.

## Output files (per guided run)

```
d_obs_linear.png                     # was d_obs.png
d_obs_log.png
<key>_dobs_traj_linear.png           # new grid, linear
<key>_dobs_traj_log.png              # new grid, symlog
```

(`<key>` mirrors the trajectory PNG stem, e.g. `dps_g2`.)

## FWI note

For `prior=none`, the row-0 maps are the current velocity **iterate**, not a Tweedie posterior
mean. The grid is still meaningful (data-fit convergence), but the title says `"iterate"` rather
than `"Tweedie"` for `method.name in {classical_fwi, realistic_fwi}`.

## Testing

Unit tests on synthetic arrays (no torch model, fast) in a new
`tests/test_dobs_viz.py`:

1. `_plot_seismic(scale="linear")` and `("log")` each write a nonempty PNG.
2. `plot_dobs_trajectory` writes a PNG of the expected panel count `(1+n_src)*(1+n_frames)` for a
   `2`-frame, `5`-source synthetic case, in both scales, with a stub `forward_fn`.
3. `make_step_saver(capture=d)` populates `d["frames"]`/`d["steps"]` with the right shape after a
   synthetic 3-step trajectory; `capture=None` leaves behavior unchanged.

Per the standing preference, these are TDD scaffolding and may be removed after the feature is
verified end-to-end.

## Non-goals

- No change to `plot_dobs_compare` (stays linear), the metrics, or `reconstructions.npz`.
- No new persisted arrays — the grid recomputes d_obs at render time; frames live only in the
  in-memory capture dict for the current run.
- No change to the acquisition geometry or the misfit.

## Verification

- `experiment=smoke` (untrained prior, `n_frames` small) emits the four PNGs without error.
- The real `prior=diffusion method=dps … n_samples=10` run produces a grid whose column-0 d_obs
  matches `d_obs_linear.png`, and whose log panels visibly lift the coda.
- `grep -rn "d_obs.png" experiments/ packages/` returns nothing after the rename.
