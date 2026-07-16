# d_obs f-k spectrum trajectory grid

## Goal

Produce a spectral (frequency–wavenumber) twin of the existing `plot_dobs_trajectory`
data-space grid. Where the current figure shows each `d_obs` as a time × receiver shot
gather, the new figure shows each `d_obs` as its 2-D f-k spectrum. Delivered in two parts:

1. **Validate now** — build the pure transform and panel renderer, and confirm it on the
   gathers already saved to disk (`d_obs_inverted_sample0.npz`: true + final-inverted
   sample 0). No inversion re-run.
2. **Wire into the run** — the next `experiments/0004` inversion automatically emits the
   full spectral grid alongside `..._dobs_traj_{linear,log}.png`.

## Physics metadata (from `physics/forward.py`)

- `dt = 1e-3 s` → temporal Nyquist `f_Nyq = 500 Hz`.
- receiver spacing `dx = 10 m` → spatial Nyquist `k_Nyq = 0.05` cycles/m.
- `nt = 1001`, `n_receivers = 70`, Ricker centre `15 Hz`, `n_sources = 5`.

The forward operator's `dx`/`dt` are the source of truth; the plotting functions receive
them as arguments (do not hard-code).

## Conventions (decided)

- **Representation:** 2-D f-k spectrum, one image per gather, same grid shape as the
  gather figure.
- **Temporal axis (Y):** `rfft` over time → positive frequencies `0 … f_Nyq`. The real
  signal's spectrum is Hermitian, so the negative-f half is redundant.
- **Spatial axis (X):** full FFT over receivers + `fftshift` → signed wavenumber
  `−k_Nyq … +k_Nyq` in cycles/m, so left- and right-going wavefronts separate.
- **Magnitude:** `20·log10(|S| / peak)`, floored at `−80 dB`, sequential cmap (`magma`).
  Normalized to a **shared global peak** across the true column and all frames, mirroring
  how the gather grid shares one `vabs` so color is comparable across columns.
- **Display crop:** default-crop the frequency axis to `fmax = 60 Hz` (≈4× centre freq);
  the full 0–500 Hz range is almost entirely empty for a 15 Hz source. `fmax` is a
  parameter, adjustable. Full k range kept.

## Components

All in `inversion/single_target.py`, next to their gather twins.

### `fk_spectrum(gather, dt, dx, *, peak=None)` — pure, no matplotlib
- Input `gather`: `(n_receivers, nt)` real array.
- Compute `S = fftshift(rfft(gather, axis=time), axes=receiver)` — half-plane in f
  (rfft), full and centered in k (fftshift over the receiver axis).
- `mag = |S|`; `mag_db = 20·log10(mag / peak)` with `peak` defaulting to `mag.max()`
  (callers pass a shared global peak for the grid); floor at `−80 dB`.
- Returns `(mag_db, f_axis, k_axis)` where
  `f_axis = rfftfreq(nt, dt)` (Hz), `k_axis = fftshift(fftfreq(n_receivers, dx))` (cyc/m).
- Pure and independently testable.

### `_fk_imshow(ax, gather, *, dt, dx, vmax_db, fmax, cmap="magma")`
- Calls `fk_spectrum`, crops rows to `f_axis <= fmax`, `imshow` with `extent` set from the
  (cropped) f and k axes so ticks read in Hz / cycles-per-m; `vmin=vmax_db−80`, `vmax=vmax_db`
  (with the shared-peak normalization `vmax_db = 0`). Returns the `AxesImage`.

### `plot_dobs_spectrum_trajectory(...)`
Same signature and grid as `plot_dobs_trajectory`
(`v_true, frames_norm, frame_steps, d_obs_true, forward_fn, out_png, *, title,
total_steps, map_label, fmax=60.0`):
- Re-simulate `d_frames = forward_fn(frames_norm)` (as the gather version does).
- Compute one **shared global peak** over `|FFT|` of the true gather and every frame gather
  (all sources) so dB is comparable everywhere.
- **Row 0 = velocity maps, identical to the gather figure** (viridis, shared scale, titled
  with map label + step). Unchanged.
- Rows 1..n_src = f-k spectra: col 0 = true d_obs, cols 1.. = frames. One shared dB
  colorbar labeled `magnitude (dB, rel. peak)`. Axis labels: `frequency (Hz)` on Y for
  col 0, `wavenumber (cyc/m)` on X for the bottom row.
- Title mirrors the gather figure:
  `{title}\nd_obs f-k spectrum from {map_label} predictions over {total_steps} steps`.

## Wiring

In the `traj_capture` block (`single_target.py` ~L206), after the `for sc in dobs_scales`
loop that emits the gather grids, add a single call:

```python
plot_dobs_spectrum_trajectory(
    v_true, frames0, list(traj_capture["steps"]), d_obs, forward_fn,
    out_png.with_name(f"{method_name}_g{gs:.2g}_dobs_fk_traj.png"),
    title=f"{cmp_label or method_name} · {label} · sample 0",
    total_steps=int(traj_capture["total_steps"]), map_label=map_label,
)
```

Emitted **once** (not looped over `dobs_scales`) — dB is itself the scale, so there is no
linear/log pair.

## Part 1 delivery (validate, no re-run)

The intermediate Tweedie frames are ephemeral (only PNGs + final reconstructions are
persisted), so the exact twin of the shown PNG cannot be regenerated from a run dir. A
throwaway validation script (kept in `scratchpad/`, not committed to package src) loads
`runs/0004_inversion/flow_matching_flow_tilt_2026-07-16T20-36-27Z/d_obs_inverted_sample0.npz`
(keys `d_obs_true`, `d_obs_inverted`, both `(n_src, n_rec, nt)`) and renders a small f-k
grid: true vs final-inverted per source, using `fk_spectrum` + `_fk_imshow` with the real
`dt`/`dx`. This confirms the transform and rendering on real gathers. The exact 7-column
trajectory twin arrives on the next real inversion run via the wiring.

Persisting trajectory frames to npz is **out of scope** (YAGNI) — the wiring emits the
grid every run.

## Testing

Extend `packages/physics-informed-flow-map/tests/test_dobs_viz.py`:
- `fk_spectrum` axis lengths and ranges: `f_axis[-1] == f_Nyq`, `k_axis` symmetric to
  `±k_Nyq`, shapes match `(n_freq, n_rec)`.
- dB sanity: all `mag_db <= 0` when `peak = mag.max()`, and `max(mag_db) == 0`.
- Energy/Hermitian sanity: a known single-frequency input peaks at the expected `f`/`k` bin.
- `plot_dobs_spectrum_trajectory` writes a valid non-empty PNG with a
  `(1+n_src) × (1+n_frames)` axes grid.

## Out of scope

- Re-running the shown config to reproduce the exact PNG.
- Persisting intermediate frames to disk.
- Any change to the existing gather figure or its linear/log switch.
