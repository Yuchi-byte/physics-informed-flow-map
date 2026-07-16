# d_obs f-k Spectrum Trajectory Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frequency–wavenumber (f-k) spectral twin of the `plot_dobs_trajectory` grid — each `d_obs` shot gather rendered as its 2-D f-k spectrum — and wire it so the next inversion run emits it automatically.

**Architecture:** A pure NumPy transform (`fk_spectrum`) computes the 2-D f-k magnitude in dB. A panel helper (`_fk_imshow`) renders one spectrum. A grid function (`plot_dobs_spectrum_trajectory`) mirrors the existing `plot_dobs_trajectory` layout (velocity maps on top, one seismic-source row per source, columns = true + trajectory frames). The run driver calls it once per method in the `traj_capture` block. A throwaway scratchpad script validates the transform on saved true/final gathers.

**Tech Stack:** Python 3.12, NumPy, PyTorch, matplotlib (Agg backend), pytest, `uv`.

## Global Constraints

- All new code lives in `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py`, beside the existing gather functions.
- Physics metadata is passed as parameters, never hard-coded inside the transform: default `dt = 1e-3` s, `dx = 10.0` m (the `physics/forward.py` defaults). Temporal Nyquist `f_Nyq = 500 Hz`, spatial Nyquist `k_Nyq = 0.05` cyc/m.
- f-k conventions: `rfft` over time → positive frequencies `0 … f_Nyq`; full FFT + `fftshift` over receivers → signed wavenumber `−k_Nyq … +k_Nyq` (cyc/m); magnitude `20·log10(|S|/peak)` normalized to a **shared global peak** across the true column and all frames, floored at `−80 dB`, sequential `magma` cmap; frequency axis default-cropped to `fmax = 60.0 Hz` (parameter, adjustable).
- Run all commands from the repo root `/home/markhaoxiang/Projects/physics-informed-flow-map`.
- Test command form: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py::<name> -v`.
- Commit after each task with the pre-commit hook (ruff + mypy + pytest) passing.

---

### Task 1: `fk_spectrum` pure transform

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py` (add `_fk_mag` and `fk_spectrum` near `_seismic_imshow`, ~L400)
- Test: `packages/physics-informed-flow-map/tests/test_dobs_viz.py`

**Interfaces:**
- Consumes: `numpy as np` (already imported at `single_target.py:18`).
- Produces:
  - `_fk_mag(gather: np.ndarray, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(mag, f_axis, k_axis)` where `mag` has shape `(n_freq, n_rec)` = (f, k), `f_axis = rfftfreq(nt, dt)` (Hz, ascending `0..f_Nyq`), `k_axis = fftshift(fftfreq(n_rec, dx))` (cyc/m, ascending `−k_Nyq..+k_Nyq`).
  - `fk_spectrum(gather: np.ndarray, dt: float, dx: float, *, peak: float | None = None, floor_db: float = -80.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(mag_db, f_axis, k_axis)`; `mag_db = 20*log10(clip(mag/peak, 10**(floor_db/20), None))`, `peak` defaults to `mag.max()`.

- [ ] **Step 1: Write the failing tests**

Add to `packages/physics-informed-flow-map/tests/test_dobs_viz.py`:

```python
import numpy as np

from physics_informed_flow_map.inversion.single_target import fk_spectrum


def test_fk_spectrum_axes_ranges() -> None:
    n_rec, nt, dt, dx = 70, 1001, 1e-3, 10.0
    gather = np.random.randn(n_rec, nt)
    mag_db, f_axis, k_axis = fk_spectrum(gather, dt, dx)
    # temporal: rfft -> nt//2 + 1 positive freqs, 0 .. Nyquist(=1/(2dt)=500 Hz)
    assert f_axis.shape == (nt // 2 + 1,)
    assert f_axis[0] == 0.0
    assert np.isclose(f_axis[-1], 1.0 / (2 * dt))  # 500 Hz
    # spatial: full fft over receivers, fftshifted -> symmetric about 0, Nyquist=1/(2dx)=0.05
    assert k_axis.shape == (n_rec,)
    assert np.isclose(np.abs(k_axis).max(), 1.0 / (2 * dx))  # 0.05 cyc/m
    assert np.all(np.diff(k_axis) > 0)  # ascending
    assert mag_db.shape == (f_axis.size, k_axis.size)  # (f, k)


def test_fk_spectrum_db_normalized_to_peak() -> None:
    gather = np.random.randn(32, 128)
    mag_db, _, _ = fk_spectrum(gather, 1e-3, 10.0)
    assert mag_db.max() <= 0.0 + 1e-9          # peak normalized to 0 dB
    assert np.isclose(mag_db.max(), 0.0)
    assert mag_db.min() >= -80.0 - 1e-9        # floored


def test_fk_spectrum_locates_single_tone() -> None:
    # A pure temporal tone at f0 with a single spatial wavelength should peak
    # at the matching (f, k) bin.
    n_rec, nt, dt, dx = 64, 512, 1e-3, 10.0
    f0, k0 = 20.0, 1.0 / (8 * dx)  # 20 Hz, wavelength 8 receivers
    t = np.arange(nt) * dt
    x = np.arange(n_rec) * dx
    gather = np.cos(2 * np.pi * f0 * t)[None, :] * np.cos(2 * np.pi * k0 * x)[:, None]
    mag_db, f_axis, k_axis = fk_spectrum(gather, dt, dx)
    fi, ki = np.unravel_index(np.argmax(mag_db), mag_db.shape)
    assert np.isclose(f_axis[fi], f0, atol=f_axis[1] - f_axis[0])
    assert np.isclose(abs(k_axis[ki]), k0, atol=k_axis[1] - k_axis[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -k fk_spectrum -v`
Expected: FAIL — `ImportError: cannot import name 'fk_spectrum'`.

- [ ] **Step 3: Implement `_fk_mag` and `fk_spectrum`**

Add to `single_target.py` immediately after `_seismic_imshow` (which ends at ~L419):

```python
def _fk_mag(
    gather: np.ndarray, dt: float, dx: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear f-k magnitude of one ``(n_receivers, nt)`` gather. ``rfft`` over time gives the
    positive-frequency half (the real signal's spectrum is Hermitian); a full ``fft`` +
    ``fftshift`` over receivers gives signed wavenumber so left/right-going wavefronts separate.
    Returns ``(mag, f_axis, k_axis)`` with ``mag`` shaped ``(n_freq, n_rec)`` = (f, k), both axes
    ascending. ``dt`` in seconds, ``dx`` receiver spacing in metres."""
    n_rec, nt = gather.shape
    s = np.fft.fftshift(
        np.fft.fft(np.fft.rfft(gather, axis=1), axis=0), axes=0
    )  # (n_rec, n_freq): fft over receivers (centred), rfft over time
    mag = np.abs(s).T  # (n_freq, n_rec) = (f, k)
    f_axis = np.fft.rfftfreq(nt, dt)  # 0 .. f_Nyq, ascending
    k_axis = np.fft.fftshift(np.fft.fftfreq(n_rec, dx))  # -k_Nyq .. +k_Nyq, ascending
    return mag, f_axis, k_axis


def fk_spectrum(
    gather: np.ndarray,
    dt: float,
    dx: float,
    *,
    peak: float | None = None,
    floor_db: float = -80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D f-k spectrum of one ``(n_receivers, nt)`` gather in dB. Magnitude is
    ``20*log10(|S| / peak)`` floored at ``floor_db``; ``peak`` defaults to this gather's own
    magnitude peak, but callers pass a shared global peak so panels stay comparable (the peak
    then maps to 0 dB everywhere). Returns ``(mag_db, f_axis, k_axis)`` — see :func:`_fk_mag`
    for axis conventions."""
    mag, f_axis, k_axis = _fk_mag(gather, dt, dx)
    p = (float(mag.max()) if peak is None else peak) or 1.0
    mag_db = 20.0 * np.log10(np.clip(mag / p, 10.0 ** (floor_db / 20.0), None))
    return mag_db, f_axis, k_axis
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -k fk_spectrum -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py packages/physics-informed-flow-map/tests/test_dobs_viz.py
git commit -m "feat(0004): fk_spectrum f-k transform for d_obs gathers"
```

---

### Task 2: `_fk_imshow` panel + `plot_dobs_spectrum_trajectory` grid

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py` (add both functions after `fk_spectrum`)
- Test: `packages/physics-informed-flow-map/tests/test_dobs_viz.py`

**Interfaces:**
- Consumes: `fk_spectrum`, `_fk_mag` (Task 1); `to_mps_native` (imported at `single_target.py:29`); `plt`, `np`, `torch`, `Tensor`, `AxesImage`, `Callable`, `Path` (already imported).
- Produces:
  - `_fk_imshow(ax, gather: np.ndarray, *, dt: float, dx: float, peak: float, fmax: float, floor_db: float = -80.0, cmap: str = "magma") -> AxesImage`
  - `plot_dobs_spectrum_trajectory(v_true: Tensor, frames_norm: Tensor, frame_steps: list[int], d_obs_true: Tensor, forward_fn: Callable[[Tensor], Tensor], out_png: Path, *, title: str, total_steps: int, map_label: str = "Tweedie", dt: float = 1e-3, dx: float = 10.0, fmax: float = 60.0) -> None`

- [ ] **Step 1: Write the failing tests**

Add to `packages/physics-informed-flow-map/tests/test_dobs_viz.py` (reuses the existing `_stub_forward` helper in that file):

```python
from physics_informed_flow_map.inversion.single_target import (
    plot_dobs_spectrum_trajectory,
)


def test_plot_dobs_spectrum_trajectory_writes_png(tmp_path: Path) -> None:
    n_frames, n_src, n_rec, nt = 2, 5, 12, 40
    v_true = torch.rand(70, 70) * 1000 + 1500
    frames_norm = torch.rand(n_frames, 1, 16, 16) * 2 - 1
    d_obs_true = torch.randn(n_src, n_rec, nt)
    out = tmp_path / "fk_traj.png"
    plot_dobs_spectrum_trajectory(
        v_true,
        frames_norm,
        [0, 3],
        d_obs_true,
        _stub_forward(n_src, n_rec, nt),
        out,
        title="demo",
        total_steps=4,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_dobs_spectrum_trajectory_panel_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_informed_flow_map.inversion.single_target as st

    seen: dict[str, tuple[int, int]] = {}
    real = st.plt.subplots

    def spy(nrows: int, ncols: int, **kw: object):
        seen["shape"] = (nrows, ncols)
        return real(nrows, ncols, **kw)

    monkeypatch.setattr(st.plt, "subplots", spy)
    plot_dobs_spectrum_trajectory(
        torch.rand(70, 70),
        torch.rand(2, 1, 16, 16),
        [0, 3],
        torch.randn(5, 12, 40),
        _stub_forward(5, 12, 40),
        tmp_path / "t.png",
        title="d",
        total_steps=4,
    )
    assert seen["shape"] == (1 + 5, 1 + 2)  # (1 + n_src) rows, (1 + n_frames) cols
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -k spectrum_trajectory -v`
Expected: FAIL — `ImportError: cannot import name 'plot_dobs_spectrum_trajectory'`.

- [ ] **Step 3: Implement `_fk_imshow` and `plot_dobs_spectrum_trajectory`**

Add to `single_target.py` after `fk_spectrum`:

```python
def _fk_imshow(
    ax: plt.Axes,
    gather: np.ndarray,
    *,
    dt: float,
    dx: float,
    peak: float,
    fmax: float,
    floor_db: float = -80.0,
    cmap: str = "magma",
) -> AxesImage:
    """imshow one ``(n_receivers, nt)`` gather's f-k spectrum: temporal frequency (Hz) up the
    vertical axis (cropped to ``fmax``), signed wavenumber (cyc/m) across, magnitude in dB
    relative to the shared ``peak``. ``vmin``/``vmax`` are fixed to ``floor_db``/``0`` so colour
    reads the same in every panel."""
    mag_db, f_axis, k_axis = fk_spectrum(
        gather, dt, dx, peak=peak, floor_db=floor_db
    )
    fmask = f_axis <= fmax
    img = mag_db[fmask]  # (n_freq_crop, n_rec)
    extent = (
        float(k_axis[0]),
        float(k_axis[-1]),
        float(f_axis[fmask][0]),
        float(f_axis[fmask][-1]),
    )
    return ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=floor_db,
        vmax=0.0,
    )


def plot_dobs_spectrum_trajectory(
    v_true: Tensor,
    frames_norm: Tensor,
    frame_steps: list[int],
    d_obs_true: Tensor,
    forward_fn: Callable[[Tensor], Tensor],
    out_png: Path,
    *,
    title: str,
    total_steps: int,
    map_label: str = "Tweedie",
    dt: float = 1e-3,
    dx: float = 10.0,
    fmax: float = 60.0,
) -> None:
    """f-k spectral twin of :func:`plot_dobs_trajectory` for sample 0. Row 0 is the identical
    velocity-map row (true v + each prediction, viridis, shared scale). Rows below (one per
    seismic source) replace each ``d_obs`` gather with its 2-D f-k spectrum: column 0 is the true
    velocity's ``d_obs``, columns 1.. are the d_obs re-simulated from each ``frames_norm``
    prediction via ``forward_fn``. All spectra share one global dB peak so colour is comparable.

    ``frames_norm`` is ``(n_frames, 1, res, res)`` in [-1,1]; ``d_obs_true`` is
    ``(n_src, n_rec, nt)``. ``dt``/``dx`` set the frequency/wavenumber axes; ``fmax`` crops the
    frequency axis (default 60 Hz for the 15 Hz Ricker source)."""
    with torch.no_grad():
        d_frames = (
            forward_fn(frames_norm).detach().cpu().numpy()
        )  # (n_frames, n_src, n_rec, nt)
    v_hat = to_mps_native(frames_norm).detach().cpu().numpy()  # (n_frames, 70, 70) m/s
    vt = v_true.detach().cpu().numpy()
    dt_arr = d_obs_true.detach().cpu().numpy()  # (n_src, n_rec, nt)
    n_frames = int(frames_norm.shape[0])
    n_src = int(dt_arr.shape[0])
    n_cols = 1 + n_frames
    vlo, vhi = float(vt.min()), float(vt.max())

    # One shared linear-magnitude peak across the true column and every frame, all sources,
    # so the dB scale (peak -> 0 dB) is comparable across the whole grid.
    peak = 1.0
    for s in range(n_src):
        peak = max(peak, float(_fk_mag(dt_arr[s], dt, dx)[0].max()))
        for j in range(n_frames):
            peak = max(peak, float(_fk_mag(d_frames[j, s], dt, dx)[0].max()))

    fig, axes = plt.subplots(
        1 + n_src, n_cols, figsize=(2.1 * n_cols, 2.1 * (1 + n_src)), squeeze=False
    )
    # Row 0 — velocity maps (identical to plot_dobs_trajectory).
    vimg = axes[0, 0].imshow(vt, cmap="viridis", vmin=vlo, vmax=vhi)
    axes[0, 0].set_title("true v", fontsize=9)
    for j in range(n_frames):
        axes[0, 1 + j].imshow(v_hat[j], cmap="viridis", vmin=vlo, vmax=vhi)
        axes[0, 1 + j].set_title(f"{map_label}\nstep {frame_steps[j]}", fontsize=9)
    for c in range(n_cols):
        axes[0, c].axis("off")
    fig.colorbar(vimg, ax=axes[0, n_cols - 1], fraction=0.046, label="m/s")

    # Rows 1..n_src — f-k spectra, column 0 = true, columns 1.. = frames.
    im = None
    for s in range(n_src):
        r = 1 + s
        im = _fk_imshow(axes[r, 0], dt_arr[s], dt=dt, dx=dx, peak=peak, fmax=fmax)
        axes[r, 0].set_ylabel(f"source {s + 1}\nfrequency (Hz)", fontsize=8)
        if s == 0:
            axes[r, 0].set_title("true d_obs", fontsize=9)
        for j in range(n_frames):
            _fk_imshow(
                axes[r, 1 + j], d_frames[j, s], dt=dt, dx=dx, peak=peak, fmax=fmax
            )
            if s == 0:
                axes[r, 1 + j].set_title(f"step {frame_steps[j]}", fontsize=9)
            axes[r, 1 + j].set_yticklabels([])
        for c in range(n_cols):
            if s < n_src - 1:
                axes[r, c].set_xticklabels([])
            else:
                axes[r, c].set_xlabel("wavenumber (cyc/m)", fontsize=8)
    fig.colorbar(
        im,
        ax=axes[1:, n_cols - 1].ravel().tolist(),
        fraction=0.046,
        label="magnitude (dB, rel. peak)",
    )
    fig.suptitle(
        f"{title} · f-k\nd_obs f-k spectrum from {map_label} predictions over "
        f"{total_steps} steps",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -k spectrum_trajectory -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py packages/physics-informed-flow-map/tests/test_dobs_viz.py
git commit -m "feat(0004): plot_dobs_spectrum_trajectory f-k grid"
```

---

### Task 3: Wire the f-k grid into the inversion run

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py:206-229` (the `traj_capture` block)
- Test: `packages/physics-informed-flow-map/tests/test_dobs_viz.py`

**Interfaces:**
- Consumes: `plot_dobs_spectrum_trajectory` (Task 2); the in-scope locals `v_true`, `frames0`, `traj_capture`, `d_obs`, `forward_fn`, `method_name`, `gs`, `cmp_label`, `label`, `map_label`, `out_png` already present in the block.
- Produces: an extra output file `{method_name}_g{gs:.2g}_dobs_fk_traj.png` per run; no new symbols.

- [ ] **Step 1: Write the failing test**

The wiring calls `plot_dobs_spectrum_trajectory` once, unconditional on `dobs_scales`. Assert that by spying on it. Add to `test_dobs_viz.py`:

```python
def test_run_emits_one_fk_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    # The traj_capture block must call plot_dobs_spectrum_trajectory exactly once
    # (dB is its own scale — not looped over dobs_scales like the gather grids).
    import physics_informed_flow_map.inversion.single_target as st

    calls: list[str] = []
    monkeypatch.setattr(
        st,
        "plot_dobs_spectrum_trajectory",
        lambda *a, **k: calls.append(str(a[5])),  # out_png is positional arg 6
    )
    src = st.__file__
    with open(src) as f:
        body = f.read()
    # The call sits in the traj_capture block and names the _dobs_fk_traj.png output.
    assert "plot_dobs_spectrum_trajectory(" in body
    assert "_dobs_fk_traj.png" in body
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py::test_run_emits_one_fk_grid -v`
Expected: FAIL — `_dobs_fk_traj.png` not found in the module source.

- [ ] **Step 3: Add the wiring call**

In `single_target.py`, the `traj_capture` block currently ends its scale loop at L217-229. Immediately after the `for sc in dobs_scales:` loop (after the closing of the `plot_dobs_trajectory(...)` call block, before `summary = {`), insert:

```python
        plot_dobs_spectrum_trajectory(
            v_true,
            frames0,
            list(traj_capture["steps"]),
            d_obs,
            forward_fn,
            out_png.with_name(f"{method_name}_g{gs:.2g}_dobs_fk_traj.png"),
            title=f"{cmp_label or method_name} · {label} · sample 0",
            total_steps=int(traj_capture["total_steps"]),
            map_label=map_label,
        )
```

Indentation: this sits inside `if traj_capture and traj_capture.get("frames") is not None:` at the same level as the `for sc in dobs_scales:` loop (8 spaces).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py::test_run_emits_one_fk_grid -v`
Expected: PASS.

- [ ] **Step 5: Run the full viz test module + mypy**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -v`
Expected: all PASS.
Run: `uv run mypy packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py`
Expected: no new errors in the added functions.

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py packages/physics-informed-flow-map/tests/test_dobs_viz.py
git commit -m "feat(0004): emit d_obs f-k spectrum grid during inversion"
```

---

### Task 4: Part 1 — validate on saved true/final gathers

**Files:**
- Create: `/tmp/claude-1000/-home-markhaoxiang-Projects-physics-informed-flow-map/1a95969a-cc2c-474f-a7d6-9f002dce85ee/scratchpad/validate_fk.py` (throwaway, NOT committed)

**Interfaces:**
- Consumes: `fk_spectrum`, `_fk_imshow` (Tasks 1–2); the saved npz at the relocated run dir.

- [ ] **Step 1: Write the validation script**

```python
"""Throwaway: render an f-k grid from saved true vs final-inverted sample-0 gathers,
to eyeball the fk_spectrum transform + panel on real data (no inversion re-run)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics_informed_flow_map.inversion.single_target import _fk_imshow, _fk_mag

NPZ = Path(
    "runs/0004_inversion/2026_07_15_marmousi_crop05_inversion/"
    "flow_matching_flow_tilt_2026-07-16T20-36-27Z_ot_g1_steps400/"
    "d_obs_inverted_sample0.npz"
)
OUT = Path(
    "/tmp/claude-1000/-home-markhaoxiang-Projects-physics-informed-flow-map/"
    "1a95969a-cc2c-474f-a7d6-9f002dce85ee/scratchpad/fk_validation.png"
)
DT, DX, FMAX = 1e-3, 10.0, 60.0

d = np.load(NPZ)
d_true = d["d_obs_true"]  # (n_src, n_rec, nt)
d_inv = d["d_obs_inverted"]  # (n_src, n_rec, nt)
n_src = d_true.shape[0]

peak = 1.0
for arr in (d_true, d_inv):
    for s in range(n_src):
        peak = max(peak, float(_fk_mag(arr[s], DT, DX)[0].max()))

fig, axes = plt.subplots(n_src, 2, figsize=(6, 2.4 * n_src), squeeze=False)
for s in range(n_src):
    _fk_imshow(axes[s, 0], d_true[s], dt=DT, dx=DX, peak=peak, fmax=FMAX)
    im = _fk_imshow(axes[s, 1], d_inv[s], dt=DT, dx=DX, peak=peak, fmax=FMAX)
    axes[s, 0].set_ylabel(f"source {s + 1}\nfrequency (Hz)", fontsize=8)
    if s == 0:
        axes[s, 0].set_title("true d_obs · f-k", fontsize=9)
        axes[s, 1].set_title("inverted d_obs · f-k", fontsize=9)
    axes[s, 1].set_yticklabels([])
    for c in range(2):
        if s == n_src - 1:
            axes[s, c].set_xlabel("wavenumber (cyc/m)", fontsize=8)
fig.colorbar(
    im, ax=axes.ravel().tolist(), fraction=0.046, label="magnitude (dB, rel. peak)"
)
fig.suptitle("f-k spectrum validation · marmousi_crop05 · sample 0", fontsize=10)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
print(f"f-k grid shape per panel cropped to <= {FMAX} Hz; shared peak = {peak:.3e}")
```

- [ ] **Step 2: Run the validation script**

Run:
```bash
uv run python "/tmp/claude-1000/-home-markhaoxiang-Projects-physics-informed-flow-map/1a95969a-cc2c-474f-a7d6-9f002dce85ee/scratchpad/validate_fk.py"
```
Expected: prints `wrote .../fk_validation.png` and the shared-peak line; the PNG exists and is non-empty.

- [ ] **Step 3: Inspect the output**

Read the PNG (`fk_validation.png`) and confirm: source energy concentrated at low frequency (≈15 Hz band) near small |k|, spectra symmetric-ish in k, dB colour spans the full −80..0 range. Report the image to the user. No commit (scratchpad only).

---

## Notes for the executor

- No new dependencies; NumPy FFT + matplotlib already in the venv.
- The gather figure, its linear/log switch, and `dobs_scales` are untouched — the f-k grid is emitted once, independently.
- The pre-commit hook runs ruff + mypy + pytest on staged package files; keep functions typed (they are, above).
