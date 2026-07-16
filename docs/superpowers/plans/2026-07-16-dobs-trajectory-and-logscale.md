# d_obs Trajectory Grid + Log-Scale Seismic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a data-space trajectory figure (sample-0 Tweedie predictions' simulated seismic per source, along the guided pass) and a linear/symmetric-log scale switch on the seismic plotters, in `experiments/0004_inversion`.

**Architecture:** A shared `_seismic_imshow(scale=...)` helper backs both the existing observed-seismic figure and a new `plot_dobs_trajectory` grid. `Run.make_step_saver` gains an opt-in `capture` dict that keeps sample-0's Tweedie frames; `invert_and_report` recomputes their d_obs post-inversion (~5 solves) and renders the grid. Every d_obs figure that gets a scale is emitted twice (`_linear.png`, `_log.png`).

**Tech Stack:** matplotlib (Agg), `matplotlib.colors.SymLogNorm`, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-dobs-trajectory-and-logscale-design.md`

## Global Constraints

- Run from the repo root. Tests live in `packages/physics-informed-flow-map/tests/`, run with `uv run pytest`.
- mypy is broken on a clean HEAD (21 pre-existing errors, torch-stub friction) and blocks every Python commit; commit with `SKIP=mypy git commit ...`. Keep your own new code mypy-clean regardless. ruff + pytest stay active.
- Never `git add -A`; stage the exact files listed.
- Seismic is signed: log scale means `SymLogNorm`, never `LogNorm`. `linthresh = max(vabs * 1e-2, 1e-12)`, computed from one global `vabs` (99th percentile of `|data|`) shared across all panels for comparability.
- `scale` is exactly `"linear" | "log"`; reject anything else with `ValueError`.
- Velocity maps stay viridis on a positive linear scale — the `scale` switch never touches them.
- Frames reaching the grid are normalized `[-1,1]` velocity; the FWI path delivers them channel-less as `(n_frames, H, W)`.

---

### Task 1: Shared seismic-imshow helper + `scale` on `_plot_seismic`

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py` (imports; new `_seismic_imshow`; `_plot_seismic` at 271-291)
- Test: `packages/physics-informed-flow-map/tests/test_dobs_viz.py`

**Interfaces:**
- Produces: `_seismic_imshow(ax, gather, *, scale, vabs, cmap="RdBu_r") -> AxesImage` and `_plot_seismic(d_obs, gidx, out_png, *, scale="linear") -> None`.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_dobs_viz.py`:

```python
"""d_obs plotting: linear/symlog scale switch and the Tweedie-trajectory grid."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import torch

from physics_informed_flow_map.inversion.single_target import _plot_seismic


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_plot_seismic_writes_png(tmp_path: Path, scale: str) -> None:
    d_obs = torch.randn(5, 70, 1001)  # (n_src, n_rec, nt)
    out = tmp_path / f"d_obs_{scale}.png"
    _plot_seismic(d_obs, gidx=42, out_png=out, scale=scale)
    assert out.exists() and out.stat().st_size > 0


def test_plot_seismic_rejects_bad_scale(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scale"):
        _plot_seismic(torch.randn(2, 8, 16), gidx=0, out_png=tmp_path / "x.png", scale="db")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -v`
Expected: FAIL — `_plot_seismic() got an unexpected keyword argument 'scale'`.

- [ ] **Step 3: Add the `SymLogNorm` import**

In `single_target.py`, under the existing `import matplotlib` block (after `import matplotlib.pyplot as plt`, line 17), add:

```python
from matplotlib.colors import SymLogNorm
from matplotlib.image import AxesImage
```

- [ ] **Step 4: Add the `_seismic_imshow` helper**

Insert immediately above `_plot_seismic` (before line 271):

```python
def _seismic_imshow(
    ax: plt.Axes, gather: np.ndarray, *, scale: str, vabs: float, cmap: str = "RdBu_r"
) -> AxesImage:
    """imshow one ``(n_receivers, nt)`` gather with time on the vertical axis, on a linear or
    symmetric-log amplitude scale.

    ``scale="linear"`` is the plain symmetric ``±vabs`` diverging map. ``scale="log"`` swaps in a
    ``SymLogNorm`` (a ±``linthresh`` linear band around zero, log beyond) so low-amplitude coda is
    lifted without discarding sign. ``vabs`` is a shared symmetric colour limit; pass one global
    value across panels so they stay comparable."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' | 'log', got {scale!r}")
    data = gather.T  # (nt, n_receivers): time down
    if scale == "log":
        linthresh = max(vabs * 1e-2, 1e-12)
        norm = SymLogNorm(linthresh=linthresh, vmin=-vabs, vmax=vabs)
        return ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    return ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)
```

- [ ] **Step 5: Route `_plot_seismic` through the helper and add `scale`**

Replace the whole `_plot_seismic` function (currently lines 271-291) with:

```python
def _plot_seismic(
    d_obs: Tensor, gidx: int, out_png: Path, *, scale: str = "linear"
) -> None:
    """Shot gathers of the observed seismic ``d_obs`` (n_sources, n_receivers, nt) — the input the
    velocity is inverted from. One panel per source: time (down) x receiver, shared symmetric scale.
    ``scale`` selects linear or symmetric-log amplitude (:func:`_seismic_imshow`)."""
    d = d_obs.detach().cpu().numpy()
    n_src = d.shape[0]
    vabs = float(np.percentile(np.abs(d), 99)) or 1.0
    fig, axes = plt.subplots(1, n_src, figsize=(2.2 * n_src, 3.6), squeeze=False)
    for s in range(n_src):
        ax = axes[0, s]
        im = _seismic_imshow(ax, d[s], scale=scale, vabs=vabs)
        ax.set_title(f"source {s + 1}", fontsize=9)
        ax.set_xlabel("receiver", fontsize=8)
        ax.set_ylabel("time sample" if s == 0 else "", fontsize=8)
        if s > 0:
            ax.set_yticklabels([])
    tag = " · log" if scale == "log" else ""
    fig.suptitle(f"observed seismic d_obs · val map {gidx}{tag}", fontsize=10)
    label = "amplitude (symlog)" if scale == "log" else "amplitude"
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, label=label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py \
        packages/physics-informed-flow-map/tests/test_dobs_viz.py
SKIP=mypy git commit -m "feat(0004): symmetric-log scale switch on the seismic plotter

Signed seismic swamps the low-amplitude coda on a linear scale; SymLogNorm
lifts it. Shared _seismic_imshow helper, scale arg on _plot_seismic."
```

---

### Task 2: `plot_dobs_trajectory` grid

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py` (new `plot_dobs_trajectory`)
- Test: `packages/physics-informed-flow-map/tests/test_dobs_viz.py`

**Interfaces:**
- Consumes: `_seismic_imshow` (Task 1), `to_mps_native` (already imported at line 27).
- Produces: `plot_dobs_trajectory(v_true, frames_norm, frame_steps, d_obs_true, forward_fn, out_png, *, scale, title, total_steps, map_label="Tweedie") -> None`.

- [ ] **Step 1: Write the failing test**

Append to `packages/physics-informed-flow-map/tests/test_dobs_viz.py`:

```python
from physics_informed_flow_map.inversion.single_target import plot_dobs_trajectory


def _stub_forward(n_src: int, n_rec: int, nt: int):
    # frames_norm (n_frames,1,res,res) -> (n_frames, n_src, n_rec, nt)
    def fwd(frames: torch.Tensor) -> torch.Tensor:
        return torch.randn(frames.shape[0], n_src, n_rec, nt)
    return fwd


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_plot_dobs_trajectory_writes_png(tmp_path: Path, scale: str) -> None:
    n_frames, n_src, n_rec, nt = 2, 5, 12, 40
    v_true = torch.rand(70, 70) * 1000 + 1500
    frames_norm = torch.rand(n_frames, 1, 16, 16) * 2 - 1
    d_obs_true = torch.randn(n_src, n_rec, nt)
    out = tmp_path / f"traj_{scale}.png"
    plot_dobs_trajectory(
        v_true, frames_norm, [0, 3], d_obs_true, _stub_forward(n_src, n_rec, nt),
        out, scale=scale, title="demo", total_steps=4,
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_dobs_trajectory_panel_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import physics_informed_flow_map.inversion.single_target as st

    seen: dict[str, tuple[int, int]] = {}
    real = st.plt.subplots

    def spy(nrows: int, ncols: int, **kw: object):
        seen["shape"] = (nrows, ncols)
        return real(nrows, ncols, **kw)

    monkeypatch.setattr(st.plt, "subplots", spy)
    plot_dobs_trajectory(
        torch.rand(70, 70), torch.rand(2, 1, 16, 16), [0, 3],
        torch.randn(5, 12, 40), _stub_forward(5, 12, 40),
        tmp_path / "t.png", scale="linear", title="d", total_steps=4,
    )
    assert seen["shape"] == (1 + 5, 1 + 2)  # (1 + n_src) rows, (1 + n_frames) cols
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -k trajectory -v`
Expected: FAIL — `cannot import name 'plot_dobs_trajectory'`.

- [ ] **Step 3: Implement `plot_dobs_trajectory`**

Insert into `single_target.py` immediately after `plot_dobs_compare` (after its `plt.close(fig)`, before `_plot_seismic`):

```python
def plot_dobs_trajectory(
    v_true: Tensor,
    frames_norm: Tensor,
    frame_steps: list[int],
    d_obs_true: Tensor,
    forward_fn: Callable[[Tensor], Tensor],
    out_png: Path,
    *,
    scale: str,
    title: str,
    total_steps: int,
    map_label: str = "Tweedie",
) -> None:
    """Data-space trajectory grid for sample 0. Top row: the true velocity + each captured
    ``frames_norm`` prediction (viridis, shared scale, titled with its trajectory step). Rows below:
    one per seismic source — column 0 is the true velocity's ``d_obs``, columns 1.. are the d_obs
    re-simulated from each prediction via ``forward_fn`` (shared symmetric scale, linear or symlog).

    ``frames_norm`` is ``(n_frames, 1, res, res)`` normalized [-1,1]; ``d_obs_true`` is
    ``(n_src, n_rec, nt)``. ``map_label`` names the prediction ("Tweedie" for prior methods,
    "iterate" for the FWI baselines). The ``scale`` switch applies to the seismic rows only."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' | 'log', got {scale!r}")
    with torch.no_grad():
        d_frames = forward_fn(frames_norm).detach().cpu().numpy()  # (n_frames, n_src, n_rec, nt)
    v_hat = to_mps_native(frames_norm).detach().cpu().numpy()      # (n_frames, 70, 70) m/s
    vt = v_true.detach().cpu().numpy()
    dt = d_obs_true.detach().cpu().numpy()                          # (n_src, n_rec, nt)
    n_frames = int(frames_norm.shape[0])
    n_src = int(dt.shape[0])
    n_cols = 1 + n_frames
    # One symmetric scale across the true column and every frame, so panels are comparable.
    vabs = float(np.percentile(np.abs(np.concatenate([dt[None], d_frames], axis=0)), 99)) or 1.0
    vlo, vhi = float(vt.min()), float(vt.max())

    fig, axes = plt.subplots(
        1 + n_src, n_cols, figsize=(2.1 * n_cols, 2.1 * (1 + n_src)), squeeze=False
    )
    # Row 0 — velocity maps (viridis; scale switch does NOT apply here).
    vimg = axes[0, 0].imshow(vt, cmap="viridis", vmin=vlo, vmax=vhi)
    axes[0, 0].set_title("true v", fontsize=9)
    for j in range(n_frames):
        axes[0, 1 + j].imshow(v_hat[j], cmap="viridis", vmin=vlo, vmax=vhi)
        axes[0, 1 + j].set_title(f"{map_label}\nstep {frame_steps[j]}", fontsize=9)
    for c in range(n_cols):
        axes[0, c].axis("off")
    fig.colorbar(vimg, ax=axes[0, n_cols - 1], fraction=0.046, label="m/s")

    # Rows 1..n_src — shot gathers, column 0 = true, columns 1.. = frames.
    im = None
    for s in range(n_src):
        r = 1 + s
        im = _seismic_imshow(axes[r, 0], dt[s], scale=scale, vabs=vabs)
        axes[r, 0].set_ylabel(f"source {s + 1}\ntime", fontsize=8)
        if s == 0:
            axes[r, 0].set_title("true d_obs", fontsize=9)
        for j in range(n_frames):
            _seismic_imshow(axes[r, 1 + j], d_frames[j, s], scale=scale, vabs=vabs)
            if s == 0:
                axes[r, 1 + j].set_title(f"step {frame_steps[j]}", fontsize=9)
            axes[r, 1 + j].set_yticklabels([])
        for c in range(n_cols):
            axes[r, c].set_xticklabels([])
    label = "amplitude (symlog)" if scale == "log" else "amplitude"
    fig.colorbar(im, ax=axes[1:, n_cols - 1].ravel().tolist(), fraction=0.046, label=label)
    tag = " · log" if scale == "log" else ""
    fig.suptitle(
        f"{title}{tag}\nd_obs from {map_label} predictions over {total_steps} steps", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py -k trajectory -v`
Expected: 3 passed (2 parametrized write + 1 panel-count).

- [ ] **Step 5: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py \
        packages/physics-informed-flow-map/tests/test_dobs_viz.py
SKIP=mypy git commit -m "feat(0004): plot_dobs_trajectory data-space grid

Sample-0 Tweedie predictions along the guided pass -> their simulated d_obs
per source, against the true-velocity column."
```

---

### Task 3: `capture` hook in `Run.make_step_saver`

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py` (`make_step_saver`, 271-332)
- Test: `packages/physics-informed-flow-map/tests/test_run.py`

**Interfaces:**
- Produces: `make_step_saver(..., capture: dict[str, Any] | None = None)`. When given, at the final checkpoint it sets `capture["frames"]` (`(n_frames, B, ...)` estimate stack), `capture["steps"]` (`list[int]`), `capture["total_steps"]` (`int`).

- [ ] **Step 1: Write the failing test**

Append to `packages/physics-informed-flow-map/tests/test_run.py`:

```python
def test_make_step_saver_capture(tmp_path: Path) -> None:
    from typing import Any

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = start_run("test_exp", tmp_path, {"lr": 0.1})

    def viz(frames: torch.Tensor, path: Path) -> None:
        fig, ax = plt.subplots()
        ax.imshow(frames[0, 0].numpy())
        fig.savefig(path)
        plt.close(fig)

    cap: dict[str, Any] = {}
    cb = run.make_step_saver("k", viz, total_steps=3, n_frames=3, capture=cap)
    for step in range(3):
        cb(step, torch.rand(2, 1, 4, 4), data_fidelity=1.0 / (step + 1))

    assert cap["steps"] == [0, 1, 2]
    assert cap["frames"].shape == (3, 2, 1, 4, 4)  # (n_frames, B, C, H, W)
    assert cap["total_steps"] == 3

    # capture=None leaves behavior unchanged (renders, does not raise).
    cb2 = run.make_step_saver("k2", viz, total_steps=2, n_frames=2, capture=None)
    cb2(0, torch.rand(2, 1, 4, 4))
    cb2(1, torch.rand(2, 1, 4, 4))
    run.finish()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `WANDB_MODE=disabled uv run pytest packages/physics-informed-flow-map/tests/test_run.py::test_make_step_saver_capture -v`
Expected: FAIL — `make_step_saver() got an unexpected keyword argument 'capture'`.

- [ ] **Step 3: Add the `capture` parameter**

In `make_step_saver` (run.py line 271), add `capture` to the signature after `n_frames`:

```python
    def make_step_saver(
        self,
        key: str,
        traj_viz_fn: Callable[[torch.Tensor, Path], None],
        *,
        total_steps: int,
        n_frames: int = 3,
        capture: dict[str, Any] | None = None,
    ) -> Callable[..., None]:
```

- [ ] **Step 4: Populate `capture` at the final checkpoint**

In the `if step == last_checkpoint:` block (currently lines 320-330), replace:

```python
            if step == last_checkpoint:
                frames = torch.stack([est_frames[s] for s in sorted(est_frames)])
                caption = f"{len(est_frames)} frames over {total_steps} steps"
```

with:

```python
            if step == last_checkpoint:
                est_stack = torch.stack([est_frames[s] for s in sorted(est_frames)])
                if capture is not None:
                    capture["frames"] = est_stack
                    capture["steps"] = sorted(est_frames)
                    capture["total_steps"] = total_steps
                frames = est_stack
                caption = f"{len(est_frames)} frames over {total_steps} steps"
```

(The rest of the block — the `if xt_frames:` interleave, `traj_viz_fn`, `log_image` — is unchanged. `capture` holds the pre-interleave estimate stack: Tweedie/iterate frames only.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `WANDB_MODE=disabled uv run pytest packages/physics-informed-flow-map/tests/test_run.py::test_make_step_saver_capture -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/experiment/run.py \
        packages/physics-informed-flow-map/tests/test_run.py
SKIP=mypy git commit -m "feat(harness): opt-in capture dict on make_step_saver

Keeps the estimate frames (Tweedie/iterate) it already collects, so callers
can post-process the trajectory after inversion instead of only rendering it."
```

---

### Task 4: Wire the grid + dual-scale d_obs into `invert_and_report` and `run.py`

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py` (`invert_and_report` signature 67-86, obs block 121-122, new grid block after 186)
- Modify: `experiments/0004_inversion/run.py` (traj_cap dict; 6 `make_step_saver` call sites; obs path 637; `invert_and_report` call 653; d_obs wandb logging ~675)

**Interfaces:**
- Consumes: `plot_dobs_trajectory` (Task 2), `_plot_seismic(scale=...)` (Task 1), `make_step_saver(capture=...)` (Task 3), `seismic_forward` (imported at single_target.py:27).
- Produces (files per guided run): `d_obs_linear.png`, `d_obs_log.png`, `<method>_g<gs>_dobs_traj_linear.png`, `<method>_g<gs>_dobs_traj_log.png`.

**Context:** `invert_and_report` computes `gs = guidance if method_name != "unguided" else 0.0` at line 124, loads `v_true`/`d_obs`, and holds `cmp_label`/`label`. `make_step_saver` is only built on the guided pass (`guidance_strength != 0.0`), so the unguided pass never fills `traj_cap`, and `unguided` methods leave it empty → grid skipped.

- [ ] **Step 1: Add params to `invert_and_report`**

First widen the typing import at single_target.py line 12 from `from typing import Callable` to:

```python
from typing import Any, Callable
```

Then in the signature (single_target.py 67-86), after `obs_cfg: ObservationConfig | None = None,` add
(`Any` value type so the captured tensors index cleanly under mypy):

```python
    traj_capture: dict[str, Any] | None = None,
    dobs_scales: tuple[str, ...] = ("linear",),
    forward_fn: Callable[[Tensor], Tensor] = seismic_forward,
```

- [ ] **Step 2: Emit the observed figure per scale**

Replace the observed-seismic block (currently lines 121-122):

```python
    if out_obs_png is not None:
        _plot_seismic(d_obs, gidx, out_obs_png)
```

with:

```python
    if out_obs_png is not None:
        base = out_obs_png.with_suffix("")  # accept "d_obs" or "d_obs.png"
        for sc in dobs_scales:
            _plot_seismic(d_obs, gidx, base.with_name(f"{base.name}_{sc}.png"), scale=sc)
```

- [ ] **Step 3: Render the trajectory grid after the fit-comparison block**

Immediately after the `out_dobs_cmp_png` block (after its `np.savez(...)` closes, currently ~line 186) and before the `summary = {` dict, insert:

```python
    if traj_capture and traj_capture.get("frames") is not None:
        frames0 = traj_capture["frames"][:, 0]  # sample 0: (n_frames, C?, H, W)
        if frames0.ndim == 3:  # FWI native path delivers (n_frames, H, W)
            frames0 = frames0[:, None]
        frames0 = frames0.to(device)
        map_label = (
            "iterate" if method_name in ("classical_fwi", "realistic_fwi") else "Tweedie"
        )
        traj_base = out_png.with_name(f"{method_name}_g{gs:.2g}_dobs_traj")
        for sc in dobs_scales:
            plot_dobs_trajectory(
                v_true,
                frames0,
                list(traj_capture["steps"]),
                d_obs,
                forward_fn,
                traj_base.with_name(f"{traj_base.name}_{sc}.png"),
                scale=sc,
                title=f"{cmp_label or method_name} · {label} · sample 0",
                total_steps=int(traj_capture["total_steps"]),
                map_label=map_label,
            )
```

- [ ] **Step 4: Create the shared capture dict in run.py**

In `experiments/0004_inversion/run.py`, next to the `solves = {"n": 0}` dict (line 326-328), add:

```python
    traj_cap: dict[str, object] = {}  # sample-0 trajectory frames, filled by the guided step saver
```

- [ ] **Step 5: Pass `capture=traj_cap` to every make_step_saver call**

Add `capture=traj_cap,` to each of the six `run.make_step_saver(...)` calls (the `n_frames=cfg.n_frames,` line is the last kwarg in each — add `capture` after it). The call sites are near lines 364, 431, 493, 528, 569, 597. Each becomes:

```python
                step_cb = run.make_step_saver(
                    f"...",
                    viz_traj,
                    total_steps=...,
                    n_frames=cfg.n_frames,
                    capture=traj_cap,
                )
```

(For the mfm branch at ~431 the variable is `traj_saver`, not `step_cb` — add `capture=traj_cap` there too.)

- [ ] **Step 6: Rename the obs output and pass the new params**

In run.py, change the obs output path (line 637) from:

```python
    obs_out = run.ckpt_dir.parent / "d_obs.png"
```

to:

```python
    obs_out = run.ckpt_dir.parent / "d_obs"  # base; invert_and_report writes _linear/_log.png
```

Then in the `invert_and_report(...)` call (line 653), after `obs_cfg=cfg.obs,` add:

```python
        traj_capture=traj_cap,
        dobs_scales=("linear", "log"),
```

- [ ] **Step 7: Replace the d_obs wandb logging with the dual-scale + grid logging**

In run.py, replace the single line (currently ~675):

```python
    run.log_image("d_obs", obs_out, caption=f"observed seismic · {caption}")
```

with:

```python
    gs_log = cfg.method.guidance_strength if cfg.method.name != "unguided" else 0.0
    for sc in ("linear", "log"):
        dobs_png = run.ckpt_dir.parent / f"d_obs_{sc}.png"
        if dobs_png.exists():
            run.log_image(f"d_obs_{sc}", dobs_png, caption=f"observed seismic ({sc}) · {caption}")
        traj_png = run.ckpt_dir.parent / f"{cfg.method.name}_g{gs_log:.2g}_dobs_traj_{sc}.png"
        if traj_png.exists():
            run.log_image(
                f"traj/dobs_{sc}", traj_png, caption=f"d_obs trajectory ({sc}) · {caption}"
            )
```

- [ ] **Step 8: Verify no `d_obs.png` literal remains**

Run: `grep -rn "d_obs.png" experiments/ packages/`
Expected: no output (exit 1).

- [ ] **Step 9: Full suite still green**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_dobs_viz.py packages/physics-informed-flow-map/tests/test_run.py -q`
Expected: all pass.

- [ ] **Step 10: End-to-end smoke run**

Run:

```bash
WANDB_MODE=disabled uv run python experiments/0004_inversion/run.py \
  prior=diffusion method=dps method.misfit=ot method.guidance_strength=2 \
  target=flatvel_a_legacy_6044 steps=6 n_samples=2 n_frames=3
```

Expected: exits 0. In the newest `runs/0004_inversion/diffusion_dps_*/` dir:

```bash
RD=$(ls -dt runs/0004_inversion/diffusion_dps_* | head -1)
ls "$RD"/d_obs_linear.png "$RD"/d_obs_log.png \
   "$RD"/dps_g2_dobs_traj_linear.png "$RD"/dps_g2_dobs_traj_log.png
```

All four exist. (Uses the real checkpoint pinned in `conf/prior/diffusion.yaml`; steps=6 keeps it quick.)

- [ ] **Step 11: Confirm grid column 0 matches the observed d_obs**

Run:

```bash
uv run python -c "
import hashlib, numpy as np
# Sanity: the grid's column-0 d_obs is the same array as the observed figure's source.
# Both come from d_obs; this just asserts the run produced non-trivial seismic.
from PIL import Image
RD='$RD'
for f in ['d_obs_linear.png','dps_g2_dobs_traj_linear.png']:
    im=np.asarray(Image.open(f'{RD}/{f}')); print(f, im.shape, im.std()>0)
"
```

Expected: both print `True` (non-blank figures). Visually confirm the log panels lift the coda relative to linear.

- [ ] **Step 12: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/single_target.py \
        experiments/0004_inversion/run.py
SKIP=mypy git commit -m "feat(0004): emit d_obs in log+linear and the Tweedie-d_obs grid

invert_and_report writes d_obs_{linear,log}.png and, from the captured
sample-0 frames, the per-source data-space trajectory grid in both scales."
```

---

## Self-review notes

- **Spec coverage:** shared helper + `_plot_seismic` scale (Task 1) ✓; `plot_dobs_trajectory` incl. velocity-row-linear rule and shared `vabs` (Task 2) ✓; capture hook (Task 3) ✓; wiring both d_obs scales + grid + FWI `map_label` + `d_obs.png` rename + skip-when-empty (Task 4) ✓. `plot_dobs_compare` untouched ✓.
- **Cleanup (optional, per standing preference):** `tests/test_dobs_viz.py` and the `test_make_step_saver_capture` case are TDD scaffolding and may be deleted after Task 4 verifies end-to-end — ask before removing.
