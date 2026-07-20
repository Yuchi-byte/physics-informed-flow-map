"""Per-step MC-posterior trajectory grids for the flow-map (MFM) steered inverter.

For each posterior particle, lay out a tall static grid: one row per sampler step, columns
``[ noisy x_t | MC 1 | MC 2 | ... | MC K ]`` — the noisy sampler state and the K Monte-Carlo
posterior draws ``x1 ~ p(x1 | x_t)`` the iwae/sne estimator actually used that step (surfaced
via ``steering.get_conditional_drift_fn``'s ``mc_x1_data``). Shows how the posterior draws
sharpen from noise into the recovered velocity as ``t: 0 -> 1``.

Rendered as tiled RGB canvases (PIL) rather than a matplotlib subplot grid, so 200 steps x 30
particles stays fast and memory-light. ``x_t`` uses a symmetric diverging map (it is a noisy
latent); the velocity draws use ``viridis`` on the fixed ``[-1, 1]`` prior scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from PIL import Image, ImageDraw

CELL = 70  # per-image cell size (px); arrays are resized to this
GAP = 3  # white gutter between cells
TITLE = 16  # top title-band height
HEADER = 16  # column-header strip height (below the title band)
LABELW = 40  # left step-label column width
TOP = TITLE + HEADER  # total top margin before the first row


def _rgb(arr: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    """Map ``arr`` (H, W) through ``cmap`` to a (H, W, 3) uint8 image."""
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return (mpl.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)


def _cell_img(arr: np.ndarray, cmap: str, vmin: float, vmax: float) -> Image.Image:
    im = Image.fromarray(_rgb(arr, cmap, vmin, vmax))
    return (
        im.resize((CELL, CELL), Image.Resampling.NEAREST)
        if im.size != (CELL, CELL)
        else im
    )


def render_particle_grid(
    xt: np.ndarray, mc: np.ndarray, out_png: Path, *, title: str
) -> None:
    """One particle. ``xt`` (steps, H, W); ``mc`` (steps, K, H, W) in ``[-1, 1]``."""
    steps = xt.shape[0]
    k = mc.shape[1]
    ncols = 1 + k

    W = LABELW + ncols * (CELL + GAP) + GAP
    H = TOP + steps * (CELL + GAP) + GAP
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((4, 3), title, fill="black")  # full title on its own band
    headers = ["x_t (noisy)"] + [f"MC {j + 1}" for j in range(k)]
    for c, name in enumerate(headers):
        x0 = LABELW + c * (CELL + GAP) + GAP
        draw.text((x0 + 2, TITLE + 3), name, fill="black")

    stride = max(1, steps // 25)  # a step index label roughly every 25 rows
    for s in range(steps):
        y0 = TOP + s * (CELL + GAP) + GAP
        if s % stride == 0 or s == steps - 1:
            draw.text((2, y0 + CELL // 2 - 4), f"t{s}", fill="black")
        # column 0: noisy sampler state — per-step symmetric scale (its magnitude grows with
        # the SDE noise as t->0), so each step's structure stays visible regardless of scale.
        xt_abs = float(np.percentile(np.abs(xt[s]), 99)) or 1.0
        canvas.paste(_cell_img(xt[s], "RdBu_r", -xt_abs, xt_abs), (LABELW + GAP, y0))
        # columns 1..K: the MC posterior velocity draws (fixed prior scale)
        for j in range(k):
            x0 = LABELW + (1 + j) * (CELL + GAP) + GAP
            canvas.paste(_cell_img(mc[s, j], "viridis", -1.0, 1.0), (x0, y0))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def render_mc_posterior_grids(
    xt: np.ndarray, mc: np.ndarray, out_dir: Path, *, title_prefix: str
) -> list[Path]:
    """All particles. ``xt`` (steps, n, H, W); ``mc`` (steps, n, K, H, W) in ``[-1, 1]``.

    Writes ``out_dir/particle_XX.png`` per particle and returns the paths."""
    out_dir = Path(out_dir)
    steps, n = xt.shape[0], xt.shape[1]
    paths: list[Path] = []
    for p in range(n):
        out_png = out_dir / f"particle_{p:02d}.png"
        render_particle_grid(
            xt[:, p],
            mc[:, p],
            out_png,
            title=f"{title_prefix} · particle {p} · {steps} steps",
        )
        paths.append(out_png)
    return paths
