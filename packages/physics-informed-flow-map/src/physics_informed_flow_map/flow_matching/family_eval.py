"""Per-family evaluation for unconditional OpenFWI priors.

The priors are unconditional, so generated samples carry no family label — samples cannot
be stratified by family. Per-family observability instead measures:

* the model's val loss on each family's held-out maps (is any family under-fit?), via
  :func:`per_family_val_loss`;
* the energy distance between one shared pool of generated samples and each family's
  held-out maps (does the generated distribution cover each family's region?), via
  :func:`per_family_energy_distance`;
* a one-time grid of real held-out maps per family (:func:`family_reference_grid`) — the
  visual reference the per-epoch sample grids are compared against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Generated-sample pool size for the final per-family energy distance.
N_ENERGY_SAMPLES = 512
# Held-out maps drawn per family for the energy-distance reference.
N_ENERGY_REAL = 512


def per_family_val_loss(
    loss_on_batch: Callable[[Tensor, Tensor], float],
    val_by_family: dict[str, Dataset],
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    """Sample-weighted global val loss + per-family means.

    ``loss_on_batch(x, labels)`` returns the mean loss of one batch; weighting by batch
    size makes each family's number an exact per-sample mean (unlike mean-of-batch-means).
    """
    fam_losses: dict[str, float] = {}
    total, n_total = 0.0, 0
    for family, ds in val_by_family.items():
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
        fam_total, fam_n = 0.0, 0
        for xb, lb in loader:
            bs = int(xb.shape[0])
            fam_total += loss_on_batch(xb, lb) * bs
            fam_n += bs
        fam_losses[family] = fam_total / max(fam_n, 1)
        total += fam_total
        n_total += fam_n
    return total / max(n_total, 1), fam_losses


def energy_distance(a: Tensor, b: Tensor) -> float:
    """Energy distance (V-statistic) between two sample sets, flattened to vectors.

    ``E = 2·E‖a−b‖ − E‖a−a′‖ − E‖b−b′‖ ≥ 0``, 0 iff the distributions match. The
    V-statistic keeps the self-distance diagonal — a small negative bias identical across
    families, so per-family comparisons are unaffected.
    """
    fa = a.flatten(1).float()
    fb = b.flatten(1).float()
    ab = torch.cdist(fa, fb).mean()
    aa = torch.cdist(fa, fa).mean()
    bb = torch.cdist(fb, fb).mean()
    return float(2 * ab - aa - bb)


def _stack_family(ds: Dataset, max_n: int, seed: int) -> Tensor:
    n = len(ds)  # type: ignore[arg-type]  # map-style dataset
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_n]
    return torch.stack([ds[int(i)][0] for i in idx])


def per_family_energy_distance(
    samples: Tensor,
    val_by_family: dict[str, Dataset],
    *,
    max_real: int = N_ENERGY_REAL,
    seed: int = 0,
) -> dict[str, float]:
    """Energy distance of one shared generated pool vs each family's held-out maps."""
    out: dict[str, float] = {}
    for family, ds in val_by_family.items():
        real = _stack_family(ds, max_real, seed).to(samples.device)
        out[family] = energy_distance(samples, real)
    return out


def family_reference_grid(
    val_by_family: dict[str, Dataset], path: Path, *, n_per_family: int = 8
) -> None:
    """One row of real held-out maps per family, family-labeled, on the sample-grid scale."""
    families = list(val_by_family)
    nrows, ncols = len(families), n_per_family
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols, nrows * 1.15), squeeze=False
    )
    for r, family in enumerate(families):
        ds = val_by_family[family]
        for c in range(ncols):
            ax = axes[r, c]
            if c < len(ds):  # type: ignore[arg-type]
                x, _ = ds[c]
                ax.imshow(x[0].numpy(), cmap="viridis", vmin=-1, vmax=1)
            ax.axis("off")
        axes[r, 0].set_ylabel(family, rotation=0, ha="right", va="center", fontsize=6)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
