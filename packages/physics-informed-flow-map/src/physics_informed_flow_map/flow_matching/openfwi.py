"""OpenFWI velocity-map dataset + a colormap visualizer.

Velocity maps are normalised from [1500, 4500] m/s to [-1, 1] and optionally
resized from the native 70x70 to a square training resolution. The dataset is
unconditional: every sample is paired with a dummy label 0.

Data is pre-loaded into RAM on first construction to avoid per-sample NFS I/O
during training (the full FlatVel_A split is ~500 MB, trivial vs. available RAM).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

VMIN = 1500.0
VMAX = 4500.0
NATIVE = 70


class OpenFWIVelocityDataset(Dataset):
    """OpenFWI velocity maps, pre-loaded into RAM for fast multi-worker access."""

    def __init__(self, root: Path, families: list[str], resolution: int = 64) -> None:
        self.resolution = resolution
        rows: list[np.ndarray] = []
        # Parallel (file, row) provenance for each sample, so callers can re-load a sample at its
        # native resolution from disk (held_out_targets uses this for the inversion target maps).
        self.index: list[tuple[Path, int]] = []
        for family in families:
            family_dir = root / family
            files = sorted(family_dir.glob("model/*.npy")) + sorted(
                family_dir.glob("vel*.npy")
            )
            if not files:
                raise FileNotFoundError(
                    f"No OpenFWI velocity files under {family_dir} "
                    f"(expected <family>/model/*.npy or <family>/vel*.npy). "
                    f"Download from the 'ashynf/OpenFWI' HuggingFace dataset."
                )
            for f in files:
                arr = np.load(f)  # load full file into RAM once
                for i in range(arr.shape[0]):
                    rows.append(arr[i])
                    self.index.append((f, i))
        # Stack into one contiguous array so workers share it via fork (copy-on-write).
        self._data = np.stack(rows, axis=0)  # (N, 1, 70, 70) float32

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        x = torch.from_numpy(self._data[idx].copy()).float()
        x = ((x - VMIN) / (VMAX - VMIN) * 2.0 - 1.0).clamp(-1.0, 1.0)
        if self.resolution != NATIVE:
            x = F.interpolate(
                x[None],
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0]
        return x, 0


def viz_velocity(samples: Tensor, path: Path, *, ncols: int = 8) -> None:
    """Grid of velocity maps with a perceptual colormap (samples are in [-1, 1])."""
    s = samples.detach().cpu().clamp(-1, 1)
    n = min(ncols * 8, len(s))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(s[i, 0].numpy(), cmap="viridis", vmin=-1, vmax=1)
        ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
