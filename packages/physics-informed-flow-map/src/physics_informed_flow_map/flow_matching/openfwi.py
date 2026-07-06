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
        # Parallel (file, row) provenance for each sample, so callers can re-load a sample at its
        # native resolution from disk (held_out_targets uses this for the inversion target maps).
        self.index: list[tuple[Path, int]] = []
        self.family_names: list[str] = list(families)
        # First pass: headers only (mmap), to size the single contiguous allocation.
        entries: list[tuple[int, Path, int]] = []  # (family_id, file, n_rows)
        total = 0
        for fid, family in enumerate(families):
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
                n_rows = int(np.load(f, mmap_mode="r").shape[0])
                entries.append((fid, f, n_rows))
                total += n_rows
        # Second pass: fill the pre-allocated array one file at a time, so peak transient
        # memory is dataset + one file (not 2x dataset as stack-of-rows would be). Workers
        # forked after construction share it copy-on-write.
        self._data = np.empty((total, 1, NATIVE, NATIVE), dtype=np.float32)
        self.family_ids = np.empty(total, dtype=np.int8)
        pos = 0
        for fid, f, n_rows in entries:
            arr = np.load(f)  # (N, 1, 70, 70) or (N, 70, 70)
            self._data[pos : pos + n_rows] = arr.reshape(n_rows, 1, NATIVE, NATIVE)
            self.family_ids[pos : pos + n_rows] = fid
            self.index.extend((f, i) for i in range(n_rows))
            pos += n_rows
        # Families are loaded in order, so each occupies one contiguous global-index slice.
        self.family_slices: list[slice] = []
        start = 0
        for fid in range(len(families)):
            n_fam = int((self.family_ids == fid).sum())
            self.family_slices.append(slice(start, start + n_fam))
            start += n_fam

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


class RandomHFlip(Dataset):
    """p=0.5 left-right flip augmentation for velocity maps (train split only).

    A mirrored geological cross-section is a physically valid velocity model (unlike a
    vertical flip — velocity increases with depth), and the prior is unconditional, so
    this doubles effective data diversity for free.
    """

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]  # map-style base

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        x, label = self.base[idx]
        if bool(torch.rand(()) < 0.5):
            x = torch.flip(x, dims=[-1])
        return x, label


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
