"""Shared normalize/resize bridge between prior samples and the physical forward operator,
plus the held-out target loader. One copy of this geometry so every module and the evaluator
agree on units and acquisition.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..flow_matching.openfwi import VMAX, VMIN
from ..physics.forward import simulate

NATIVE = 70


def to_mps_native(v_norm: Tensor, native: int = NATIVE) -> Tensor:
    """``(B,1,res,res)`` in ``[-1,1]`` -> ``(B,native,native)`` velocity in m/s (clamped)."""
    v = F.interpolate(v_norm, size=native, mode="bilinear", align_corners=False).clamp(
        -1.0, 1.0
    )
    return ((v + 1.0) / 2.0 * (VMAX - VMIN) + VMIN)[:, 0]


def seismic_forward(v_norm: Tensor) -> Tensor:
    """Differentiable map from normalized prior samples ``(B,1,res,res)`` to stacked seismic
    data ``(B, n_sources, n_receivers, nt)`` — the guidance/likelihood operator."""
    v_mps = to_mps_native(v_norm)
    return torch.stack([simulate(v_mps[b]) for b in range(v_mps.shape[0])])


def held_out_targets(cfg: OpenFWIDatasetConfig, n: int) -> list[tuple[int, Tensor]]:
    """First ``n`` validation-split maps as ``(global_index, native 70x70 m/s tensor)``.

    Drawn from the same seed-0 split the priors held out of training, so evaluation targets
    are genuinely unseen.
    """
    full, _, val_idx = cfg._split()
    targets: list[tuple[int, Tensor]] = []
    for i in range(min(n, len(val_idx))):
        gidx = val_idx[i]
        path, row = full.index[gidx]
        native = np.ascontiguousarray(np.load(path, mmap_mode="r")[row, 0])
        targets.append((gidx, torch.from_numpy(native).float()))
    return targets
