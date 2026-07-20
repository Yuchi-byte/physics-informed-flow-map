"""Per-family eval helpers: hermetic tests on synthetic tensors/datasets."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from physics_informed_flow_map.flow_matching.family_eval import (
    energy_distance,
    family_reference_grid,
)


def _const_ds(value: float, n: int) -> TensorDataset:
    return TensorDataset(
        torch.full((n, 1, 8, 8), value), torch.zeros(n, dtype=torch.long)
    )


def test_energy_distance_zero_for_identical_sets() -> None:
    a = torch.randn(64, 1, 8, 8)
    assert energy_distance(a, a.clone()) == pytest.approx(0.0, abs=1e-4)


def test_energy_distance_orders_by_shift() -> None:
    torch.manual_seed(0)
    a = torch.randn(128, 1, 8, 8)
    near = torch.randn(128, 1, 8, 8) + 0.1
    far = torch.randn(128, 1, 8, 8) + 2.0
    assert energy_distance(a, near) < energy_distance(a, far)


def test_family_reference_grid_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "ref.png"
    val_by_family = {"A": _const_ds(0.5, 4), "B": _const_ds(-0.5, 4)}
    family_reference_grid(val_by_family, out, n_per_family=8)
    assert out.exists() and out.stat().st_size > 0
