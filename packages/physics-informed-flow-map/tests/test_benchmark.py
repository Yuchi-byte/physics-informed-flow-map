"""Inversion benchmark: hermetic tests on synthetic .npy fixtures."""

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.inversion.benchmark import (
    InversionBenchmark,
    TargetEntry,
    legacy_entry,
    select_targets,
    total_variation,
    write_benchmark,
)


def _write_family(
    root: Path, family: str, layout: str, n_files: int, rows: int
) -> None:
    """Synthetic velocity files with per-row structure so TV varies across rows."""
    if layout == "model":
        out_dir = root / family / "model"
        names = [f"model{i}.npy" for i in range(n_files)]
    else:
        out_dir = root / family
        names = [f"vel{i}.npy" for i in range(n_files)]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(family.encode()))
    for name in names:
        arr = np.empty((rows, 1, 70, 70), dtype=np.float32)
        for r in range(rows):
            # r+2 layers of ALTERNATING velocity, so total variation grows ~linearly
            # with the layer count and TV rank tracks row structure, not noise.
            layers = np.where(np.arange(r + 2) % 2 == 0, 1500.0, 4500.0).astype(
                np.float32
            )
            arr[r, 0] = np.repeat(
                layers, np.diff(np.linspace(0, 70, r + 3).astype(int))
            )[:, None].repeat(70, axis=1)
            arr[r, 0] += rng.normal(0, 1.0, (70, 70)).astype(np.float32)
        np.save(out_dir / name, arr)


@pytest.fixture()
def bench_cfg(tmp_path: Path) -> OpenFWIDatasetConfig:
    # 600 maps/family -> 60 val maps, enough that the 5 percentile windows (9 ranks
    # each at this n) are disjoint, as they are at real scale (3000+ val maps/family).
    _write_family(tmp_path, "FlatVel_A", "model", n_files=4, rows=150)
    _write_family(tmp_path, "FlatFault_A", "flat", n_files=4, rows=150)
    return OpenFWIDatasetConfig(
        data_dir=str(tmp_path),
        families=["FlatFault_A", "FlatVel_A"],
        split_scheme="per_family",
    )


def test_total_variation_orders_by_structure() -> None:
    flat = np.full((1, 70, 70), 2000.0)
    two_layer = np.concatenate(
        [np.full((1, 35, 70), 2000.0), np.full((1, 35, 70), 4000.0)], axis=1
    )
    tv = total_variation(np.concatenate([flat, two_layer]))
    assert tv[0] == 0.0
    assert tv[1] > 0.0


def test_selection_counts_ids_and_core(bench_cfg: OpenFWIDatasetConfig) -> None:
    entries = select_targets(bench_cfg)
    # 2 families x 5 percentiles x 4 draws from 60 val maps per family
    assert len(entries) == 40
    per_family = {
        f: [e for e in entries if e.family == f] for f in ("FlatVel_A", "FlatFault_A")
    }
    for family, fam in per_family.items():
        assert [e.id for e in fam] == [f"{family.lower()}_{k:02d}" for k in range(20)]
        assert sum(e.core for e in fam) == 10  # 2 core draws x 5 percentiles
        assert all(e.has_fault == ("Fault" in family) for e in fam)
    # no duplicate provenance
    assert len({(e.family, e.file, e.row) for e in entries}) == 40


def test_selection_is_deterministic(bench_cfg: OpenFWIDatasetConfig) -> None:
    a = select_targets(bench_cfg)
    b = select_targets(bench_cfg)
    assert [(e.id, e.file, e.row, e.tv) for e in a] == [
        (e.id, e.file, e.row, e.tv) for e in b
    ]


def test_selection_spans_percentiles(bench_cfg: OpenFWIDatasetConfig) -> None:
    entries = [e for e in select_targets(bench_cfg) if e.family == "FlatVel_A"]
    by_pct = {p: [e.tv for e in entries if e.tv_percentile == p] for p in (5, 95)}
    assert max(by_pct[5]) < min(by_pct[95])


def test_selection_draws_only_from_val(bench_cfg: OpenFWIDatasetConfig) -> None:
    _, _, val_idx = bench_cfg._split()
    assert {e.global_index for e in select_targets(bench_cfg)} <= set(val_idx)


def test_legacy_entry_pinned_by_provenance(bench_cfg: OpenFWIDatasetConfig) -> None:
    provenance = {
        "legacy_global_index": 6044,
        "family": "FlatVel_A",
        "file": "model2.npy",
        "row": 7,
    }
    e = legacy_entry(bench_cfg, provenance)
    assert e.id == "flatvel_a_legacy_6044"
    assert e.legacy and e.core
    assert (e.file, e.row) == ("model/model2.npy", 7)
    full, _, val_idx = bench_cfg._split()
    assert full.index[e.global_index][0].name == "model2.npy"
    assert e.in_current_val == (e.global_index in set(val_idx))


def test_write_benchmark_and_loader_roundtrip(
    bench_cfg: OpenFWIDatasetConfig, tmp_path: Path
) -> None:
    out = tmp_path / "bench"
    provenance = {
        "legacy_global_index": 6044,
        "family": "FlatVel_A",
        "file": "model0.npy",
        "row": 3,
    }
    manifest = write_benchmark(bench_cfg, out, legacy_provenance=provenance)
    assert len(manifest["targets"]) == 41
    assert manifest["dataset_fingerprint"] == bench_cfg.fingerprint()

    bench = InversionBenchmark(out)
    assert len(bench) == 41
    assert set(bench.core_ids) < set(bench.ids)
    # manifest <-> file consistency: every target has velocity + preview on disk
    for tid in bench.ids:
        assert (out / "velocity" / f"{tid}.npy").exists()
        assert (out / "previews" / f"{tid}.png").exists()
    for family in ("FlatVel_A", "FlatFault_A"):
        assert (out / "previews" / f"gallery_{family}.png").exists()

    # loader returns the native m/s map recorded by provenance while bulk data exists
    full, _, _ = bench_cfg._split()
    for tid in ["flatvel_a_legacy_6044", bench.core_ids[0]]:
        entry = bench.entry(tid)
        v = bench.velocity(tid)
        assert v.shape == (70, 70) and v.dtype == torch.float32
        src = np.load(
            Path(bench_cfg.data_dir) / entry["family"] / entry["file"], mmap_mode="r"
        )[entry["row"], 0]
        np.testing.assert_array_equal(v.numpy(), src)
        np.testing.assert_array_equal(v.numpy(), full._data[entry["global_index"], 0])


def test_load_target_by_benchmark_id(
    bench_cfg: OpenFWIDatasetConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import physics_informed_flow_map.inversion.single_target as st

    from physics_informed_flow_map.physics.observation import Observation

    out = tmp_path / "bench"
    write_benchmark(bench_cfg, out)
    monkeypatch.setattr(
        st, "observe", lambda v, cfg: Observation(torch.zeros(1), None, None)
    )
    bench = InversionBenchmark(out)
    tid = bench.core_ids[0]
    gidx, label, v_true, obs = st.load_target(
        bench_cfg, 0, torch.device("cpu"), target=tid, benchmark_root=out
    )
    assert label == tid
    assert gidx == bench.entry(tid)["global_index"]
    assert obs.sigma is None
    torch.testing.assert_close(v_true, bench.velocity(tid))


def test_manifest_is_json_clean(
    bench_cfg: OpenFWIDatasetConfig, tmp_path: Path
) -> None:
    out = tmp_path / "bench"
    write_benchmark(bench_cfg, out)
    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["targets"][0]
    assert set(entry) == {f.name for f in TargetEntry.__dataclass_fields__.values()}
