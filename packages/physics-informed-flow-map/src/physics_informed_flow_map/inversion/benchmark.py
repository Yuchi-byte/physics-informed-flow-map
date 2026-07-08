"""Self-contained inversion benchmark set: selection, manifest, previews, loader.

Distills the 470k-map OpenFWI collection into a small fixed target set
(``data/inversion_bench/``) that survives bulk-data deletion: per family, val-split maps
are ranked by total variation (a structural-complexity proxy) and drawn at fixed
percentiles, so the set spans easy→hard within every family without hand-picking bias
(design spec §7). The legacy FlatVel_A map 6044 is appended by ``(file, row)`` provenance
for continuity with all existing journal results.

Layout under the benchmark root:
- ``manifest.json`` — ids, provenance, stats, tags + the dataset fingerprint.
- ``velocity/<id>.npy`` — native 70x70 float32 m/s.
- ``previews/<id>.png`` + ``previews/gallery_<family>.png`` — fixed 1500–4500 m/s scale.

:class:`InversionBenchmark` reads these back with no dependency on the bulk dataset.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..flow_matching.datasets import OpenFWIDatasetConfig
from ..flow_matching.openfwi import NATIVE, VMAX, VMIN, OpenFWIVelocityDataset

PERCENTILES = (5, 25, 50, 75, 95)
DRAWS_PER_PERCENTILE = 4
# The first CORE_DRAWS draws at every percentile are tagged core: true — a fixed ~half-size
# subset for day-to-day comparisons; the rest is reserve (spec §7).
CORE_DRAWS = 2
# Piecewise-constant families have a handful of distinct velocities; Style maps are
# continuous-valued. Above this many unique values, "number of layers" is meaningless.
_MAX_LAYERS = 24

ALL_FAMILIES = [
    "CurveFault_A",
    "CurveFault_B",
    "CurveVel_A",
    "CurveVel_B",
    "FlatFault_A",
    "FlatFault_B",
    "FlatVel_A",
    "FlatVel_B",
    "Style_A",
    "Style_B",
]


@dataclass
class TargetEntry:
    """One benchmark target: identity, provenance, stats, and selection tags."""

    id: str
    family: str
    file: str  # path relative to the family directory, e.g. "model/model20.npy"
    row: int
    global_index: int  # under the manifest's dataset fingerprint (bulk data order)
    tv: float
    tv_percentile: int | None  # None for the provenance-pinned legacy target
    draw: int | None
    vmin_mps: float
    vmax_mps: float
    vmean_mps: float
    n_layers: int | None
    has_fault: bool
    core: bool
    legacy: bool = False
    in_current_val: bool = True


def total_variation(maps: np.ndarray) -> np.ndarray:
    """Anisotropic total variation per map, ``(N, H, W)`` -> ``(N,)``.

    Sum of absolute horizontal + vertical first differences in m/s — layered maps score
    low, faulted/textured maps high, so it orders val maps by structural complexity.
    """
    dy = np.abs(np.diff(maps, axis=-2)).sum(axis=(-2, -1))
    dx = np.abs(np.diff(maps, axis=-1)).sum(axis=(-2, -1))
    return np.asarray(dy + dx, dtype=np.float64)


def _n_layers(native: np.ndarray) -> int | None:
    n_unique = len(np.unique(native))
    return n_unique if n_unique <= _MAX_LAYERS else None


def _entry_stats(native: np.ndarray) -> tuple[float, float, float]:
    return float(native.min()), float(native.max()), float(native.mean())


def _relative_file(full: OpenFWIVelocityDataset, gidx: int, data_dir: Path) -> str:
    path, _ = full.index[gidx]
    family_dir = path.parent if path.parent.parent == data_dir else path.parent.parent
    return str(path.relative_to(family_dir))


def select_targets(
    cfg: OpenFWIDatasetConfig,
    *,
    percentiles: tuple[int, ...] = PERCENTILES,
    draws: int = DRAWS_PER_PERCENTILE,
    core_draws: int = CORE_DRAWS,
) -> list[TargetEntry]:
    """TV-percentile-stratified draws from every family's val split.

    Deterministic: draw seeds derive from the family name and percentile, and candidate
    buckets are windows in TV-rank order, so the same dataset + config always selects the
    same targets.
    """
    full, _, val_idx = cfg._split()
    data_dir = Path(cfg.data_dir)
    entries: list[TargetEntry] = []
    for family, sl in zip(full.family_names, full.family_slices):
        fam_val = np.array([i for i in val_idx if sl.start <= i < sl.stop])
        native = full._data[fam_val, 0]  # raw m/s values
        order = fam_val[np.argsort(total_variation(native), kind="stable")]
        n = len(order)
        chosen: list[int] = []  # positions in TV-rank order
        seq = 0
        for p in percentiles:
            anchor = round(p / 100 * (n - 1))
            half = max(draws, round(0.025 * n))
            rng = np.random.default_rng(zlib.crc32(f"{family}|p{p}".encode()))
            for d in range(min(draws, n - len(chosen))):
                pool = np.array([], dtype=np.int64)
                while len(pool) == 0:  # widen until an unchosen candidate exists
                    lo, hi = max(0, anchor - half), min(n, anchor + half + 1)
                    pool = np.setdiff1d(np.arange(lo, hi), np.array(chosen, dtype=int))
                    half *= 2
                rank = int(rng.choice(pool))
                chosen.append(rank)
                gidx = int(order[rank])
                nat = full._data[gidx, 0]
                vmin, vmax, vmean = _entry_stats(nat)
                entries.append(
                    TargetEntry(
                        id=f"{family.lower()}_{seq:02d}",
                        family=family,
                        file=_relative_file(full, gidx, data_dir),
                        row=full.index[gidx][1],
                        global_index=gidx,
                        tv=float(total_variation(nat[None])[0]),
                        tv_percentile=p,
                        draw=d,
                        vmin_mps=vmin,
                        vmax_mps=vmax,
                        vmean_mps=vmean,
                        n_layers=_n_layers(nat),
                        has_fault="Fault" in family,
                        core=d < core_draws,
                    )
                )
                seq += 1
    return entries


def legacy_entry(cfg: OpenFWIDatasetConfig, provenance: dict[str, Any]) -> TargetEntry:
    """The provenance-pinned legacy target (FlatVel_A map 6044 of the single-family era).

    Matched by ``(file basename, row)`` — its old global index is meaningless under the
    current families list, and per-family reseeding means it may no longer sit in the val
    split (``in_current_val`` records the truth for downstream caveats).
    """
    full, _, val_idx = cfg._split()
    family = provenance["family"]
    basename, row = provenance["file"], int(provenance["row"])
    sl = full.family_slices[full.family_names.index(family)]
    matches = [
        i
        for i in range(sl.start, sl.stop)
        if full.index[i][0].name == basename and full.index[i][1] == row
    ]
    if len(matches) != 1:
        raise ValueError(
            f"legacy provenance ({family}, {basename}, row {row}) matched "
            f"{len(matches)} samples; expected exactly 1"
        )
    gidx = matches[0]
    nat = full._data[gidx, 0]
    vmin, vmax, vmean = _entry_stats(nat)
    return TargetEntry(
        id=f"{family.lower()}_legacy_{provenance['legacy_global_index']}",
        family=family,
        file=_relative_file(full, gidx, Path(cfg.data_dir)),
        row=row,
        global_index=gidx,
        tv=float(total_variation(nat[None])[0]),
        tv_percentile=None,
        draw=None,
        vmin_mps=vmin,
        vmax_mps=vmax,
        vmean_mps=vmean,
        n_layers=_n_layers(nat),
        has_fault="Fault" in family,
        core=True,
        legacy=True,
        in_current_val=gidx in set(val_idx),
    )


def _preview(native: np.ndarray, entry: TargetEntry, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    im = ax.imshow(native, cmap="viridis", vmin=VMIN, vmax=VMAX)
    ax.set_title(entry.id, fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _gallery(
    natives: list[np.ndarray], entries: list[TargetEntry], path: Path, *, ncols: int = 5
) -> None:
    nrows = (len(entries) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        if i < len(entries):
            ax.imshow(natives[i], cmap="viridis", vmin=VMIN, vmax=VMAX)
            ax.set_title(entries[i].id, fontsize=7)
        ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def write_benchmark(
    cfg: OpenFWIDatasetConfig,
    out_dir: Path,
    *,
    legacy_provenance: dict[str, Any] | None = None,
    percentiles: tuple[int, ...] = PERCENTILES,
    draws: int = DRAWS_PER_PERCENTILE,
) -> dict[str, Any]:
    """Build the complete benchmark directory; returns the manifest."""
    full, _, _ = cfg._split()
    entries = select_targets(cfg, percentiles=percentiles, draws=draws)
    if legacy_provenance is not None:
        entries.append(legacy_entry(cfg, legacy_provenance))
    (out_dir / "velocity").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    for e in entries:
        native = np.ascontiguousarray(full._data[e.global_index, 0], dtype=np.float32)
        np.save(out_dir / "velocity" / f"{e.id}.npy", native)
        _preview(native, e, out_dir / "previews" / f"{e.id}.png")
    for family in {e.family for e in entries}:
        fam = [e for e in entries if e.family == family]
        _gallery(
            [full._data[e.global_index, 0] for e in fam],
            fam,
            out_dir / "previews" / f"gallery_{family}.png",
        )
    manifest = {
        "schema_version": 1,
        "normalization_mps": [VMIN, VMAX],
        "native_resolution": NATIVE,
        "selection": {
            "complexity_proxy": "anisotropic total variation (m/s)",
            "percentiles": list(percentiles),
            "draws_per_percentile": draws,
            "core_draws": CORE_DRAWS,
        },
        "dataset_fingerprint": cfg.fingerprint(),
        "targets": [asdict(e) for e in entries],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


class InversionBenchmark:
    """Manifest-driven access to the benchmark targets — no bulk-data dependency."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifest: dict[str, Any] = json.loads(
            (self.root / "manifest.json").read_text()
        )
        self.entries: dict[str, dict[str, Any]] = {
            t["id"]: t for t in self.manifest["targets"]
        }

    @property
    def ids(self) -> list[str]:
        return list(self.entries)

    @property
    def core_ids(self) -> list[str]:
        return [i for i, t in self.entries.items() if t["core"]]

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, target_id: str) -> bool:
        return target_id in self.entries

    def entry(self, target_id: str) -> dict[str, Any]:
        return self.entries[target_id]

    def velocity(self, target_id: str) -> Tensor:
        """Native ``(70, 70)`` float32 velocity in m/s (the held_out_targets contract)."""
        native = np.load(self.root / "velocity" / f"{target_id}.npy")
        return torch.from_numpy(native).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/openfwi")
    parser.add_argument("--out", default="data/inversion_bench")
    parser.add_argument(
        "--legacy-provenance",
        default="data/inversion_bench/legacy_6044_provenance.json",
    )
    args = parser.parse_args()
    cfg = OpenFWIDatasetConfig(
        data_dir=args.data_dir,
        families=ALL_FAMILIES,
        split_scheme="per_family",
        hflip=True,  # matches the definitive training configs' fingerprint
    )
    legacy = json.loads(Path(args.legacy_provenance).read_text())
    manifest = write_benchmark(cfg, Path(args.out), legacy_provenance=legacy)
    targets = manifest["targets"]
    n_core = sum(1 for t in targets if t["core"])
    legacy_t = next(t for t in targets if t["legacy"])
    print(f"{len(targets)} targets ({n_core} core) -> {args.out}")
    print(
        f"legacy {legacy_t['id']}: global_index={legacy_t['global_index']}, "
        f"in_current_val={legacy_t['in_current_val']}"
    )


if __name__ == "__main__":
    main()
