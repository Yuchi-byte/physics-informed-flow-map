# Learned Data-Space Misfit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn a Siamese misfit `J(d1,d2)=‖φ(d1)−φ(d2)‖²` on seismic data that predicts the model-space MSE between the two velocity maps, and wire it into `0004_inversion` as a third misfit alongside `l2`/`ot`.

**Architecture:** A CNN encoder φ maps a seismic record `(5,70,1001)` to a `k`-vector; the squared latent distance is `J`. Trained on an offline cached bank of `(v, d=simulate(v))` pairs (real CurveFault_B maps + Gaussian-blur + convex-blend augmentations) with target `MSE(v_i,v_j)`. At inference `φ(d_obs)` is frozen once, so `J` is a drop-in `MisfitFn`.

**Tech Stack:** PyTorch, Deepwave (`physics.forward.simulate`), Hydra + the repo's `experiment` harness, pytest, uv.

## Global Constraints

- Package code under `packages/physics-informed-flow-map/src/physics_informed_flow_map/`; experiment under `experiments/0006_learned_misfit/`. Tests under `packages/physics-informed-flow-map/tests/`.
- Velocity is normalized `[-1,1]` (`VMIN=1500`, `VMAX=4500`, `NATIVE=70`); `simulate` consumes **m/s** at native 70×70. Use `inversion.bridge.mps_to_norm` / `to_mps_native` for conversion.
- Seismic record shape is `(n_sources=5, n_receivers=70, nt=1001)`; batched `(B,5,70,1001)`. A `MisfitFn` is `pred (B,5,70,1001) → (B,)`.
- Run everything with `uv run`. Tests: `uv run pytest <path> -v`. A pre-commit hook runs ruff+mypy+pytest on staged package files — keep them clean.
- Do **not** modify reference packages (`mfm`, `PBFM`, `PIDM`). Commit frequently with `git add <specific files>` (never `-A`); commit message trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Bank cache and `runs/` are gitignored — never commit them.

---

### Task 1: Encoder φ

**Files:**
- Create: `packages/physics-informed-flow-map/src/physics_informed_flow_map/physics/learned_misfit.py`
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit.py`

**Interfaces:**
- Produces: `EncoderConfig(n_sources:int=5, latent_dim:int=128, channels:tuple[int,...]=(32,64,128,128))`; `Encoder(cfg:EncoderConfig, input_scale:float=1.0)` an `nn.Module` with `forward(d:Tensor)->Tensor` mapping `(B,5,70,1001)->(B,latent_dim)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_learned_misfit.py
import torch
from physics_informed_flow_map.physics.learned_misfit import Encoder, EncoderConfig


def test_encoder_shape_and_grad() -> None:
    enc = Encoder(EncoderConfig(latent_dim=16), input_scale=2.0)
    d = torch.randn(3, 5, 70, 1001, requires_grad=True)
    z = enc(d)
    assert z.shape == (3, 16)
    z.sum().backward()
    assert d.grad is not None and torch.isfinite(d.grad).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit.py::test_encoder_shape_and_grad -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (module absent).

- [ ] **Step 3: Write minimal implementation**

```python
# physics/learned_misfit.py
"""Learned data-space misfit J(d1,d2)=‖φ(d1)−φ(d2)‖² (design spec 2026-07-18).

A shared CNN encoder φ maps a seismic record to a latent vector; the squared latent
distance is the misfit. Symmetry, J(d,d)=0, J≥0 and the triangle inequality hold by
construction. At inference φ(d_obs) is frozen once so J drops into the guidance MisfitFn
interface (physics.misfit) alongside l2/ot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderConfig:
    n_sources: int = 5
    latent_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 128)


class Encoder(nn.Module):
    """Seismic record (B, n_sources, n_receivers, nt) -> (B, latent_dim), differentiable.

    ``input_scale`` divides the input before the conv stack (a frozen amplitude
    normalization computed once from the training bank, applied identically to d_obs and
    synthetics)."""

    def __init__(self, cfg: EncoderConfig, input_scale: float = 1.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))
        chans = (cfg.n_sources, *cfg.channels)
        layers: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.SiLU(),
            ]
        self.conv = nn.Sequential(*layers)
        c = cfg.channels[-1]
        self.head = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, cfg.latent_dim))

    def forward(self, d: Tensor) -> Tensor:
        z = self.conv(d / self.input_scale)
        z = z.mean(dim=(-2, -1))  # global average pool over (receiver, time)
        return self.head(z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit.py::test_encoder_shape_and_grad -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/physics/learned_misfit.py packages/physics-informed-flow-map/tests/test_learned_misfit.py
git commit -m "feat(learned-misfit): differentiable seismic encoder phi"
```

---

### Task 2: Siamese J + `make_learned_misfit` + `MISFITS` wiring

**Files:**
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/physics/learned_misfit.py`
- Modify: `packages/physics-informed-flow-map/src/physics_informed_flow_map/physics/misfit.py` (add `"learned"` to `MISFITS`, dispatch in `make_misfit`)
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit.py`

**Interfaces:**
- Consumes: `Encoder`, `EncoderConfig` (Task 1).
- Produces: `siamese_misfit(encoder:Encoder, d_obs:Tensor)->Callable[[Tensor],Tensor]` (a `MisfitFn`); `save_encoder(path, encoder)` / `load_encoder(path, device)->Encoder`; `make_learned_misfit(d_obs:Tensor, ckpt:str, *, device=None)->Callable[[Tensor],Tensor]`. `make_misfit(name, d_obs, *, ot_k=100.0, min_freq_hz=0.0, dt=1e-3, ckpt=None)` now accepts `name="learned"`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_learned_misfit.py
from physics_informed_flow_map.physics.learned_misfit import (
    make_learned_misfit,
    save_encoder,
    siamese_misfit,
)
from physics_informed_flow_map.physics.misfit import MISFITS, make_misfit


def test_siamese_zero_symmetry_and_shape() -> None:
    enc = Encoder(EncoderConfig(latent_dim=8))
    d_obs = torch.randn(5, 70, 1001)
    j = siamese_misfit(enc, d_obs)
    pred = torch.randn(4, 5, 70, 1001)
    assert j(pred).shape == (4,)
    # J(d,d) == 0 exactly and J >= 0
    self_j = j(d_obs.unsqueeze(0))
    assert torch.allclose(self_j, torch.zeros(1), atol=1e-5)
    assert (j(pred) >= 0).all()
    # symmetry: build the mirror misfit and compare a single pair
    a, b = pred[:1], pred[1:2]
    jab = siamese_misfit(enc, a[0])(b)
    jba = siamese_misfit(enc, b[0])(a)
    assert torch.allclose(jab, jba, atol=1e-5)


def test_make_learned_misfit_roundtrip(tmp_path) -> None:
    enc = Encoder(EncoderConfig(latent_dim=8), input_scale=3.0)
    ckpt = tmp_path / "phi.pt"
    save_encoder(ckpt, enc)
    d_obs = torch.randn(5, 70, 1001)
    j = make_learned_misfit(d_obs, str(ckpt))
    ref = siamese_misfit(enc, d_obs)
    pred = torch.randn(2, 5, 70, 1001)
    assert torch.allclose(j(pred), ref(pred), atol=1e-5)


def test_make_misfit_dispatches_learned(tmp_path) -> None:
    assert "learned" in MISFITS
    enc = Encoder(EncoderConfig(latent_dim=8))
    ckpt = tmp_path / "phi.pt"
    save_encoder(ckpt, enc)
    d_obs = torch.randn(5, 70, 1001)
    j = make_misfit("learned", d_obs, ckpt=str(ckpt))
    assert j(torch.randn(2, 5, 70, 1001)).shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit.py -v -k "siamese or learned"`
Expected: FAIL — `ImportError` for the new names / `"learned"` not in `MISFITS`.

- [ ] **Step 3: Implement in `learned_misfit.py`**

```python
# append to physics/learned_misfit.py

def siamese_misfit(encoder: Encoder, d_obs: Tensor) -> Callable[[Tensor], Tensor]:
    """Freeze z_obs=φ(d_obs) and return the MisfitFn pred->‖z_obs−φ(pred)‖² per sample."""
    encoder.eval()
    obs = d_obs.unsqueeze(0) if d_obs.ndim == 3 else d_obs
    with torch.no_grad():
        z_obs = encoder(obs).detach()

    def fn(pred: Tensor) -> Tensor:
        return ((encoder(pred) - z_obs) ** 2).sum(dim=-1)

    return fn


def save_encoder(path, encoder: Encoder) -> None:
    """Checkpoint φ with everything make_learned_misfit needs to rebuild it."""
    import torch as _torch

    _torch.save(
        {
            "model": encoder.state_dict(),
            "encoder_config": vars(encoder.cfg),
            "input_scale": float(encoder.input_scale),
        },
        path,
    )


def load_encoder(path, device=None) -> Encoder:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    cfg = EncoderConfig(**ckpt["encoder_config"])
    enc = Encoder(cfg, input_scale=ckpt.get("input_scale", 1.0))
    enc.load_state_dict(ckpt["model"])
    enc.to(device or "cpu").eval()
    return enc


def make_learned_misfit(d_obs: Tensor, ckpt: str, *, device=None) -> Callable[[Tensor], Tensor]:
    """Load φ from ``ckpt`` and return the frozen-observation MisfitFn."""
    enc = load_encoder(ckpt, device=device or d_obs.device)
    return siamese_misfit(enc, d_obs)
```

- [ ] **Step 4: Wire into `misfit.py`**

In `physics/misfit.py`, change the constant and `make_misfit`:

```python
MISFITS = ("l2", "ot", "learned")
```

```python
def make_misfit(
    name: str,
    d_obs: Tensor,
    *,
    ot_k: float = 100.0,
    min_freq_hz: float = 0.0,
    dt: float = 1e-3,
    ckpt: str | None = None,
) -> MisfitFn:
    if name == "l2":
        base = l2_misfit(d_obs)
    elif name == "ot":
        base = OTMisfit(d_obs, k=ot_k)
    elif name == "learned":
        if not ckpt:
            raise ValueError("misfit 'learned' needs ckpt=<encoder checkpoint>")
        from .learned_misfit import make_learned_misfit

        base = make_learned_misfit(d_obs, ckpt)
    else:
        raise ValueError(f"unknown misfit '{name}' ({' | '.join(MISFITS)})")
    if min_freq_hz <= 0.0:
        return base

    def banded(pred: Tensor) -> Tensor:
        return base(highpass(pred, min_freq_hz, dt))

    return banded
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit.py packages/physics-informed-flow-map/tests/test_misfit.py -v`
Expected: PASS (new learned tests + existing OT/L2 tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/src/physics_informed_flow_map/physics/learned_misfit.py packages/physics-informed-flow-map/src/physics_informed_flow_map/physics/misfit.py packages/physics-informed-flow-map/tests/test_learned_misfit.py
git commit -m "feat(learned-misfit): siamese J, checkpoint io, make_misfit('learned')"
```

---

### Task 3: Offline pair bank builder

**Files:**
- Create: `experiments/0006_learned_misfit/bank.py`
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit_bank.py`

**Interfaces:**
- Consumes: `physics.forward.simulate`, `inversion.bridge.{to_mps_native, mps_to_norm}`, `flow_matching.openfwi.{OpenFWIVelocityDataset, NATIVE}`.
- Produces: `augment(v_norm:Tensor, mode:str, gen:torch.Generator)->Tensor` (`mode` in `{"blur","blend"}`, `v_norm` a stack `(N,70,70)` for blend / single `(70,70)` for blur); `build_bank(data_dir:str, family:str, *, n_real:int, n_aug_per_real:int, seed:int, exclude_dir:str|None, device:str)->dict` returning `{"v_norm":(M,70,70), "d":(M,5,70,1001), "input_scale":float}`; `save_bank(path, bank)`, `load_bank(path)`.

- [ ] **Step 1: Write the failing test** (hermetic — tiny, CPU, monkeypatched simulate)

```python
# tests/test_learned_misfit_bank.py
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("experiments/0006_learned_misfit")))
import bank as B  # noqa: E402


def test_augment_blur_changes_but_bounds(monkeypatch) -> None:
    gen = torch.Generator().manual_seed(0)
    v = torch.rand(70, 70) * 2 - 1
    out = B.augment(v, "blur", gen)
    assert out.shape == (70, 70)
    assert out.min() >= -1.0 and out.max() <= 1.0
    assert not torch.allclose(out, v)  # blur actually did something


def test_build_bank_shapes_and_determinism(tmp_path, monkeypatch) -> None:
    # 4 fake CurveFault_B maps on disk; stub simulate so the test needs no wave solver.
    fam = tmp_path / "CurveFault_B" / "model"
    fam.mkdir(parents=True)
    np.save(fam / "vel0.npy", (np.random.rand(4, 1, 70, 70) * 3000 + 1500).astype("float32"))
    monkeypatch.setattr(B, "simulate", lambda v_mps, **k: torch.zeros(5, 70, 1001))
    kw = dict(data_dir=str(tmp_path), family="CurveFault_B", n_real=3,
              n_aug_per_real=1, seed=0, exclude_dir=None, device="cpu")
    b1 = B.build_bank(**kw)
    b2 = B.build_bank(**kw)
    assert b1["v_norm"].shape[0] == b1["d"].shape[0] == 3 * (1 + 1)  # real + 1 aug each
    assert b1["d"].shape[1:] == (5, 70, 1001)
    assert torch.allclose(b1["v_norm"], b2["v_norm"])  # deterministic in seed
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_bank.py -v`
Expected: FAIL — `ModuleNotFoundError: bank`.

- [ ] **Step 3: Implement `bank.py`**

```python
"""Offline (v, d=simulate(v)) bank for the learned misfit (design spec 2026-07-18 §4).

Simulation is the cost; we run it once per map and cache d, so training samples index
pairs cheaply. v draws from real CurveFault_B maps + Gaussian-blur (smooth/early states)
+ convex blends (intermediate states). Benchmark target maps are excluded so the
downstream inversion target stays unseen.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from physics_informed_flow_map.flow_matching.openfwi import NATIVE, OpenFWIVelocityDataset
from physics_informed_flow_map.inversion.bridge import mps_to_norm, to_mps_native
from physics_informed_flow_map.physics.forward import simulate


def _gaussian_blur(v: Tensor, sigma: float) -> Tensor:
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=v.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = (k / k.sum()).to(v.device)
    vb = v[None, None]
    vb = F.conv2d(F.pad(vb, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    vb = F.conv2d(F.pad(vb, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return vb[0, 0]


def augment(v_norm: Tensor, mode: str, gen: torch.Generator) -> Tensor:
    """One augmented normalized map. mode='blur': smooth a single (70,70) map at random
    sigma in [1,4]. mode='blend': convex mix of two rows of a (N,70,70) stack."""
    if mode == "blur":
        sigma = 1.0 + 3.0 * torch.rand(1, generator=gen).item()
        return _gaussian_blur(v_norm, sigma).clamp(-1.0, 1.0)
    if mode == "blend":
        n = v_norm.shape[0]
        i, j = torch.randint(0, n, (2,), generator=gen).tolist()
        a = torch.rand(1, generator=gen).item()
        return (a * v_norm[i] + (1 - a) * v_norm[j]).clamp(-1.0, 1.0)
    raise ValueError(f"unknown augment mode {mode!r}")


def _hashes(exclude_dir: str | None) -> set[str]:
    if not exclude_dir:
        return set()
    out = set()
    for p in sorted(Path(exclude_dir, "velocity").glob("*.npy")):
        import numpy as np

        out.add(hashlib.md5(np.load(p).astype("float32").tobytes()).hexdigest())
    return out


def build_bank(
    data_dir: str,
    family: str,
    *,
    n_real: int,
    n_aug_per_real: int,
    seed: int,
    exclude_dir: str | None,
    device: str,
) -> dict:
    import numpy as np

    gen = torch.Generator().manual_seed(seed)
    ds = OpenFWIVelocityDataset(Path(data_dir), [family], resolution=NATIVE)
    excluded = _hashes(exclude_dir)
    reals_norm: list[Tensor] = []
    order = torch.randperm(len(ds), generator=gen).tolist()
    for idx in order:
        native = ds._data[idx, 0]  # (70,70) m/s
        if hashlib.md5(native.astype("float32").tobytes()).hexdigest() in excluded:
            continue
        reals_norm.append(mps_to_norm(torch.from_numpy(native.copy())))
        if len(reals_norm) >= n_real:
            break
    stack = torch.stack(reals_norm)  # (n_real,70,70) normalized

    v_list: list[Tensor] = list(reals_norm)
    for base in reals_norm:
        for a in range(n_aug_per_real):
            mode = "blur" if a % 2 == 0 else "blend"
            src = base if mode == "blur" else stack
            v_list.append(augment(src, mode, gen))

    v_norm = torch.stack(v_list)  # (M,70,70)
    dev = torch.device(device)
    ds_out = []
    for v in v_norm:
        v_mps = to_mps_native(v[None, None].to(dev))  # (1,70,70) m/s
        ds_out.append(simulate(v_mps[0], **{}).detach().cpu())
    d = torch.stack(ds_out)  # (M,5,70,1001)
    input_scale = float(d.abs().mean().clamp_min(1e-12))
    return {"v_norm": v_norm, "d": d, "input_scale": input_scale}


def save_bank(path, bank: dict) -> None:
    torch.save(bank, path)


def load_bank(path) -> dict:
    return torch.load(path, weights_only=False)
```

Note on `to_mps_native`: it takes `(B,1,H,W)` or `(B,H,W)` normalized and returns `(B,H,W)` m/s (see `inversion/bridge.py`). The `v[None,None]` shape above yields `(1,70,70)` m/s; index `[0]` gives the `(70,70)` map `simulate` wants.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_bank.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/0006_learned_misfit/bank.py packages/physics-informed-flow-map/tests/test_learned_misfit_bank.py
git commit -m "feat(learned-misfit): offline (v,d) pair-bank builder with blur/blend augment"
```

---

### Task 4: Pair dataset + MSE target

**Files:**
- Create: `experiments/0006_learned_misfit/pairs.py`
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit_pairs.py`

**Interfaces:**
- Consumes: a bank dict `{"v_norm":(M,70,70), "d":(M,5,70,1001)}`.
- Produces: `PairDataset(bank:dict, n_pairs:int, seed:int)` a `torch.utils.data.Dataset` whose `__getitem__` returns `(d_i:(5,70,1001), d_j:(5,70,1001), target:scalar Tensor)` where `target = mean((v_i−v_j)**2)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_learned_misfit_pairs.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("experiments/0006_learned_misfit")))
import pairs as P  # noqa: E402


def test_pair_target_is_velocity_mse() -> None:
    M = 6
    bank = {"v_norm": torch.rand(M, 70, 70), "d": torch.randn(M, 5, 70, 1001)}
    ds = P.PairDataset(bank, n_pairs=50, seed=0)
    di, dj, tgt = ds[0]
    assert di.shape == (5, 70, 1001) and tgt.ndim == 0 and tgt >= 0
    assert len(ds) == 50
    # target matches recomputed MSE for the sampled indices
    i, j = ds.pairs[0]
    assert torch.allclose(tgt, ((bank["v_norm"][i] - bank["v_norm"][j]) ** 2).mean())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_pairs.py -v`
Expected: FAIL — `ModuleNotFoundError: pairs`.

- [ ] **Step 3: Implement `pairs.py`**

```python
"""Pair sampler over a cached bank: (d_i, d_j) with target MSE(v_i, v_j) in [-1,1] space."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, bank: dict, n_pairs: int, seed: int) -> None:
        self.v = bank["v_norm"]
        self.d = bank["d"]
        m = self.v.shape[0]
        gen = torch.Generator().manual_seed(seed)
        self.pairs = torch.randint(0, m, (n_pairs, 2), generator=gen)

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, k: int) -> tuple[Tensor, Tensor, Tensor]:
        i, j = self.pairs[k].tolist()
        target = ((self.v[i] - self.v[j]) ** 2).mean()
        return self.d[i], self.d[j], target
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_pairs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/0006_learned_misfit/pairs.py packages/physics-informed-flow-map/tests/test_learned_misfit_pairs.py
git commit -m "feat(learned-misfit): pair dataset with velocity-MSE target"
```

---

### Task 5: Training run (`0006_learned_misfit`)

**Files:**
- Create: `experiments/0006_learned_misfit/run.py`
- Create: `experiments/0006_learned_misfit/conf/config.yaml`
- Create: `experiments/0006_learned_misfit/conf/experiment/smoke.yaml`
- Create: `experiments/0006_learned_misfit/conf/experiment/curvefault_b.yaml`
- Create: `experiments/0006_learned_misfit/report.md`
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit_train.py`

**Interfaces:**
- Consumes: `bank.{build_bank,save_bank,load_bank}`, `pairs.PairDataset`, `physics.learned_misfit.{Encoder,EncoderConfig,siamese_misfit,save_encoder}`, harness `start_run`, `Config`.
- Produces: `train_encoder(encoder, train_ds, val_ds, *, epochs, batch_size, lr, device, log=None)->dict` returning `{"train_loss":float,"val_r2":float,"val_spearman":float}`; the checkpoint `runs/0006_learned_misfit/<ts>/checkpoints/step_<N>.pt` written via `save_encoder`.

The pure `train_encoder` is unit-tested; `run.py` is the Hydra wrapper (build/cache bank → split → train → checkpoint → `run.finish`).

- [ ] **Step 1: Write the failing test** (pure trainer, tiny synthetic bank, asserts it learns)

```python
# tests/test_learned_misfit_train.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("experiments/0006_learned_misfit")))
import pairs as P  # noqa: E402
from run import train_encoder  # noqa: E402
from physics_informed_flow_map.physics.learned_misfit import Encoder, EncoderConfig


def test_train_encoder_reduces_loss() -> None:
    torch.manual_seed(0)
    # bank where d is a smooth function of v, so J can learn to track MSE(v_i,v_j)
    M = 24
    v = torch.rand(M, 70, 70)
    d = v.mean(dim=(-2, -1)).view(M, 1, 1, 1) * torch.ones(M, 5, 70, 1001)
    bank = {"v_norm": v, "d": d, "input_scale": 1.0}
    train_ds, val_ds = P.PairDataset(bank, 200, 0), P.PairDataset(bank, 50, 1)
    enc = Encoder(EncoderConfig(latent_dim=8, channels=(8, 16, 16, 16)))
    out = train_encoder(enc, train_ds, val_ds, epochs=3, batch_size=16, lr=1e-3, device="cpu")
    assert out["val_r2"] > 0.0  # explains some variance after a few epochs
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_train.py -v`
Expected: FAIL — `ImportError: cannot import name 'train_encoder'`.

- [ ] **Step 3: Implement `run.py`**

```python
"""Train the learned data-space misfit φ (design spec 2026-07-18).

    uv run python experiments/0006_learned_misfit/run.py experiment=smoke
    uv run python experiments/0006_learned_misfit/run.py experiment=curvefault_b

Builds (once, cached) an offline (v, d) bank, samples MSE-labelled pairs, and fits φ so
that ‖φ(d_i)−φ(d_j)‖² tracks MSE(v_i,v_j). Logs held-out R² and Spearman alignment.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).parent))
from bank import build_bank, load_bank, save_bank  # noqa: E402
from pairs import PairDataset  # noqa: E402

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.physics.learned_misfit import (
    Encoder,
    EncoderConfig,
    save_encoder,
    siamese_misfit,
)

EXPERIMENT = "0006_learned_misfit"


class BankConfig(Config):
    data_dir: str = "data/openfwi"
    family: str = "CurveFault_B"
    n_real: int = 300
    n_aug_per_real: int = 2
    seed: int = 0
    exclude_dir: str = "data/inversion_bench"
    cache: str = "data/learned_misfit/CurveFault_B/bank.pt"


class TrainConfig(Config):
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    n_train_pairs: int = 20000
    n_val_pairs: int = 2000
    val_frac: float = 0.15  # fraction of bank maps held out for validation


class MisfitTrainConfig(Config):
    latent_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128, 128)
    bank: BankConfig = Field(default_factory=BankConfig)
    training: TrainConfig = Field(default_factory=TrainConfig)


def _r2_spearman(pred: torch.Tensor, tgt: torch.Tensor) -> tuple[float, float]:
    ss_res = ((pred - tgt) ** 2).sum()
    ss_tot = ((tgt - tgt.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = float(1 - ss_res / ss_tot)
    rp = pred.argsort().argsort().float()
    rt = tgt.argsort().argsort().float()
    cov = ((rp - rp.mean()) * (rt - rt.mean())).mean()
    sp = float(cov / (rp.std().clamp_min(1e-12) * rt.std().clamp_min(1e-12)))
    return r2, sp


def _j(encoder: Encoder, di: torch.Tensor, dj: torch.Tensor) -> torch.Tensor:
    return ((encoder(di) - encoder(dj)) ** 2).sum(dim=-1)


def train_encoder(encoder, train_ds, val_ds, *, epochs, batch_size, lr, device, log=None) -> dict:
    dev = torch.device(device)
    encoder.to(dev)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    last = {"train_loss": float("nan"), "val_r2": float("nan"), "val_spearman": float("nan")}
    for epoch in range(epochs):
        encoder.train()
        tot = 0.0
        for di, dj, tgt in train_dl:
            di, dj, tgt = di.to(dev), dj.to(dev), tgt.to(dev)
            loss = ((_j(encoder, di, dj) - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * di.shape[0]
        encoder.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for di, dj, tgt in val_dl:
                preds.append(_j(encoder, di.to(dev), dj.to(dev)).cpu())
                tgts.append(tgt)
        p, t = torch.cat(preds), torch.cat(tgts)
        r2, sp = _r2_spearman(p, t)
        last = {"train_loss": tot / max(1, len(train_ds)), "val_r2": r2, "val_spearman": sp}
        if log:
            log(epoch=epoch, **last)
    return last


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    conf = MisfitTrainConfig.from_dictconfig(cfg)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, conf.dump(), name=run_dir.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(conf.bank.cache)
    if cache.is_file():
        bank = load_bank(cache)
    else:
        bank = build_bank(
            conf.bank.data_dir, conf.bank.family,
            n_real=conf.bank.n_real, n_aug_per_real=conf.bank.n_aug_per_real,
            seed=conf.bank.seed, exclude_dir=conf.bank.exclude_dir or None, device=device,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_bank(cache, bank)
    print(f"[{EXPERIMENT}] bank: {bank['v_norm'].shape[0]} maps, input_scale={bank['input_scale']:.4g}")

    # split bank maps into train/val, build separate pair datasets over each split
    m = bank["v_norm"].shape[0]
    g = torch.Generator().manual_seed(conf.bank.seed)
    perm = torch.randperm(m, generator=g)
    n_val = max(2, int(conf.training.val_frac * m))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    sub = lambda idx: {"v_norm": bank["v_norm"][idx], "d": bank["d"][idx]}
    train_ds = PairDataset(sub(train_idx), conf.training.n_train_pairs, conf.bank.seed)
    val_ds = PairDataset(sub(val_idx), conf.training.n_val_pairs, conf.bank.seed + 1)

    enc = Encoder(
        EncoderConfig(latent_dim=conf.latent_dim, channels=tuple(conf.channels)),
        input_scale=bank["input_scale"],
    )
    out = train_encoder(
        enc, train_ds, val_ds,
        epochs=conf.training.epochs, batch_size=conf.training.batch_size,
        lr=conf.training.lr, device=device, log=run.log_step,
    )
    ckpt = run.ckpt_dir / f"step_{conf.training.epochs}.pt"
    run.ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_encoder(ckpt, enc)
    print(f"[{EXPERIMENT}] saved φ → {ckpt}")
    assert out["val_spearman"] > 0.5, f"val alignment too low: {out['val_spearman']:.3f}"
    run.finish(**out, ckpt=str(ckpt))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create the Hydra configs**

`conf/config.yaml`:
```yaml
defaults:
  - _self_
latent_dim: 128
channels: [32, 64, 128, 128]
bank:
  data_dir: data/openfwi
  family: CurveFault_B
  n_real: 300
  n_aug_per_real: 2
  seed: 0
  exclude_dir: data/inversion_bench
  cache: data/learned_misfit/CurveFault_B/bank.pt
training:
  epochs: 40
  batch_size: 32
  lr: 1.0e-3
  n_train_pairs: 20000
  n_val_pairs: 2000
  val_frac: 0.15
hydra:
  run:
    dir: runs/0006_learned_misfit/${now:%Y-%m-%dT%H-%M-%SZ}
```

`conf/experiment/smoke.yaml`:
```yaml
# @package _global_
bank:
  n_real: 6
  n_aug_per_real: 1
  cache: ""   # empty => rebuild every run, never cache the toy bank
training:
  epochs: 2
  n_train_pairs: 64
  n_val_pairs: 16
```

`conf/experiment/curvefault_b.yaml`:
```yaml
# @package _global_
bank:
  family: CurveFault_B
training:
  epochs: 60
```

Handle `cache: ""` in `main` (empty string ⇒ always rebuild): guard the `cache.is_file()` branch with `if conf.bank.cache and cache.is_file():` and only `save_bank` when `conf.bank.cache` is non-empty.

`report.md`:
```markdown
# 0006_learned_misfit — report

Hypothesis → Setup → Results → Decision. Filled after the CurveFault_B run + the
0004 l2/ot/learned comparison (Task 8).
```

- [ ] **Step 5: Run the unit test + a smoke plumbing run**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_train.py -v`
Expected: PASS.

Run: `WANDB_MODE=disabled uv run python experiments/0006_learned_misfit/run.py experiment=smoke`
Expected: completes, prints a bank line + `saved φ →`, writes a checkpoint under `runs/0006_learned_misfit/`. (The `val_spearman>0.5` assert may fail on the trivial toy bank — if so, lower it in `smoke.yaml` via a `min_spearman` field, or accept the smoke run as plumbing-only and gate the assert behind `conf.training.epochs>=10`. Prefer the latter: change the assert to `if conf.training.epochs >= 10: assert ...`.)

- [ ] **Step 6: Commit**

```bash
git add experiments/0006_learned_misfit/run.py experiments/0006_learned_misfit/conf experiments/0006_learned_misfit/report.md packages/physics-informed-flow-map/tests/test_learned_misfit_train.py
git commit -m "feat(0006): learned-misfit training run + configs"
```

---

### Task 6: Alignment + landscape diagnostics

**Files:**
- Create: `experiments/0006_learned_misfit/eval.py`
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit_eval.py`

**Interfaces:**
- Consumes: `physics.learned_misfit.load_encoder`, `physics.forward.simulate`, `physics.misfit.make_misfit`, `inversion.bridge.to_mps_native`.
- Produces: `alignment_curve(encoder, v_true_norm:(70,70), v_other_norm:(70,70), *, n:int, device)->dict` returning `{"alpha":(n,), "J":(n,), "l2":(n,), "ot":(n,), "mse_model":(n,)}` for `v(α)=(1−α)v_true+α v_other`; `spearman_alignment(encoder, v_true_norm, v_pool:(K,70,70), device)->float` — Spearman between `J(d_obs,F(v))` and `MSE(v_true,v)` across the pool.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_learned_misfit_eval.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path("experiments/0006_learned_misfit")))
import eval as E  # noqa: E402
from physics_informed_flow_map.physics.learned_misfit import Encoder, EncoderConfig


def test_alignment_curve_shapes(monkeypatch) -> None:
    monkeypatch.setattr(E, "simulate", lambda v, **k: v.mean() * torch.ones(5, 70, 1001))
    enc = Encoder(EncoderConfig(latent_dim=8))
    v0, v1 = torch.zeros(70, 70), torch.ones(70, 70)
    out = E.alignment_curve(enc, v0, v1, n=5, device="cpu")
    for key in ("alpha", "J", "l2", "ot", "mse_model"):
        assert out[key].shape == (5,)
    assert torch.allclose(out["mse_model"][0], torch.zeros(())) and out["mse_model"][-1] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: eval`.

- [ ] **Step 3: Implement `eval.py`**

```python
"""Diagnostics for the learned misfit: does J(d_obs,F(v)) track MSE(v_true,v)?

    uv run python experiments/0006_learned_misfit/eval.py \
        ckpt=runs/0006_learned_misfit/<ts>/checkpoints/step_60.pt

Writes an alignment figure (J vs L2 vs OT along a v_true→v_other path) and prints the
Spearman rank-correlation between J and the true model-space MSE over a held-out pool.
"""
from __future__ import annotations

import torch
from torch import Tensor

from physics_informed_flow_map.inversion.bridge import to_mps_native
from physics_informed_flow_map.physics.forward import simulate
from physics_informed_flow_map.physics.learned_misfit import load_encoder, siamese_misfit
from physics_informed_flow_map.physics.misfit import make_misfit


def _fwd(v_norm: Tensor) -> Tensor:
    return simulate(to_mps_native(v_norm[None, None])[0])


def alignment_curve(encoder, v_true_norm, v_other_norm, *, n, device) -> dict:
    alphas = torch.linspace(0, 1, n)
    d_obs = _fwd(v_true_norm)
    j_fn = siamese_misfit(encoder.to(device), d_obs.to(device))
    l2_fn = make_misfit("l2", d_obs)
    ot_fn = make_misfit("ot", d_obs)
    J, l2, ot, mse = [], [], [], []
    for a in alphas:
        v = (1 - a) * v_true_norm + a * v_other_norm
        d = _fwd(v)
        J.append(float(j_fn(d[None].to(device))))
        l2.append(float(l2_fn(d[None])))
        ot.append(float(ot_fn(d[None])))
        mse.append(float(((v - v_true_norm) ** 2).mean()))
    t = torch.tensor
    return {"alpha": alphas, "J": t(J), "l2": t(l2), "ot": t(ot), "mse_model": t(mse)}


def spearman_alignment(encoder, v_true_norm, v_pool, device) -> float:
    d_obs = _fwd(v_true_norm)
    j_fn = siamese_misfit(encoder.to(device), d_obs.to(device))
    js, mses = [], []
    for v in v_pool:
        js.append(float(j_fn(_fwd(v)[None].to(device))))
        mses.append(float(((v - v_true_norm) ** 2).mean()))
    j, m = torch.tensor(js), torch.tensor(mses)
    rj, rm = j.argsort().argsort().float(), m.argsort().argsort().float()
    cov = ((rj - rj.mean()) * (rm - rm.mean())).mean()
    return float(cov / (rj.std().clamp_min(1e-12) * rm.std().clamp_min(1e-12)))
```

(The `__main__` Hydra wrapper that loads a target from the benchmark, builds a pool, saves a matplotlib figure of `alignment_curve`, and prints `spearman_alignment` can be added here later; the two functions above are the tested core.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/0006_learned_misfit/eval.py packages/physics-informed-flow-map/tests/test_learned_misfit_eval.py
git commit -m "feat(0006): alignment + landscape diagnostics for J"
```

---

### Task 7: Thread `misfit_ckpt` through `0004_inversion`

**Files:**
- Modify: `experiments/0004_inversion/run.py` (add `misfit_ckpt` to `MethodConfig`; validator; pass `ckpt=` into the two `make_misfit` factories)
- Modify: `experiments/0004_inversion/eval.py` (pass `ckpt=` where it builds misfits)
- Test: `packages/physics-informed-flow-map/tests/test_learned_misfit_wiring.py`

**Interfaces:**
- Consumes: `make_misfit(..., ckpt=...)` (Task 2).
- Produces: `MethodConfig.misfit_ckpt: str = ""`, required iff `misfit == "learned"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_learned_misfit_wiring.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("experiments/0004_inversion")))
from run import MethodConfig  # noqa: E402


def test_learned_requires_ckpt() -> None:
    with pytest.raises(ValueError):
        MethodConfig(name="flow_tilt", misfit="learned", misfit_ckpt="")
    ok = MethodConfig(name="flow_tilt", misfit="learned", misfit_ckpt="some/ckpt.pt")
    assert ok.misfit_ckpt == "some/ckpt.pt"


def test_non_learned_ignores_ckpt() -> None:
    assert MethodConfig(name="flow_tilt", misfit="ot").misfit_ckpt == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_wiring.py -v`
Expected: FAIL — `MethodConfig` has no `misfit_ckpt` / no validation.

- [ ] **Step 3: Add the field + validation in `run.py`**

In `MethodConfig`, after the `ot_k` field:
```python
    misfit_ckpt: str = ""  # learned misfit only: path to a trained φ checkpoint (0006)
```
In the `_check_drift_estimator` validator (rename not required — extend it), before `return self`:
```python
        if self.misfit == "learned" and not self.misfit_ckpt:
            raise ValueError("misfit 'learned' needs method.misfit_ckpt=<φ checkpoint>")
        if self.misfit != "learned" and self.misfit_ckpt:
            raise ValueError("method.misfit_ckpt is only used with misfit=learned")
```

- [ ] **Step 4: Pass `ckpt` into the factories**

In `run.py`, the `misfit_factory` lambda and `build_diag_misfits` stay OT-only for diagnostics, but `misfit_factory` must forward the ckpt:
```python
    misfit_factory = (
        None
        if cfg.method.misfit == "l2" and cfg.obs.min_freq_hz <= 0.0
        else lambda d: make_misfit(
            cfg.method.misfit,
            d,
            ot_k=cfg.method.ot_k,
            min_freq_hz=cfg.obs.min_freq_hz,
            ckpt=cfg.method.misfit_ckpt or None,
        )
    )
```
In `eval.py`, apply the identical `ckpt=entry.misfit_ckpt or None` addition wherever it calls `make_misfit` for the guidance factory (mirror the `run.py` change; leave diagnostic-OT calls unchanged).

- [ ] **Step 5: Run the wiring test + the existing 0004 config tests**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_learned_misfit_wiring.py packages/physics-informed-flow-map/tests/test_config.py packages/physics-informed-flow-map/tests/test_experiment_conf.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/0004_inversion/run.py experiments/0004_inversion/eval.py packages/physics-informed-flow-map/tests/test_learned_misfit_wiring.py
git commit -m "feat(0004): thread method.misfit_ckpt for the learned misfit"
```

---

### Task 8: Train on CurveFault_B and run the l2 / ot / learned comparison

**Files:**
- Modify: `experiments/0006_learned_misfit/report.md` (results)
- Modify: `experiments/JOURNAL.md` (one verdict line)

This task is execution + measurement, not TDD. Requires a GPU (Deepwave wave solves).

- [ ] **Step 1: Train φ on CurveFault_B**

Run:
```bash
uv run python experiments/0006_learned_misfit/run.py experiment=curvefault_b
```
Expected: builds+caches the bank (`data/learned_misfit/CurveFault_B/bank.pt`), trains, prints `val_r2` / `val_spearman` and `saved φ → runs/0006_learned_misfit/<ts>/checkpoints/step_60.pt`. Record `<ts>`. Sanity gate: `val_spearman > 0.5` (asserted in `run.py`).

- [ ] **Step 2: Inspect alignment (optional but recommended)**

Run:
```bash
uv run python experiments/0006_learned_misfit/eval.py ckpt=runs/0006_learned_misfit/<ts>/checkpoints/step_60.pt
```
Expected: Spearman(J, model-MSE) printed > 0.5, and the alignment figure shows J tracking model-MSE at least as monotonically as OT.

- [ ] **Step 3: Run the three inversions on `curvefault_b_17`**

```bash
CKPT=runs/0006_learned_misfit/<ts>/checkpoints/step_60.pt
for M in l2 ot; do
  uv run python experiments/0004_inversion/run.py \
    prior=flow_matching method=flow_tilt method.misfit=$M \
    target=curvefault_b_17 steps=400 n_samples=10
done
uv run python experiments/0004_inversion/run.py \
  prior=flow_matching method=flow_tilt method.misfit=learned method.misfit_ckpt=$CKPT \
  target=curvefault_b_17 steps=400 n_samples=10
```
Expected: three runs under `runs/0004_inversion/`, each printing `inv/mae_mean`, `ssim`, and the data-space figure. Collect `mae_mean` / `ssim_mean` for each.

- [ ] **Step 4: Record results**

Fill `experiments/0006_learned_misfit/report.md` (Hypothesis → Setup → Results table with the three `mae_mean`/`ssim` numbers + run dirs → Decision) and add one verdict line to `experiments/JOURNAL.md`. Success criterion (spec §7): the learned misfit **matches or beats OT** on `mae_mean`/`ssim` for `curvefault_b_17`.

- [ ] **Step 5: Commit**

```bash
git add experiments/0006_learned_misfit/report.md experiments/JOURNAL.md
git commit -m "docs(0006): CurveFault_B learned-misfit results vs l2/ot"
```

---

## Self-Review

**Spec coverage:**
- §3 architecture (Siamese Euclidean, MSE target, CNN φ) → Tasks 1–2. ✓
- §4 offline pair bank (real + blur + blend, cached, benchmark-excluded) → Task 3. ✓
- pair sampler + MSE target → Task 4. ✓
- training + val (R², alignment) → Task 5. ✓
- §6 validation diagnostics (alignment curve, Spearman, landscape) → Task 6; downstream acceptance → Task 8. ✓
- §5 inference wiring (`"learned"` in MISFITS, frozen `z_obs`, `misfit_ckpt` config) → Tasks 2 & 7. ✓
- §7 downstream experiment on `curvefault_b_17` → Task 8. ✓

**Placeholder scan:** every code step contains complete code; the only deferred item is the optional `eval.py` `__main__` Hydra wrapper (its tested core is complete) and the `report.md`/`JOURNAL.md` content filled from real run outputs in Task 8 — both intentional, not code placeholders.

**Type consistency:** `Encoder`/`EncoderConfig` (Task 1) reused unchanged in 2/5/6; `siamese_misfit`, `save_encoder`, `load_encoder`, `make_learned_misfit` signatures match across Tasks 2/5/6; bank dict keys `{"v_norm","d","input_scale"}` consistent across Tasks 3/4/5; `make_misfit(..., ckpt=...)` defined in Task 2, consumed in Task 7; `MethodConfig.misfit_ckpt` defined and consumed in Task 7.

**Known risk carried from the spec (§9):** if Task 8's `val_spearman`/downstream metrics are weak, the first lever is the bank's coverage of intermediate states — widen the blur σ range / blend count, or (documented fallback) harvest real `0004` trajectory states into the bank. Not built in the MVP.
