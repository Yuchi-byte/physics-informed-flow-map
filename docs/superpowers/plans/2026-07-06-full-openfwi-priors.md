# Full-OpenFWI Priors + Inversion Benchmark — Implementation Plan

> **Status: DRAFT — design questions resolved 2026-07-06 (spec §9); ready for code-level TDD
> expansion (exact test/code listings per step, per repo convention) on sign-off.**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train definitive flow-matching (0001), flow-map (0002), and diffusion (0003) priors on
all 10 2D OpenFWI families (~470k maps), publish verified checkpoints as wandb artifacts, build a
101-target self-contained inversion benchmark, then delete the bulk data.

**Design:** `docs/superpowers/specs/2026-07-06-full-openfwi-priors-design.md`

**Tech stack:** existing harness (pydantic Config + Hydra + wandb), `hf` CLI, Deepwave,
single-GPU PyTorch (bf16 autocast to be added).

## Global constraints

- Do NOT edit reference packages (`mfm-meta-flow-map-main/`, `PhysicsInformedDiffusionModels-main/`, `PBFM/`).
- Work and commit directly on `main`; pre-commit chain (ruff → mypy → pytest) must pass.
- Tests hermetic — synthetic `.npy` fixtures, never the real download; wandb tests under `WANDB_MODE=disabled`.
- Normalization stays fixed: `[1500, 4500] m/s → [-1, 1]`, native 70×70, training resolution 64.
- Training stays **unconditional** — family ids are for eval/metrics only, never model input.
- **Task 0 (legacy pin) must land before any split-logic change.**

---

### Task 0: Pin legacy provenance + download the six missing families

No production code. Blocks everything else.

- [ ] Resolve map 6044's `(file, row)` under the *current* `families=[FlatVel_A]` seed-0 split
      (`cfg._split()` + `full.index[6044]`); record it in a scratch note for Task 8.
- [ ] Velocity-only download (~7 GB):
      `uv run hf download ashynf/OpenFWI --repo-type dataset --include "*model*.npy" "*vel*.npy" --local-dir data/openfwi`
- [ ] Verification script (scratch): per family, count files + rows, assert map totals
      (Vel 30k ×4, Fault 54k ×4, Style 67k ×2 = 470k), assert shapes `(N,1,70,70)` — Fault
      velocity files may be `(N,70,70)`; if so, note it for Task 1's loader normalisation.
- [ ] Confirm `OpenFWIVelocityDataset(root, all_10_families)` constructs and reports len 470k.

### Task 1: Loader upgrades (family ids, per-family split, hflip, load peak)

**Files:** `flow_matching/openfwi.py`, `flow_matching/datasets.py`, `tests/test_openfwi.py`,
`tests/test_experiment_conf.py`.

- [ ] Pre-allocate the `(N,1,70,70)` array (replace `np.stack(rows)`); handle 3-D fault files if
      Task 0 found any.
- [ ] `family_ids` int8 array + `family_names` list on the dataset; `__getitem__` contract
      unchanged (`(x, 0)`).
- [ ] Per-family deterministic split in `OpenFWIDatasetConfig._split()`
      (seed = `crc32(family)`, 10% per family), preserving the return signature
      `(full, train_idx, val_idx)` so `build/build_val/held_out_targets` are untouched.
- [ ] `hflip: bool = False` config field; flip applied with p=0.5 on train samples only.
- [ ] Dataset fingerprint helper (families, file/map counts, val_fraction, seeds) logged into run
      config.
- [ ] Tests: split stability under family addition/removal; exact 10% per family; hflip
      distribution; 3-D file handling; fingerprint content.

### Task 2: Training loop speed + model presets

**Files:** `training/loop.py`, `flow_matching/models.py`, conf `model/dit_b.yaml` (+`dit_m.yaml`)
in all three experiments, loader kwargs where the DataLoader is built.

- [ ] Config-gated bf16 autocast in the train/val step (`training.precision: fp32|bf16`);
      optional `training.compile: bool`.
- [ ] `dit_b` (768/12/12, patch 4) and `dit_m` (512/8/8) presets; param counts asserted in a test.
- [ ] DataLoader: `num_workers=8`, `pin_memory`, `persistent_workers`.
- [ ] Parity smoke: FlatVel_A, 5 epochs, fp32 vs bf16 — val-loss curves agree within noise before
      bf16 is used anywhere definitive.

### Task 3: Per-family eval observability

**Files:** `training/loop.py` (val pass), experiment `run.py` eval hooks, viz in `openfwi.py`.

- [ ] Val loss logged per family (`val/loss/<family>`) alongside the global scalar.
- [ ] Per-eval sample grid stratified by family (fixed noise, 8×10 grid, family-labeled rows).
- [ ] Per-family energy distance vs held-out val at final eval (extends the existing
      energy-distance path).

### Task 4: Throughput calibration on the target GPU

Gate for the long runs — replaces the §6 extrapolations with measurements. Target card per spec
§6: RTX PRO 6000 Blackwell 96 GB recommended; 5090/H100 acceptable (32 GB cards → bs 128–192).

- [ ] Provision the pod (≥16 vCPU / ≥64 GB RAM); `uv sync`; verify Deepwave import + SDPA path.
- [ ] 1-epoch run of each experiment at `dit_b` + bf16 on all 10 families; record maps/s and
      VRAM; confirm bs 256 fits (else bs 128 + lr 1.4e-4 per spec §6).
- [ ] Update the three `openfwi_full.yaml`s: model `dit_b`, bs 256 / lr 2e-4 / n_epochs 90
      (decided), `hflip: true`, per-family eval on, `ckpt_every_epochs: 0` (wandb quota, spec
      §6.1). Commit with the measured numbers in the yaml comment (repo convention).

### Task 5: Train 0001 flow-matching prior (definitive)

- [ ] `experiments/0001_flow_matching/run.py experiment=openfwi_full` (est. 4–10 h on
      H100/PRO 6000, up to ~24 h on the current PRO 4500).
- [ ] Review: loss curves, per-family val losses converged and comparable; per-family grids show
      family-appropriate structure (faults in Fault rows, textures in Style rows).
- [ ] JOURNAL entry; best/final/EMA stay on the network volume, **final EMA + final raw uploaded
      as wandb artifacts** (EMA = inference copy, raw = warm-start copy; spec §6.1).

### Task 6: Train 0003 diffusion prior (parallel with Task 5, second pod)

- [ ] `experiments/0003_diffusion/run.py experiment=openfwi_full` (est. as Task 5).
- [ ] Same review + journal + backup gates as Task 5.

### Task 7: Train 0002 flow-map prior (esd_teacher distill from Task 5)

- [ ] `experiment=openfwi_full training=teacher training.teacher_ckpt=<0001 EMA ckpt>`
      (est. 12–24 h on H100/PRO 6000, ~1–2 days on 5090, up to ~4 days on PRO 4500).
      Diagonal stays on data (`data_fm=True` is hard-coded in `flow_matching/train.py`);
      teacher supplies off-diagonal target + warm start only — no config change needed (spec
      §9.3).
- [ ] Optional (deferred): from-scratch `mf` full run as scaling ablation.
- [ ] Same review + journal + backup gates.

### Task 8: Inversion benchmark set

**Files:** new `inversion/benchmark.py` + selection script, `data/inversion_bench/` assets,
`0004_inversion` conf (`target: <id>`), tests.

- [ ] Selection script: per family, rank val maps by total variation, take percentiles
      {5,25,50,75,95} × 4 draws → 200 targets; append legacy 6044 by Task-0 provenance → 201.
      Tag a fixed ~100-target `core: true` subset in the manifest for day-to-day work.
- [ ] Write `manifest.json` (ids, provenance, stats, tags, dataset fingerprint),
      `velocity/<id>.npy`, `previews/<id>.png` (fixed 1500–4500 m/s scale), per-family galleries.
- [ ] Seismic extraction (recommended): transiently download only the `data/seis` files containing
      selected rows (~30–40 GB), extract `seismic/<id>.npy` (5×1000×70, ~280 MB total), delete
      transients.
- [ ] `InversionBenchmark` loader (manifest-driven, no bulk-data dependency) + `target=<id>`
      support in 0004; velocity + previews + manifest committed to git; seismic → volume + wandb
      artifact.
- [ ] Tests: manifest↔file consistency; loader returns native m/s maps matching
      `held_out_targets` output for the same provenance while bulk data still exists.

### Task 9: Prior zoo + verification gates + deletion

- [ ] `docs/prior-zoo.md`: one row per definitive prior (run, volume path, wandb artifact ref,
      config, per-family metrics, thumbnail); 0004 configs point at these refs.
- [ ] Gate 1: each EMA checkpoint reloads **from its wandb artifact** on a clean path and
      samples 64 maps.
- [ ] Gate 2: per-family energy distance within agreed factor of the FlatVel-era baseline.
- [ ] Gate 3: flow_tilt + mfm_g inversion of benchmark id `flatvel_a_legacy_6044` reproduces
      journal numbers within seed noise, using only checkpoint + benchmark.
- [ ] Gate 4: benchmark assets (incl. seismic, ~290 MB) uploaded as a wandb artifact.
- [x] Delete `data/openfwi/*/data/` and `seis*` files (~160 GB reclaimed); **keep the ~8 GB of
      velocity maps** (spec §9.5). Done early (2026-07-06, user request): the Vel-family seismic
      was never read by anything. Task 8's benchmark seismic extraction transiently re-downloads
      just the specific files containing selected rows, as designed.

---

## Sequencing

```
Task 0 ──► Task 1 ──► Task 2 ──► Task 3 ──► Task 4 ──► Task 5 (0001) ──► Task 7 (0002)
                                                  └──► Task 6 (0003, parallel pod)
                                            Task 8 (benchmark, after Task 4's split freeze;
                                                    seismic + gates need Tasks 5–7) ──► Task 9
```

Critical path: 0 → 4 (code, ~1–2 days of work) then 5 → 7 (GPU, ~1–1.5 days wall-clock).
