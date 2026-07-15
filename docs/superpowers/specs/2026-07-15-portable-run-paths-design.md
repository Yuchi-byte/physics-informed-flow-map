# Portable run paths — same command on RunPod and local

**Status:** approved, not yet implemented
**Date:** 2026-07-15

## Problem

The four experiment frameworks hardcode `hydra.run.dir` to `/workspace/runs/...`. That path
exists only on RunPod, so every command in `experiments/README.md` fails on the local Arch
machine, and running there today means a per-invocation `hydra.run.dir=` override.

Checkpoints have the same disease one level up: `ckpt=/workspace/runs/0003_diffusion/
openfwi_2026-07-07T23-26-16Z/checkpoints/step_59_ema.pt` names a timestamped directory that
exists on exactly one machine, and it must be paired by hand with `model.hidden=768
model.depth=12 model.num_heads=12` — a shape the checkpoint already implies but does not record.

## Findings that shape the design

1. **The absolute path was never needed for persistence.** It arrived in `9b9f2e0`
   ("move run dirs to /workspace/runs/") justified as *"so logs land on the persistent network
   volume regardless of CWD"*. But the repo itself lives on the volume at
   `/workspace/physics-informed-flow-map` (`README.md:37`; the OpenFWI download in the
   2026-07-06 spec targets `/workspace/physics-informed-flow-map/data/openfwi`). A repo-relative
   `runs/` is therefore already on the volume. The absolute path bought CWD-independence only.
2. **CWD-independence is not load-bearing.** `dataset.data_dir: data/openfwi` is already
   repo-relative, so launching from the repo root is a standing requirement. `runs/` is the only
   path that opted out of it.
3. **`/workspace` appears in exactly four config files.** `packages/*/src/` contains zero
   references — the harness takes `run_dir` as a parameter (`start_run`, `run.py:352`) and
   `0004`'s `run.py:265` reads it from `HydraConfig`. No Python changes are needed.
4. **Checkpoints carry no provenance.** `Run.save_checkpoint` (`run.py:135`) writes
   `{"model": state_dict, "step": step, **meta}` — no run id, no config, no backbone shape.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Path strategy | Repo-relative, no env var, no anchor helper | The repo is on the volume in both environments; launching from the repo root is already required |
| `runs/` location | Inside the repo | Confirmed nothing requires it outside; still gitignored, still on the volume |
| Checkpoint addressing | Stable aliases pinned in `conf/prior/*.yaml` | Kills both the timestamp and the repeated `model.*` overrides |
| `prior=diffusion` denoiser | Default to `dit` | The definitive 0003 prior is a 768/12/12 DiT; ckpt, shape and denoiser then describe one coherent object |
| Existing `/workspace/runs` | Move into the repo's `runs/` | One location afterwards; directory names preserved so old citations stay greppable |
| `data/inversion_bench` | Regenerate per machine, never sync | Deterministic from `data/openfwi`, fingerprint-checked |

## Changes

### 1. Run directories become repo-relative

Drop the `/workspace` prefix in:

- `experiments/0001_flow_matching/conf/config.yaml:13`
- `experiments/0002_flow_map/conf/config.yaml:13`
- `experiments/0003_diffusion/conf/config.yaml:14`
- `experiments/0004_inversion/conf/config.yaml:24` (`hydra.run.dir`) and `:26` (`hydra.sweep.dir`)

`/workspace/runs/0004_inversion/...` → `runs/0004_inversion/...`. The timestamp/interpolation
suffix of each line is unchanged.

### 2. `checkpoints/` at the repo root

A new gitignored directory holding the definitive priors under names that never move:

```
checkpoints/
├── 0001_flow_matching_openfwi.pt
├── 0002_flow_map_openfwi.pt
├── 0003_diffusion_openfwi.pt
└── PROVENANCE.md          # committed; the .pt files are not
```

Naming: `<framework>_<dataset>.pt`. Retraining replaces the file and updates `PROVENANCE.md`;
the commands never change.

`PROVENANCE.md` is a committed table — alias → source run dir, step, wandb run — recovering the
link that `save_checkpoint` doesn't store. Without it a stable alias is an anonymous blob.

`.gitignore` gains, mirroring the existing `data/inversion_bench/` treatment at `.gitignore:63-65`:

```
checkpoints/*
!checkpoints/PROVENANCE.md
```

The glob matters: git cannot un-ignore a file inside an ignored *directory*, so `checkpoints/`
plus a negation would silently keep `PROVENANCE.md` untracked. Ignore the contents, not the
directory.

### 3. Prior groups pin ckpt + shape

Each `conf/prior/*.yaml` becomes `# @package _global_` (the convention `conf/experiment/*.yaml`
already uses) so it can set top-level `ckpt` and `model` alongside `prior.*`:

```yaml
# conf/prior/diffusion.yaml
# @package _global_
prior:
  name: diffusion
  denoiser_kind: dit
  num_train_timesteps: 1000
ckpt: checkpoints/0003_diffusion_openfwi.pt
model: { hidden: 768, depth: 12, num_heads: 12 }
```

`flow_matching` and `flow_map` follow the same shape (both 768/12/12, `patch_size: 4` is the
`ModelConfig` default). `none` sets `ckpt: ""` — there is no prior to load.

**Precedence is already correct.** `config.yaml` defaults order is
`_self_ → prior → method → dataset → obs → experiment`, so:

- `experiment/smoke.yaml`'s `ckpt: ""` still wins → untrained plumbing check intact.
- `experiment/flatfault.yaml`'s `model: { hidden: 320, depth: 8 }` and `ckpt: ""` still win →
  the 320/8 sweep is unaffected and keeps taking `ckpt=` from the CLI.
- A CLI `ckpt=` or `model.hidden=` overrides everything, as now.

`eval.py`'s per-entry `EvalEntry.ckpt`/`.model` mechanism is untouched.

### 4. Fix the README

`experiments/README.md:30` invokes `target=marmousi_fault05`, which does not exist: benchmark ids
are generated as `f"{family.lower()}_{seq:02d}"` (`benchmark.py:149`) plus the legacy
`flatvel_a_legacy_6044`, and Marmousi is unbuilt future work (2026-07-11 spec, §Task 5). Replace
with a real id in all three examples and drop the now-redundant `ckpt=`, `model.*` and
`prior.denoiser_kind=` overrides.

## Resulting command

```bash
uv run python experiments/0004_inversion/run.py \
  prior=diffusion method=dps method.misfit=ot \
  +guidance_strength=2 target=flatvel_a_legacy_6044 steps=200 n_samples=10
```

Identical on both machines.

## Migration

RunPod must go first — it is the only machine holding the trained priors, so there is nothing to
pull until `checkpoints/` exists there.

RunPod (once, next session): copy each definitive prior out of its run dir into `checkpoints/`
under its alias, record the source run in `PROVENANCE.md`, then
`mv /workspace/runs/* /workspace/physics-informed-flow-map/runs/`.

Local (once, afterwards):

```bash
uv run python -m physics_informed_flow_map.inversion.benchmark   # rebuild data/inversion_bench
rsync -a pod:/workspace/physics-informed-flow-map/checkpoints/ checkpoints/
```

Until the rsync lands, local runs are limited to `experiment=smoke` (untrained prior).

## Non-goals

- No env var (`PIFM_RUNS`) or `repo_root()` helper. Launching from the repo root is already
  required; indirection with one call site is unused flexibility.
- No config-time existence check on `ckpt`. `torch.load` already raises on a missing file, and
  `ckpt: ""` must stay legal for `smoke`. (A missing-but-nonempty `ckpt` failing late rather than
  early is a pre-existing wart, out of scope here.)
- No change to `data/openfwi` staging, the harness, or any reference package.
- Marmousi remains unbuilt; this spec only stops the README from advertising it.

## Verification

1. `uv run python experiments/0004_inversion/run.py experiment=smoke` — untrained prior still
   works, output lands in `runs/0004_inversion/...`, not `/workspace`.
2. The resulting command above with a real checkpoint in place — run dir under the repo,
   no `ckpt=`/`model.*` overrides needed.
3. `experiment=flatfault` composes with `model.hidden=320` intact (config-time check; no run).
4. `grep -rn "/workspace" experiments/` returns nothing.
