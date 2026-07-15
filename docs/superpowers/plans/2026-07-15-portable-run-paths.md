# Portable Run Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `experiments/` command run unchanged on both RunPod and the local machine by dropping the `/workspace` hardcoding and addressing checkpoints by stable alias.

**Architecture:** Config-only change. `hydra.run.dir` becomes repo-relative in all four frameworks (the repo already lives on the RunPod network volume, so `runs/` is equally durable there). `0004`'s `conf/prior/*.yaml` groups switch to `# @package _global_` so each pins its own checkpoint alias *and* backbone shape — the two things that must agree. No Python source changes: the harness already takes `run_dir` as a parameter.

**Tech Stack:** Hydra 1.3 config composition, OmegaConf, pydantic v2 (`extra="forbid"`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-portable-run-paths-design.md`

## Global Constraints

- Run from the repo root: `/home/markhaoxiang/Projects/physics-informed-flow-map`. All relative paths below assume it.
- Tests live in `packages/physics-informed-flow-map/tests/`; run with `uv run pytest`.
- mypy is `strict = true` (`disallow_any_explicit = false`, so explicit `Any` is allowed). Every test function needs `-> None` and annotated parameters.
- Never `git add -A`. Stage the exact files listed in each commit step.
- Pre-commit runs ruff, mypy and the full pytest suite on every commit; expect the hook output before the commit line.
- Checkpoint aliases are `checkpoints/<framework>_<dataset>.pt`, exactly: `0001_flow_matching_openfwi.pt`, `0002_flow_map_openfwi.pt`, `0003_diffusion_openfwi.pt`.
- All three definitive priors are 768/12/12 DiTs with the default `patch_size: 4`.

---

### Task 1: Repo-relative run directories

**Files:**
- Create: `packages/physics-informed-flow-map/tests/test_experiment_paths.py`
- Modify: `experiments/0001_flow_matching/conf/config.yaml:13`
- Modify: `experiments/0002_flow_map/conf/config.yaml:13`
- Modify: `experiments/0003_diffusion/conf/config.yaml:14`
- Modify: `experiments/0004_inversion/conf/config.yaml:24,26`

**Interfaces:**
- Consumes: nothing.
- Produces: the guarantee that `hydra.run.dir` starts with `runs/<framework>/` for all four frameworks. Task 4's README rewrite depends on it.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_experiment_paths.py`:

```python
"""Run dirs are repo-relative, so one command works on RunPod and locally alike.

The repo sits on the RunPod network volume, so a relative ``runs/`` is as durable there
as the old absolute ``/workspace/runs`` was — and it resolves locally too.
"""

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
FRAMEWORKS = ["0001_flow_matching", "0002_flow_map", "0003_diffusion", "0004_inversion"]


def _hydra_node(framework: str) -> Any:
    raw = OmegaConf.load(REPO / "experiments" / framework / "conf" / "config.yaml")
    # resolve=False: ${now:...} has no resolver outside a running Hydra app.
    return OmegaConf.to_container(raw, resolve=False)["hydra"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_run_dir_is_repo_relative(framework: str) -> None:
    run_dir = _hydra_node(framework)["run"]["dir"]
    assert run_dir.startswith(f"runs/{framework}/"), run_dir


def test_multirun_sweep_dir_is_repo_relative() -> None:
    sweep_dir = _hydra_node("0004_inversion")["sweep"]["dir"]
    assert sweep_dir.startswith("runs/0004_inversion/"), sweep_dir
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_paths.py -v`

Expected: 5 FAILED, each asserting on a string starting `/workspace/runs/...`.

- [ ] **Step 3: Drop the `/workspace` prefix in all four configs**

`experiments/0001_flow_matching/conf/config.yaml` line 13:

```yaml
    dir: runs/0001_flow_matching/${dataset.name}_${now:%Y-%m-%dT%H-%M-%SZ}
```

`experiments/0002_flow_map/conf/config.yaml` line 13:

```yaml
    dir: runs/0002_flow_map/${dataset.name}_${training.distillation_type}_${now:%Y-%m-%dT%H-%M-%SZ}
```

`experiments/0003_diffusion/conf/config.yaml` line 14:

```yaml
    dir: runs/0003_diffusion/${dataset.name}_${now:%Y-%m-%dT%H-%M-%SZ}
```

`experiments/0004_inversion/conf/config.yaml` lines 24 and 26 (`hydra.run.dir` and `hydra.sweep.dir` — both):

```yaml
hydra:
  run:
    dir: runs/0004_inversion/${prior.name}_${method.name}_${now:%Y-%m-%dT%H-%M-%SZ}
  sweep:  # keep --multirun output under the gitignored runs/, not a root multirun/
    dir: runs/0004_inversion/${prior.name}_${method.name}_${now:%Y-%m-%dT%H-%M-%SZ}
    subdir: ${hydra.job.num}
  job:
    chdir: false
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_experiment_paths.py -v`

Expected: 5 passed.

- [ ] **Step 5: Verify no `/workspace` remains in experiments/**

Run: `grep -rn "/workspace" experiments/ --include=*.yaml`

Expected: no output (exit 1).

- [ ] **Step 6: Verify the smoke run writes under the repo**

Run: `WANDB_MODE=disabled uv run python experiments/0004_inversion/run.py experiment=smoke`

Expected: exits 0, and `ls runs/0004_inversion/` shows a fresh `flow_matching_flow_tilt_<stamp>` directory. Nothing is written to `/workspace`.

- [ ] **Step 7: Commit**

```bash
git add packages/physics-informed-flow-map/tests/test_experiment_paths.py \
        experiments/0001_flow_matching/conf/config.yaml \
        experiments/0002_flow_map/conf/config.yaml \
        experiments/0003_diffusion/conf/config.yaml \
        experiments/0004_inversion/conf/config.yaml
git commit -m "fix(conf): repo-relative run dirs so experiments run off RunPod too

The repo already lives on the network volume, so /workspace/runs bought
CWD-independence, not persistence — and data_dir never had it either."
```

---

### Task 2: The `checkpoints/` convention

**Files:**
- Create: `checkpoints/PROVENANCE.md`
- Modify: `.gitignore` (append to the "Outputs and checkpoints" block, currently lines 41-50)

**Interfaces:**
- Consumes: nothing.
- Produces: `checkpoints/` — gitignored for `*.pt`, tracked for `PROVENANCE.md`. Task 3's prior groups point their `ckpt:` at paths inside it.

- [ ] **Step 1: Add the ignore rules**

In `.gitignore`, immediately after the `runs/` / `multirun/` / `reward_experiments/` lines (currently line 48), insert:

```
# Definitive prior checkpoints, addressed by stable alias from conf/prior/*.yaml. The
# weights are pulled from a training run dir (rsync between machines); only the
# alias->run mapping is tracked. Ignore the contents, not the directory: git cannot
# un-ignore a file inside an ignored directory.
checkpoints/*
!checkpoints/PROVENANCE.md
```

- [ ] **Step 2: Create the provenance table**

Create `checkpoints/PROVENANCE.md`:

```markdown
# Checkpoint provenance

`conf/prior/*.yaml` addresses each definitive prior by a stable alias, so commands survive
retraining and move between machines unchanged. The `.pt` files are gitignored — copy them
here from the source run and keep this table current. `Run.save_checkpoint` stores only
`{"model", "step"}`, so without this table an alias is an anonymous blob.

Fetch on a fresh machine:

    rsync -a pod:/workspace/physics-informed-flow-map/checkpoints/ checkpoints/

| alias | source run | step | backbone |
|---|---|---|---|
| `0001_flow_matching_openfwi.pt` | `runs/0001_flow_matching/openfwi_2026-07-07T11-19-11Z` | `step_89_ema` | 768/12/12 DiT |
| `0002_flow_map_openfwi.pt` | `runs/0002_flow_map/openfwi_mf_2026-07-08T10-23-48Z` | `step_89_ema` | 768/12/12 DiT |
| `0003_diffusion_openfwi.pt` | `runs/0003_diffusion/openfwi_2026-07-07T23-26-16Z` | `step_59_ema` | 768/12/12 DiT (eps-denoiser) |

Each source run dir holds the wandb run id in its manifest; the run dirs live on the RunPod
network volume under the repo's `runs/`.

Retraining a prior: copy the new `step_<N>_ema.pt` over the alias, update the row, and update
`model:` in the matching `conf/prior/*.yaml` if the backbone shape changed.
```

- [ ] **Step 3: Verify git ignores the weights but tracks the table**

Run:

```bash
touch checkpoints/0003_diffusion_openfwi.pt
git check-ignore -v checkpoints/0003_diffusion_openfwi.pt
git check-ignore -v checkpoints/PROVENANCE.md; echo "PROVENANCE exit=$?"
rm checkpoints/0003_diffusion_openfwi.pt
```

Expected: the first prints a `.gitignore:<line>:checkpoints/*` match; the second prints nothing and reports `PROVENANCE exit=1` (not ignored).

- [ ] **Step 4: Commit**

```bash
git add .gitignore checkpoints/PROVENANCE.md
git commit -m "feat(checkpoints): stable-alias directory + provenance table

save_checkpoint records only {model, step}, so an alias severs the link to
the run that made it unless the mapping is tracked alongside it."
```

---

### Task 3: Prior groups pin checkpoint and backbone

**Files:**
- Create: `packages/physics-informed-flow-map/tests/test_inversion_conf.py`
- Modify: `experiments/0004_inversion/conf/prior/flow_matching.yaml`
- Modify: `experiments/0004_inversion/conf/prior/flow_map.yaml`
- Modify: `experiments/0004_inversion/conf/prior/diffusion.yaml`
- Modify: `experiments/0004_inversion/conf/prior/none.yaml`

**Interfaces:**
- Consumes: the alias paths from Task 2's `checkpoints/PROVENANCE.md` table.
- Produces: `prior=<name>` alone resolves `ckpt` and `model.{hidden,depth,num_heads}`. Task 4's README examples depend on it.

**Context the implementer needs:**

`conf/config.yaml`'s defaults order is `_self_ → prior → method → dataset → obs → experiment`. `_self_` is *first*, so a prior group overrides the base `model:` block; `experiment` is *last*, so `smoke.yaml`'s `ckpt: ""` and `flatfault.yaml`'s 320/8 still win. That precedence is the point — do not reorder the defaults list.

`ckpt` and `model` are top-level fields of `InversionConfig`, not members of `PriorConfig`. A file under `conf/prior/` defaults to package `prior`, so it can only set `prior.*`. Adding `# @package _global_` (the convention `conf/experiment/*.yaml` already uses) lets it set top-level keys, with `prior.*` nested explicitly. `PriorConfig` is `extra="forbid"` — a stray top-level `ckpt:` left un-nested inside package `prior` becomes `prior.ckpt` and raises a `ValidationError`.

- [ ] **Step 1: Write the failing test**

Create `packages/physics-informed-flow-map/tests/test_inversion_conf.py`:

```python
"""0004 prior groups pin their checkpoint alias and backbone shape.

`prior=<name>` must resolve the ckpt and the DiT shape together — they have to agree, and
splitting them across the CLI is how you load a 768/12/12 checkpoint into a 256/6/8 model.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "0004_inversion"
CONF = EXP / "conf"


def _load_run_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inversion_run", EXP / "run.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Pydantic resolves InversionConfig's forward refs (EvalEntry) via sys.modules; without
    # registering first, model_rebuild() leaves EvalConfig undefined and from_dictconfig raises.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _compose(overrides: list[str]) -> Any:
    cfg_cls = _load_run_module().InversionConfig
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        dcfg = compose(config_name="config", overrides=overrides)
    return cfg_cls.from_dictconfig(dcfg)


@pytest.mark.parametrize(
    "prior,method,ckpt",
    [
        ("flow_matching", "flow_tilt", "checkpoints/0001_flow_matching_openfwi.pt"),
        ("flow_map", "mfm_g", "checkpoints/0002_flow_map_openfwi.pt"),
        ("diffusion", "dps", "checkpoints/0003_diffusion_openfwi.pt"),
    ],
)
def test_prior_group_pins_ckpt_and_shape(prior: str, method: str, ckpt: str) -> None:
    cfg = _compose([f"prior={prior}", f"method={method}"])
    assert cfg.ckpt == ckpt
    assert (cfg.model.hidden, cfg.model.depth, cfg.model.num_heads) == (768, 12, 12)
    assert cfg.model.patch_size == 4  # ModelConfig default survives the group merge


def test_diffusion_prior_defaults_to_dit_denoiser() -> None:
    cfg = _compose(["prior=diffusion", "method=dps"])
    assert cfg.prior.denoiser_kind == "dit"
    assert cfg.prior.num_train_timesteps == 1000


def test_prior_none_pins_no_ckpt() -> None:
    cfg = _compose(["prior=none", "method=classical_fwi"])
    assert cfg.ckpt == ""


def test_smoke_still_runs_an_untrained_prior() -> None:
    # experiment groups load after prior, so smoke's ckpt: "" blanks the alias.
    cfg = _compose(["experiment=smoke"])
    assert cfg.ckpt == ""


def test_flatfault_keeps_its_320_8_backbone() -> None:
    cfg = _compose(["experiment=flatfault"])
    assert (cfg.model.hidden, cfg.model.depth) == (320, 8)
    assert cfg.ckpt == ""


def test_cli_ckpt_override_beats_the_alias() -> None:
    cfg = _compose(["prior=diffusion", "method=dps", "ckpt=runs/legacy/step_9.pt"])
    assert cfg.ckpt == "runs/legacy/step_9.pt"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_inversion_conf.py -v`

Expected: 4 FAILED, 4 PASSED. The failures are the three `test_prior_group_pins_ckpt_and_shape` cases (`assert '' == 'checkpoints/...'`; the shape is `(256, 6, 8)` today) plus `test_diffusion_prior_defaults_to_dit_denoiser` (`'unet' != 'dit'`). The four passing ones — `prior_none`, `smoke`, `flatfault`, `cli_ckpt_override` — are regression guards that must stay green throughout.

- [ ] **Step 3: Rewrite the four prior groups**

`experiments/0004_inversion/conf/prior/flow_matching.yaml` — replace the whole file:

```yaml
# @package _global_
prior:
  name: flow_matching  # flow-matching prior (0001); DiT via flow_matching.build_model
ckpt: checkpoints/0001_flow_matching_openfwi.pt
model: { hidden: 768, depth: 12, num_heads: 12 }
```

`experiments/0004_inversion/conf/prior/flow_map.yaml` — replace the whole file:

```yaml
# @package _global_
prior:
  name: flow_map  # Meta-Flow-Map prior (0002); same DiT backbone, time-conditional posterior
ckpt: checkpoints/0002_flow_map_openfwi.pt
model: { hidden: 768, depth: 12, num_heads: 12 }
```

`experiments/0004_inversion/conf/prior/diffusion.yaml` — replace the whole file:

```yaml
# @package _global_
prior:
  name: diffusion  # diffusion prior (0003); a diffusers denoiser + DDPM scheduler
  denoiser_kind: dit  # unet | dit (the flow priors' DiT backbone as an eps-denoiser)
  num_train_timesteps: 1000  # DDPM schedule the prior was trained on
ckpt: checkpoints/0003_diffusion_openfwi.pt
model: { hidden: 768, depth: 12, num_heads: 12 }
```

`experiments/0004_inversion/conf/prior/none.yaml` — replace the whole file:

```yaml
# @package _global_
prior:
  name: none  # no learned prior — the classical (hand-regularised) FWI baseline
ckpt: ""  # nothing to load
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest packages/physics-informed-flow-map/tests/test_inversion_conf.py -v`

Expected: 8 passed.

- [ ] **Step 5: Verify the full suite still passes**

Run: `uv run pytest`

Expected: all pass. If `test_inversion_eval.py` fails, a sweep entry is picking up the new top-level `ckpt` default — report it rather than papering over it; `EvalEntry.ckpt` is meant to be independent.

- [ ] **Step 6: Commit**

```bash
git add packages/physics-informed-flow-map/tests/test_inversion_conf.py \
        experiments/0004_inversion/conf/prior/flow_matching.yaml \
        experiments/0004_inversion/conf/prior/flow_map.yaml \
        experiments/0004_inversion/conf/prior/diffusion.yaml \
        experiments/0004_inversion/conf/prior/none.yaml
git commit -m "feat(0004): prior groups pin their checkpoint alias and backbone shape

ckpt and model shape have to agree; carrying both on the CLI is how a
768/12/12 checkpoint gets loaded into a 256/6/8 model."
```

---

### Task 4: Fix the README commands

**Files:**
- Modify: `experiments/README.md:8-31` (the "Running the scripts" section)

**Interfaces:**
- Consumes: Task 1's repo-relative run dirs and Task 3's prior-group aliases.
- Produces: nothing downstream.

**Context the implementer needs:**

Two of the three current examples are broken independently of the `/workspace` paths:

1. `target=marmousi_fault05` does not exist. Benchmark ids are generated as `f"{family.lower()}_{seq:02d}"` (`packages/physics-informed-flow-map/src/physics_informed_flow_map/inversion/benchmark.py:149`), plus the legacy `flatvel_a_legacy_6044` (`:194`). Marmousi is unbuilt future work (`docs/superpowers/specs/2026-07-11-benchmark-hardening-design.md:168`).
2. `+guidance_strength=2` raises `ValidationError`. The `+` creates a **top-level** `guidance_strength` key, and `InversionConfig` is `extra="forbid"`; the knob lives on `MethodConfig`. Verified: `+guidance_strength=2` → `Extra inputs are not permitted`, `method.guidance_strength=2` → `2.0`. Use the dotted form and no `+`.

- [ ] **Step 1: Confirm the target id exists before documenting it**

Run:

```bash
uv run python -m physics_informed_flow_map.inversion.benchmark
uv run python -c "
from physics_informed_flow_map.inversion import InversionBenchmark
b = InversionBenchmark('data/inversion_bench')
print('flatvel_a_legacy_6044' in b)
"
```

Expected: the builder prints a `N targets (M core) -> data/inversion_bench` line, then `True`.

If the builder fails, stop and report — the README must not document a target that cannot load.

- [ ] **Step 2: Replace the "Running the scripts" section**

In `experiments/README.md`, replace lines 8-31 (from `## Running the scripts` through the closing fence of the diffusion example) with:

````markdown
## Running the scripts

Run from the repo root — `runs/`, `data/openfwi` and `checkpoints/` are all resolved relative
to it. Each `prior=` group pins its own checkpoint alias and backbone shape
(`checkpoints/PROVENANCE.md` maps each alias to the run that produced it), so a target and a
method are all an inversion needs.

Flow-map prior, OT misfit — a fast plumbing check at 5 steps:

```
uv run python experiments/0004_inversion/run.py \
  prior=flow_map method=flow_tilt method.misfit=ot target=flatvel_a_legacy_6044 steps=5
```

Flow-matching prior, L2 misfit:

```
uv run python experiments/0004_inversion/run.py \
  prior=flow_matching method=flow_tilt method.misfit=l2 \
  target=flatvel_a_legacy_6044 steps=200 n_samples=10
```

Diffusion prior, canonical DPS:

```
uv run python experiments/0004_inversion/run.py \
  prior=diffusion method=dps method.misfit=ot method.guidance_strength=2 \
  target=flatvel_a_legacy_6044 steps=200 n_samples=10
```

Target ids come from `data/inversion_bench/manifest.json` — rebuild it on a fresh machine with
`uv run python -m physics_informed_flow_map.inversion.benchmark` (deterministic from
`data/openfwi`; never sync it between machines). Pass `ckpt=` and `model.hidden=…` explicitly
only to override a group's pinned prior, e.g. for a legacy 320/8 checkpoint.

Guidance strength is `method.guidance_strength=…`, not `+guidance_strength=…`: the config is
`extra="forbid"`, so a `+`-added top-level key is a `ValidationError`, not a silent no-op.
````

- [ ] **Step 3: Verify no stale references survive**

Run: `grep -rn "marmousi\|/workspace" experiments/README.md`

Expected: no output (exit 1).

- [ ] **Step 4: Verify a documented command composes**

Run:

```bash
uv run python experiments/0004_inversion/run.py \
  prior=diffusion method=dps method.misfit=ot method.guidance_strength=2 \
  target=flatvel_a_legacy_6044 steps=200 n_samples=10 --cfg job 2>&1 | head -30
```

Expected: Hydra prints the composed config (`--cfg job` exits without running) showing `ckpt: checkpoints/0003_diffusion_openfwi.pt`, `denoiser_kind: dit`, and `hidden: 768`.

`--cfg job` prints without constructing `InversionConfig`, so it cannot catch an `extra="forbid"` rejection. The real guard for that is `test_cli_ckpt_override_beats_the_alias` in Task 3 plus the smoke run in Task 1; a full run of this exact command has to wait for the checkpoint to land (see Migration).

- [ ] **Step 5: Commit**

```bash
git add experiments/README.md
git commit -m "docs(experiments): commands that work on both machines

target=marmousi_fault05 never existed — Marmousi is unbuilt (2026-07-11 spec
Task 5), so the example failed before it reached the missing checkpoint."
```

---

## Post-implementation: migration (human, not agent)

The agent cannot do these — they need the pod.

**On RunPod, next session:**

```bash
cd /workspace/physics-informed-flow-map
mkdir -p checkpoints
cp runs/0001_flow_matching/openfwi_2026-07-07T11-19-11Z/checkpoints/step_89_ema.pt checkpoints/0001_flow_matching_openfwi.pt
cp runs/0002_flow_map/openfwi_mf_2026-07-08T10-23-48Z/checkpoints/step_89_ema.pt   checkpoints/0002_flow_map_openfwi.pt
cp runs/0003_diffusion/openfwi_2026-07-07T23-26-16Z/checkpoints/step_59_ema.pt     checkpoints/0003_diffusion_openfwi.pt
mv /workspace/runs/* /workspace/physics-informed-flow-map/runs/
```

(The `cp` source paths assume the `mv` has already happened, or read from `/workspace/runs/...`
if not — do the `mv` first if the run dirs are still outside the repo.)

**Locally, afterwards:**

```bash
rsync -a pod:/workspace/physics-informed-flow-map/checkpoints/ checkpoints/
```

Until that rsync lands, local runs are limited to `experiment=smoke` — every other `prior=`
now points at a checkpoint alias that does not exist yet, and `torch.load` will raise
`FileNotFoundError`.
