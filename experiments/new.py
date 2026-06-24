"""Scaffold the next experiment framework.

    uv run python experiments/new.py "short title of the idea"

Allocates ``experiments/NNNN_slug/`` (NNNN = next free number, slug from the
title) and writes a runnable ``run.py`` stub plus a ``report.md`` skeleton.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "experiment"


def next_number() -> int:
    existing = [
        int(p.name[:4])
        for p in EXPERIMENTS.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}_.+", p.name)
    ]
    return (max(existing) + 1) if existing else 1


RUN_STUB = '''\
"""{title}

    uv run python experiments/{dirname}/run.py                    # default
    uv run python experiments/{dirname}/run.py experiment=smoke
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field

from physics_informed_flow_map.experiment import Config, start_run

EXPERIMENT = "{dirname}"


class {cls}(Config):
    seed: int = 0
    # TODO: declare typed knobs here.
    gate: float = Field(0.0)  # verdict threshold, asserted in code


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = {cls}.from_dictconfig(dcfg)
    assert isinstance(cfg, {cls})

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())
    # TODO: do the work; call run.log(**metrics) per step.
    run.finish("todo")


if __name__ == "__main__":
    main()
'''

CONFIG_STUB = """\
defaults:
  - _self_
  - experiment: default

seed: 0
gate: 0.0

hydra:
  run:
    dir: runs/{dirname}/${{now:%Y-%m-%dT%H-%M-%SZ}}
  job:
    chdir: false
"""

EXPERIMENT_DEFAULT_STUB = """\
# @package _global_
# TODO: default-variant overrides.
"""

EXPERIMENT_SMOKE_STUB = """\
# @package _global_
# Trivial budget for a fast end-to-end plumbing check (no strength claim).
gate: 1000000000.0
"""

REPORT_STUB = """\
# {number} — {title}

Status: open

## Hypothesis

<one sentence to be proven>

## Setup

`run.py [experiment=<variant>] [key=value ...]` — config, loop steps, stack.

## Results

Cite run directories under `runs/{dirname}/`; quote numbers + the verdict from the
wandb run (config / metrics / summary). Checkpoints live in `<run>/checkpoints/`.

## Decision

Adopted / Falsified / Parked; what changes. Mirror the verdict line to
`../JOURNAL.md`.
"""


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        sys.exit('usage: uv run python experiments/new.py "short title"')

    title = " ".join(sys.argv[1:]).strip()
    number = next_number()
    dirname = f"{number:04d}_{slugify(title)}"
    cls = "".join(w.capitalize() for w in slugify(title).split("_")) + "Config"

    target = EXPERIMENTS / dirname
    if target.exists():
        sys.exit(f"refusing to overwrite existing {target}")
    (target / "conf" / "experiment").mkdir(parents=True)

    (target / "run.py").write_text(
        RUN_STUB.format(title=title, dirname=dirname, cls=cls)
    )
    (target / "conf" / "config.yaml").write_text(CONFIG_STUB.format(dirname=dirname))
    (target / "conf" / "experiment" / "default.yaml").write_text(
        EXPERIMENT_DEFAULT_STUB
    )
    (target / "conf" / "experiment" / "smoke.yaml").write_text(EXPERIMENT_SMOKE_STUB)
    (target / "report.md").write_text(
        REPORT_STUB.format(number=f"{number:04d}", title=title, dirname=dirname)
    )
    print(f"scaffolded experiments/{dirname}/")
    print(f"  edit experiments/{dirname}/run.py and conf/")


if __name__ == "__main__":
    main()
