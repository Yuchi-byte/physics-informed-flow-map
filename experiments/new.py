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

    uv run python experiments/{dirname}/run.py [variant] [key=value ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from physics_informed_flow_map.experiment import Config, start_run


class {cls}(Config):
    seed: int = 0
    # TODO: declare typed knobs here.
    gate: float = 0.0  # verdict threshold, asserted in code


VARIANTS: dict[str, dict[str, object]] = {{
    "default": {{}},
    "smoke": {{}},  # trivial budget for a fast end-to-end plumbing check
}}


def main() -> None:
    argv = sys.argv[1:]
    variant = argv[0] if argv and "=" not in argv[0] else "default"
    overrides = argv[1:] if argv and "=" not in argv[0] else argv
    cfg = {cls}.resolve(VARIANTS[variant], overrides)

    run = start_run(Path(__file__).parent, cfg.dump())
    # TODO: do the work; call run.log(**metrics) per step.
    run.finish("todo")


if __name__ == "__main__":
    main()
'''

REPORT_STUB = """\
# {number} — {title}

Status: open

## Hypothesis

<one sentence to be proven>

## Setup

`run.py [variant]` — config, loop steps, stack.

## Results

Cite run directories under `runs/{dirname}/`; quote numbers from
`metrics.jsonl` / `result.json`.

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
    target.mkdir()

    (target / "run.py").write_text(
        RUN_STUB.format(title=title, dirname=dirname, cls=cls)
    )
    (target / "report.md").write_text(
        REPORT_STUB.format(number=f"{number:04d}", title=title, dirname=dirname)
    )
    print(f"scaffolded experiments/{dirname}/")
    print(f"  edit experiments/{dirname}/run.py")


if __name__ == "__main__":
    main()
