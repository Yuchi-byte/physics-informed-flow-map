"""Evaluate a flow prior's inversion on any OpenFWI family (for the harder/combined-dataset runs).

Tilts one flow prior on a chosen family's held-out maps and reports it next to the prior-free
classical-TV reference, at matched cost. Parametrized by env so it serves the CurveVel_A and
combined-family priors the overnight queue trains.

    FAMILY=CurveVel_A FLOW_CKPT=runs/0001_flow_matching/<ts>/checkpoints/step_99_ema.pt \
        uv run python notebooks/eval_family.py

Writes runs/inversion_eval/results_<FAMILY>.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.inversion import (
    ClassicalFWIModule,
    Evaluator,
    FlowTiltModule,
)

FAMILY = os.environ.get("FAMILY", "CurveVel_A")
FLOW_CKPT = os.environ["FLOW_CKPT"]  # required: the prior trained on this family
HIDDEN = int(os.environ.get("HIDDEN", 256))
DEPTH = int(os.environ.get("DEPTH", 6))
N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", 8))
STEPS, N_SAMPLES = 200, 4


def _root() -> Path:
    r = Path.cwd()
    while not (r / ".git").exists() and r != r.parent:
        r = r.parent
    return r


def main() -> None:
    os.chdir(_root())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  family={FAMILY}  dims={HIDDEN}/{DEPTH}  n={N_TARGETS}")

    prior = build_model(
        (1, 64, 64), None,
        DiTModelConfig(hidden=HIDDEN, depth=DEPTH, num_heads=8, patch_size=4),
    ).to(dev)
    prior.load_state_dict(
        torch.load(FLOW_CKPT, map_location=dev, weights_only=False)["model"]
    )
    prior.eval()

    ev = Evaluator.from_openfwi([FAMILY], N_TARGETS, device=dev)
    print(f"{len(ev.targets)} held-out {FAMILY} targets\n")

    def named(m: object, name: str) -> object:
        m.name = name  # type: ignore[attr-defined]
        return m

    modules = [
        named(
            FlowTiltModule(prior, guidance=1.0, steps=STEPS, n_samples=N_SAMPLES, device=dev),
            f"fm·tilt·{FAMILY}",
        ),
        named(
            FlowTiltModule(prior, guidance=0.0, steps=STEPS, n_samples=N_SAMPLES, device=dev),
            f"fm·prior·{FAMILY}",
        ),
        named(
            ClassicalFWIModule(reg="tv", reg_weight=1e-3, iters=2 * STEPS, lr=0.02,
                               n_samples=2, device=dev),
            "classical·tv",
        ),
    ]

    rows = []
    for m in modules:
        torch.manual_seed(0)
        stats = ev.evaluate(m)  # type: ignore[arg-type]
        print(stats, "\n")
        rows.append({"method": m.name, **stats.summary()})  # type: ignore[attr-defined,dict-item]

    out_dir = Path("runs/inversion_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["method", "mae_mean", "rmse_mean", "ssim_mean", "misfit_mean", "n_solves"]
    lines = [
        f"# Inversion eval — {FAMILY}, {N_TARGETS} held-out maps",
        "",
        "Normalized [-1,1], expected across samples.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in sorted(rows, key=lambda d: d["mae_mean"]):
        cells = [str(r["method"])] + [
            f"{r[c]:.4g}" if c != "n_solves" else f"{r[c]:.0f}" for c in cols[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / f"results_{FAMILY}.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir / f'results_{FAMILY}.md'}")


if __name__ == "__main__":
    main()
