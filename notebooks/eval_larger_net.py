"""Did the larger network help? Compare 320/8 vs 256/6 priors on held-out FlatVel_A.

Loads the larger-net (hidden=320, depth=8) flow-matching and flow-map priors trained by the
overnight queue against the incumbent 256/6 flow-matching champion, plus classical-TV as a
prior-free reference, all tilted on the same n held-out maps at matched cost. Paths/dims are
env-overridable.

    uv run python notebooks/eval_larger_net.py

Writes runs/inversion_eval/results_larger_net.md.
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

N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", 8))
STEPS, N_SAMPLES = 200, 4
BIG_H, BIG_D = int(os.environ.get("BIG_HIDDEN", 320)), int(os.environ.get("BIG_DEPTH", 8))

FM_SMALL = "runs/0001_flow_matching/2026-06-26T23-40-26Z/checkpoints/step_99_ema.pt"
FM_LARGE = os.environ.get(
    "FM_LARGE", "runs/0001_flow_matching/2026-06-27T05-08-13Z/checkpoints/step_119_ema.pt"
)
FLOWMAP_LARGE = os.environ.get(
    "FLOWMAP_LARGE", "runs/0002_flow_map/2026-06-27T05-57-07Z/checkpoints/step_119_ema.pt"
)


def _root() -> Path:
    r = Path.cwd()
    while not (r / ".git").exists() and r != r.parent:
        r = r.parent
    return r


def main() -> None:
    os.chdir(_root())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  n_targets={N_TARGETS}  large={BIG_H}/{BIG_D}")

    def load_flow(ckpt: str, hidden: int, depth: int) -> torch.nn.Module:
        m = build_model(
            (1, 64, 64), None,
            DiTModelConfig(hidden=hidden, depth=depth, num_heads=8, patch_size=4),
        ).to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
        return m.eval()

    ev = Evaluator.from_openfwi(["FlatVel_A"], N_TARGETS, device=dev)
    print(f"{len(ev.targets)} held-out targets\n")

    def named(m: object, name: str) -> object:
        m.name = name  # type: ignore[attr-defined]
        return m

    def tilt(prior: torch.nn.Module) -> FlowTiltModule:
        return FlowTiltModule(prior, guidance=1.0, steps=STEPS, n_samples=N_SAMPLES, device=dev)

    modules = []
    for label, ckpt, h, d in [
        ("fm_small·256-6", FM_SMALL, 256, 6),
        ("fm_large·320-8", FM_LARGE, BIG_H, BIG_D),
        ("flowmap_large·320-8", FLOWMAP_LARGE, BIG_H, BIG_D),
    ]:
        if Path(ckpt).is_file():
            modules.append(named(tilt(load_flow(ckpt, h, d)), label + "·tilt"))
        else:
            print(f"[skip] {label}: {ckpt} not found")
    modules.append(
        named(
            ClassicalFWIModule(reg="tv", reg_weight=1e-3, iters=2 * STEPS, lr=0.02,
                               n_samples=2, device=dev),
            "classical·tv",
        )
    )

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
        f"# Larger-net comparison — FlatVel_A, {N_TARGETS} held-out maps",
        "",
        "Normalized [-1,1], expected across samples. 320/8 (~16M) vs 256/6 (~8M) priors.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in sorted(rows, key=lambda d: d["mae_mean"]):
        cells = [str(r["method"])] + [
            f"{r[c]:.4g}" if c != "n_solves" else f"{r[c]:.0f}" for c in cols[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "results_larger_net.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir / 'results_larger_net.md'}")


if __name__ == "__main__":
    main()
