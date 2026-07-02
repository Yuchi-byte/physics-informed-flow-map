"""Larger-n inversion eval + flow-tilt posterior-sample sweep.

Two questions on existing FlatVel_A checkpoints: (1) do the rankings hold at more held-out maps
(n=8 vs 4), and (2) does giving the flow-tilt inverse process more posterior samples help
(n_samples 4 -> 8 -> 16, at the same per-sample step count, so cost scales with samples). Metrics
are the OpenFWI-scale [-1,1] MAE/RMSE/SSIM, expected across samples.

    uv run python notebooks/eval_n8_sample_sweep.py

Writes runs/inversion_eval/results_n8.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import DDPMScheduler

from physics_informed_flow_map.baselines import build_denoiser
from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.inversion import (
    ClassicalFWIModule,
    Evaluator,
    FlowTiltModule,
    REDDiffEqModule,
)

N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", 8))
STEPS = 200

FM = "runs/0001_flow_matching/2026-06-26T23-40-26Z/checkpoints/step_99_ema.pt"
FLOWMAP_T = os.environ.get(
    "FLOWMAP_CKPT", "runs/0002_flow_map/2026-06-27T01-44-09Z/checkpoints/step_99_ema.pt"
)
DIFF_DIT = "runs/0003_diffusion/2026-06-27T00-21-12Z/checkpoints/step_99_ema.pt"


def _root() -> Path:
    r = Path.cwd()
    while not (r / ".git").exists() and r != r.parent:
        r = r.parent
    return r


def main() -> None:
    os.chdir(_root())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  n_targets={N_TARGETS}")

    def load_flow(ckpt: str) -> torch.nn.Module:
        m = build_model(
            (1, 64, 64), None, DiTModelConfig(hidden=256, depth=6, num_heads=8, patch_size=4)
        ).to(dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
        return m.eval()

    fm, fmap = load_flow(FM), load_flow(FLOWMAP_T)
    dit = build_denoiser("dit", sample_size=64, channels=1).to(dev)
    dit.load_state_dict(torch.load(DIFF_DIT, map_location=dev, weights_only=False)["model"])
    dit.eval()
    sched = DDPMScheduler(num_train_timesteps=1000)  # type: ignore[no-untyped-call]

    ev = Evaluator.from_openfwi(["FlatVel_A"], N_TARGETS, device=dev)
    print(f"{len(ev.targets)} held-out targets\n")

    def named(m: object, name: str) -> object:
        m.name = name  # type: ignore[attr-defined]
        return m

    def tilt(prior: torch.nn.Module, ns: int) -> FlowTiltModule:
        return FlowTiltModule(prior, guidance=1.0, steps=STEPS, n_samples=ns, device=dev)

    modules = [
        # Core ranking at n_samples=4, more maps.
        named(tilt(fm, 4), "fm·tilt·n4"),
        named(tilt(fmap, 4), "flowmap_teacher·tilt·n4"),
        named(
            REDDiffEqModule(dit, sched, eta_data=0.3, eta_reg=0.1, t_denoise=100,
                            iters=STEPS, n_samples=4, device=dev),
            "red_diffeq·dit·n4",
        ),
        named(
            ClassicalFWIModule(reg="tv", reg_weight=1e-3, iters=2 * STEPS, lr=0.02,
                               n_samples=2, device=dev),
            "classical·tv",
        ),
        # Posterior-sample sweep: more samples to the flow inverse process.
        named(tilt(fm, 8), "fm·tilt·n8"),
        named(tilt(fm, 16), "fm·tilt·n16"),
        named(tilt(fmap, 8), "flowmap_teacher·tilt·n8"),
        named(tilt(fmap, 16), "flowmap_teacher·tilt·n16"),
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
        f"# Inversion eval — FlatVel_A, {N_TARGETS} held-out maps (+ flow sample sweep)",
        "",
        "Normalized [-1,1], expected across samples. n4/n8/n16 = posterior samples.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in sorted(rows, key=lambda d: d["mae_mean"]):
        cells = [str(r["method"])] + [
            f"{r[c]:.4g}" if c != "n_solves" else f"{r[c]:.0f}" for c in cols[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "results_n8.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir / 'results_n8.md'}")


if __name__ == "__main__":
    main()
