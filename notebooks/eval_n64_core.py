"""High-n core comparison to shrink the error bars (n=64 held-out FlatVel_A maps).

The four core inverters at n_samples=4, matched ~800-solve budget: flow-matching tilt, the
teacher-distilled flow-map tilt, RED-DiffEq, and classical-TV. With n=64 the standard error of
the mean is ~0.005 (vs ~0.014 at n=8), enough to actually rank methods whose gaps are ~0.01.
Results are written after each method, so a partial run still leaves a usable table.

    uv run python notebooks/eval_n64_core.py    # EVAL_N_TARGETS overridable
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

N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", 64))
STEPS, N_SAMPLES = 200, 4

FM = "runs/0001_flow_matching/2026-06-26T23-40-26Z/checkpoints/step_99_ema.pt"
FLOWMAP_T = "runs/0002_flow_map/2026-06-27T01-44-09Z/checkpoints/step_99_ema.pt"
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

    modules = [
        named(FlowTiltModule(fm, guidance=1.0, steps=STEPS, n_samples=N_SAMPLES, device=dev),
              "fm·tilt"),
        named(FlowTiltModule(fmap, guidance=1.0, steps=STEPS, n_samples=N_SAMPLES, device=dev),
              "flowmap_teacher·tilt"),
        named(REDDiffEqModule(dit, sched, eta_data=0.3, eta_reg=0.1, t_denoise=100,
                              iters=STEPS, n_samples=N_SAMPLES, device=dev), "red_diffeq·dit"),
        named(ClassicalFWIModule(reg="tv", reg_weight=1e-3, iters=2 * STEPS, lr=0.02,
                                 n_samples=2, device=dev), "classical·tv"),
    ]

    out = Path("runs/inversion_eval/results_n64.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in modules:
        torch.manual_seed(0)
        stats = ev.evaluate(m)  # type: ignore[arg-type]
        print(stats, "\n", flush=True)
        a = stats.agg
        rows.append(
            f"| {m.name} | {a['mae_mean']:.4g} ± {a['mae_std']:.2g} | "  # type: ignore[attr-defined]
            f"{a['rmse_mean']:.4g} | {a['ssim_mean']:.4g} ± {a['ssim_std']:.2g} | "
            f"{a['misfit_mean']:.4g} |"
        )
        header = [
            f"# Core inversion comparison — FlatVel_A, {N_TARGETS} held-out maps",
            "",
            f"Normalized [-1,1], expected across samples (n_samples={N_SAMPLES}). ± = std across "
            f"maps; SEM ≈ std/√{N_TARGETS}.",
            "",
            "| method | mae ± std | rmse | ssim ± std | misfit |",
            "|---|---|---|---|---|",
        ]
        out.write_text("\n".join(header + rows) + "\n")  # incremental: rewrite after each method
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
