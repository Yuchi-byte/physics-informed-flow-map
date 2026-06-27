"""Batch inversion evaluation across all camp-A inverters on held-out FlatVel_A maps.

The script counterpart to ``inversion_eval.ipynb`` — robust for long unattended runs. Scores
every :class:`InversionModule` on the same held-out targets and Deepwave operator through the
shared ``physics_informed_flow_map.inversion`` harness, at matched forward-solve budget
(``n_solves`` ~ STEPS * N_SAMPLES for every method). Metrics are the OpenFWI-convention
MAE/RMSE/SSIM on the ``[-1, 1]`` velocity, expected across posterior samples.

    uv run python notebooks/run_inversion_eval.py

Prints a table and writes it to ``runs/inversion_eval/results.md`` (gitignored).
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
    DiffusionDPSModule,
    Evaluator,
    FlowTiltModule,
    REDDiffEqModule,
)

# --- config (env-overridable for a cheap smoke run) -------------------------------------------
N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", 4))  # held-out FlatVel_A maps to average over
STEPS = int(os.environ.get("EVAL_STEPS", 200))  # generative budget: STEPS * N_SAMPLES solves
N_SAMPLES = int(os.environ.get("EVAL_N_SAMPLES", 4))
SEED = 0

# FlatVel_A prior checkpoints (gitignored runs/ — update to your latest). The flow priors and the
# DiT diffusion share one DiT backbone; the UNet diffusion is the architecture control.
FLOW_CKPT = "runs/0001_flow_matching/2026-06-26T23-40-26Z/checkpoints/step_99_ema.pt"
FLOWMAP_CKPT = "runs/0002_flow_map/2026-06-26T20-57-58Z/checkpoints/step_99_ema.pt"
DIFF_UNET_CKPT = "runs/0003_baselines/2026-06-26T11-22-31Z/checkpoints/step_99_ema.pt"
DIFF_DIT_CKPT = "runs/0003_baselines/2026-06-27T00-21-12Z/checkpoints/step_99_ema.pt"


def _repo_root() -> Path:
    root = Path.cwd()
    while not (root / ".git").exists() and root != root.parent:
        root = root.parent
    return root


def load_flow(ckpt: str, device: torch.device) -> torch.nn.Module:
    m = build_model(
        (1, 64, 64), None, DiTModelConfig(hidden=256, depth=6, num_heads=8, patch_size=4)
    ).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    return m.eval()


def load_denoiser(kind: str, ckpt: str, device: torch.device) -> torch.nn.Module:
    d = build_denoiser(kind, sample_size=64, channels=1).to(device)
    d.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    return d.eval()


def main() -> None:
    os.chdir(_repo_root())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    flow_prior = load_flow(FLOW_CKPT, device)
    flowmap_prior = load_flow(FLOWMAP_CKPT, device)
    unet_denoiser = load_denoiser("unet", DIFF_UNET_CKPT, device)
    dit_denoiser = load_denoiser("dit", DIFF_DIT_CKPT, device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)  # type: ignore[no-untyped-call]

    ev = Evaluator.from_openfwi(["FlatVel_A"], N_TARGETS, device=device)
    print(f"{len(ev.targets)} held-out targets\n")

    def named(module: object, name: str) -> object:
        module.name = name  # type: ignore[attr-defined]
        return module

    def flow(prior: torch.nn.Module, g: float) -> FlowTiltModule:
        return FlowTiltModule(
            prior, guidance=g, steps=STEPS, n_samples=N_SAMPLES, device=device
        )

    def dps(denoiser: torch.nn.Module, g: float) -> DiffusionDPSModule:
        return DiffusionDPSModule(
            denoiser, scheduler, guidance=g, steps=STEPS, n_samples=N_SAMPLES, device=device
        )

    # All methods at a matched ~800-solve budget. Generative: STEPS*N_SAMPLES. Classical:
    # iters*n_samples (more iters per model, fewer near-identical restarts). RED-DiffEq: as DPS.
    modules = [
        named(flow(flow_prior, 1.0), "fm·tilt·g1"),
        named(flow(flow_prior, 0.0), "fm·prior·g0"),
        named(flow(flowmap_prior, 1.0), "flowmap·tilt·g1"),
        named(flow(flowmap_prior, 0.0), "flowmap·prior·g0"),
        named(dps(dit_denoiser, 0.3), "diff_dit·dps·g0.3"),
        named(dps(unet_denoiser, 0.3), "diff_unet·dps·g0.3"),
        named(
            REDDiffEqModule(
                dit_denoiser, scheduler, eta_data=0.3, eta_reg=0.1, t_denoise=100,
                iters=STEPS, n_samples=N_SAMPLES, device=device,
            ),
            "red_diffeq·dit",
        ),
        named(
            ClassicalFWIModule(
                reg="tikhonov", reg_weight=1e-3, iters=2 * STEPS, lr=0.02,
                n_samples=N_SAMPLES // 2, device=device,
            ),
            "classical·tikhonov",
        ),
        named(
            ClassicalFWIModule(
                reg="tv", reg_weight=1e-3, iters=2 * STEPS, lr=0.02,
                n_samples=N_SAMPLES // 2, device=device,
            ),
            "classical·tv",
        ),
    ]

    rows: list[dict[str, float]] = []
    for m in modules:
        torch.manual_seed(SEED)  # same posterior noise across modules for a fair compare
        stats = ev.evaluate(m)  # type: ignore[arg-type]
        print(stats, "\n")
        rows.append({"method": m.name, **stats.summary()})  # type: ignore[attr-defined,dict-item]

    _write_table(rows)


def _write_table(rows: list[dict[str, float]]) -> None:
    out_dir = Path("runs/inversion_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "results.md"
    cols = ["method", "mae_mean", "rmse_mean", "ssim_mean", "misfit_mean", "n_solves"]
    lines = [
        f"# Inversion evaluation — FlatVel_A, {N_TARGETS} held-out maps",
        "",
        "Normalized [-1,1] MAE/RMSE/SSIM, expected across posterior samples. Lower MAE/RMSE,"
        " higher SSIM is better. Matched forward-solve budget.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in sorted(rows, key=lambda d: d["mae_mean"]):
        cells = [str(r["method"])] + [
            f"{r[c]:.4g}" if c != "n_solves" else f"{r[c]:.0f}" for c in cols[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
