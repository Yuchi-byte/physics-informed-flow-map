"""Evaluate the faithful MFM-G gradient steering (paper Eq. 22/37 + steered SDE Eq. 38) against
the DPS-tilt baselines and the unguided control, on the same n=64 held-out FlatVel_A maps.

MFM-G steers the flow-map prior with sigma_t^2 * grad V_t, where grad V_t is the IWAE estimator
backpropped through mc_samples one-step posterior draws — NO renormalisation (renorm=False is the
paper). We sweep the likelihood temperature sigma (the only FWI-specific free parameter in the
reward r = -||F(v)-d||^2 / 2sigma^2) and include one renorm=True point to show the audited
deviation over-fits.

    uv run python notebooks/eval_mfm_g.py     # EVAL_N_TARGETS overridable

Writes runs/inversion_eval/results_mfm_g.md incrementally (partial runs leave a usable table).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.inversion import (
    Evaluator,
    FlowMapSteerModule,
    FlowTiltModule,
)

N_TARGETS = int(os.environ.get("EVAL_N_TARGETS", 64))
N_SAMPLES = 4
MFM_STEPS, MFM_MC = 100, 4  # MFM-G: 100 SDE steps, 4 posterior draws/step
TILT_STEPS = 200  # DPS baselines (their tuned budget from the n=64 core run)

FM = "runs/0001_flow_matching/2026-06-26T23-40-26Z/checkpoints/step_99_ema.pt"
FLOWMAP = "runs/0002_flow_map/2026-06-27T01-44-09Z/checkpoints/step_99_ema.pt"


def _root() -> Path:
    r = Path.cwd()
    while not (r / ".git").exists() and r != r.parent:
        r = r.parent
    return r


def main() -> None:
    os.chdir(_root())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}  n_targets={N_TARGETS}", flush=True)

    def load_flow(ckpt: str) -> torch.nn.Module:
        m = build_model(
            (1, 64, 64),
            None,
            DiTModelConfig(hidden=256, depth=6, num_heads=8, patch_size=4),
        ).to(dev)
        m.load_state_dict(
            torch.load(ckpt, map_location=dev, weights_only=False)["model"]
        )
        return m.eval()

    fm, fmap = load_flow(FM), load_flow(FLOWMAP)

    ev = Evaluator.from_openfwi(["FlatVel_A"], N_TARGETS, device=dev)
    print(f"{len(ev.targets)} held-out targets\n", flush=True)

    def steer(name: str, *, sigma: float, renorm: bool, gs: float = 1.0) -> object:
        est = "iwae" if gs != 0.0 else "base"
        m = FlowMapSteerModule(
            fmap,
            drift_estimator=est,
            mc_samples=MFM_MC,
            sigma=sigma,
            n_steps=MFM_STEPS,
            n_samples=N_SAMPLES,
            device=dev,
            guidance_scale=gs,
            renorm=renorm,
            sde=True,
            resolution=64,
        )
        m.name = name
        return m

    def tilt(prior: torch.nn.Module, name: str) -> object:
        m = FlowTiltModule(
            prior, guidance=1.0, steps=TILT_STEPS, n_samples=N_SAMPLES, device=dev
        )
        m.name = name
        return m

    modules = [
        steer("base·sde (unguided)", sigma=1000.0, renorm=False, gs=0.0),
        tilt(fm, "fm·tilt (DPS champion)"),
        steer("mfm_g·σ300", sigma=300.0, renorm=False),
        steer("mfm_g·σ1000", sigma=1000.0, renorm=False),
        steer("mfm_g·σ1000·renorm (audited deviation)", sigma=1000.0, renorm=True),
    ]

    out = Path("runs/inversion_eval/results_mfm_g.md")
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
            f"{a['misfit_mean']:.4g} | {stats.n_solves:.0f} |"
        )
        header = [
            f"# Faithful MFM-G vs DPS — FlatVel_A, {N_TARGETS} held-out maps",
            "",
            f"MFM-G: {MFM_STEPS}-step SDE, mc={MFM_MC}, renorm=False (paper Eq. 38). Normalized "
            f"[-1,1], expected across n_samples={N_SAMPLES}. ± = std across maps; SEM ≈ std/√{N_TARGETS}.",
            "",
            "| method | mae ± std | rmse | ssim ± std | misfit | solves/inv |",
            "|---|---|---|---|---|---|",
        ]
        out.write_text("\n".join(header + rows) + "\n")  # incremental
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
