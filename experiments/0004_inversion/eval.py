"""Score several inverters on the same held-out velocity maps — the multi-map sweep counterpart
to ``run.py`` (single target + figure).

    uv run python experiments/0004_inversion/eval.py experiment=eval

Builds an :class:`~physics_informed_flow_map.inversion.InversionModule` per ``evaluation.methods``
entry (each a prior + method + ckpt), scores them all on the same held-out OpenFWI maps and
Deepwave operator at a matched forward-solve budget (``steps * n_samples`` per inversion), and
writes a markdown results table under ``runs/0004_inversion/eval/``. Metrics are the OpenFWI
convention — MAE/RMSE/SSIM on the ``[-1, 1]`` velocity, expected across posterior samples.

Caveat (shared with ``run.py``): ``d_obs`` is the same noiseless forward operator used in guidance
(an "inverse crime"), so recovery is optimistic and data-fitting methods are flattered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from physics_informed_flow_map.experiment import start_run
from physics_informed_flow_map.physics.misfit import make_misfit
from physics_informed_flow_map.inversion import (
    ClassicalFWIModule,
    DiffusionDPSModule,
    Evaluator,
    FlowMapSteerModule,
    FlowTiltModule,
    InversionModule,
    RealisticFWIModule,
    REDDiffEqModule,
)

sys.path.insert(0, str(Path(__file__).parent))
from loaders import FLOW_PRIORS, load_diffusion_prior, load_flow_prior  # noqa: E402
from run import EvalEntry, InversionConfig  # noqa: E402

EXPERIMENT = "0004_inversion"
_MFM_METHODS = {"mfm_g", "mfm_gf"}
_COLS = [
    "method",
    "mae_mean",
    "ssim_mean",
    "crps_mean",
    "energy_mean",
    "cov90_mean",
    "cov_err_mean",
    "misfit_mean",
    "n_solves",
]


def build_module(
    entry: EvalEntry,
    *,
    steps: int,
    n_samples: int,
    resolution: int,
    device: torch.device,
    min_freq_hz: float = 0.0,
) -> InversionModule:
    """Construct the configured inverter for one sweep entry (prior loaded from its checkpoint)."""
    if min_freq_hz > 0.0:
        raise ValueError(
            "guidance-side band-limiting was removed along with the highpass filter — "
            "obs.min_freq_hz > 0 can no longer be honoured by the misfit"
        )
    shape = (1, resolution, resolution)
    g = entry.method.guidance_strength
    # Guidance data-misfit knob (method.misfit: l2 | ot), built lazily from each target's
    # d_obs; None keeps the samplers' hard-wired L2 path. The prior-free FWI baselines
    # reject non-l2 at config time.
    misfit_factory = (
        None
        if entry.method.misfit == "l2"
        else lambda d: make_misfit(entry.method.misfit, d, ot_k=entry.method.ot_k)
    )

    if entry.prior.name == "none":
        # Prior-free FWI baselines: optimise the velocity map directly against the data misfit
        # at the forward operator's native grid. g=0 => the starting model (0 solves), matching
        # run.py's control semantics.
        if entry.method.name == "realistic_fwi":
            module: InversionModule = RealisticFWIModule(
                freqs_hz=entry.method.freqs_hz,
                iters_per_stage=entry.method.iters_per_stage if g != 0.0 else 0,
                lr=entry.method.lr,
                reg=entry.method.reg,
                reg_weight=entry.method.reg_weight,
                optimizer=entry.method.optimizer,
                n_samples=n_samples,
                device=device,
            )
        else:  # classical_fwi
            module = ClassicalFWIModule(
                reg=entry.method.reg,
                reg_weight=entry.method.reg_weight,
                iters=steps if g != 0.0 else 0,
                lr=entry.method.lr,
                n_samples=n_samples,
                device=device,
                init=entry.method.init,
            )
    elif entry.prior.name in FLOW_PRIORS:
        prior = load_flow_prior(
            shape,
            hidden=entry.model.hidden,
            depth=entry.model.depth,
            num_heads=entry.model.num_heads,
            patch_size=entry.model.patch_size,
            ckpt=entry.ckpt,
            device=device,
            prior=entry.prior.name,
        )
        if entry.method.name in _MFM_METHODS:
            module = FlowMapSteerModule(
                prior,
                drift_estimator=entry.method.drift_estimator if g != 0.0 else "base",
                mc_samples=entry.method.mc_samples,
                sigma=entry.method.sigma,
                n_steps=steps,
                n_samples=n_samples,
                device=device,
                guidance_scale=g,
                renorm=entry.method.renorm,
                sde=entry.method.sde,
                resolution=resolution,
                misfit_factory=misfit_factory,
            )
        else:
            module = FlowTiltModule(
                prior,
                guidance=g,
                steps=steps,
                n_samples=n_samples,
                device=device,
                resolution=resolution,
                normalize_grad=entry.method.normalize_grad,
                misfit_factory=misfit_factory,
            )
    else:  # diffusion prior
        denoiser, scheduler = load_diffusion_prior(
            shape,
            denoiser_kind=entry.prior.denoiser_kind,
            hidden=entry.model.hidden,
            depth=entry.model.depth,
            num_heads=entry.model.num_heads,
            patch_size=entry.model.patch_size,
            num_train_timesteps=entry.prior.num_train_timesteps,
            ckpt=entry.ckpt,
            device=device,
        )
        if entry.method.name == "red_diffeq":
            module = REDDiffEqModule(
                denoiser,
                scheduler,
                eta_data=entry.method.eta_data if g != 0.0 else 0.0,
                eta_reg=entry.method.eta_reg,
                t_denoise=entry.method.t_denoise,
                iters=entry.method.iters or steps,
                n_samples=n_samples,
                device=device,
                resolution=resolution,
                normalize_grad=entry.method.normalize_grad,
                misfit_factory=misfit_factory,
            )
        else:  # dps / unguided
            module = DiffusionDPSModule(
                denoiser,
                scheduler,
                guidance=g,
                steps=steps,
                n_samples=n_samples,
                device=device,
                resolution=resolution,
                normalize_grad=entry.method.normalize_grad,
                misfit_factory=misfit_factory,
            )
    module.name = entry.label
    return module


def write_table(
    rows: list[dict[str, str | float]], families: list[str], n_targets: int, out: Path
) -> None:
    lines = [
        f"# Inversion evaluation — {', '.join(families)}, {n_targets} held-out maps",
        "",
        "Normalized [-1,1] MAE/RMSE/SSIM, expected across posterior samples. Lower MAE/RMSE,"
        " higher SSIM is better. Matched forward-solve budget.",
        "",
        "| " + " | ".join(_COLS) + " |",
        "|" + "|".join(["---"] * len(_COLS)) + "|",
    ]
    for r in sorted(rows, key=lambda d: float(d["mae_mean"])):
        cells = [str(r["method"])] + [
            f"{r[c]:.4g}" if c != "n_solves" else f"{r[c]:.0f}" for c in _COLS[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = InversionConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, InversionConfig)
    if not cfg.evaluation.methods:
        raise SystemExit(
            "no methods to evaluate — select the sweep variant (experiment=eval) or set "
            "evaluation.methods=[...]"
        )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name="eval")

    ev = Evaluator.from_openfwi(
        cfg.evaluation.families,
        cfg.evaluation.n_targets,
        device=dev,
        resolution=cfg.dataset.shape[1],
        obs_cfg=cfg.obs,
    )
    print(
        f"{len(ev.targets)} held-out targets, {len(cfg.evaluation.methods)} methods\n"
    )

    rows: list[dict[str, str | float]] = []
    for entry in cfg.evaluation.methods:
        module = build_module(
            entry,
            steps=cfg.steps,
            n_samples=cfg.n_samples,
            resolution=cfg.dataset.shape[1],
            device=dev,
            min_freq_hz=cfg.obs.min_freq_hz,
        )
        torch.manual_seed(
            cfg.seed
        )  # same posterior noise across modules for a fair compare
        stats = ev.evaluate(module)
        print(stats, "\n")
        rows.append({"method": module.name, **stats.summary()})
        run.log(**{f"{module.name}/{k}": v for k, v in stats.summary().items()})

    out = run_dir / "results.md"
    write_table(rows, cfg.evaluation.families, cfg.evaluation.n_targets, out)
    run.finish(n_methods=len(rows))


if __name__ == "__main__":
    main()
