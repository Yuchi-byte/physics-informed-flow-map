"""Invert a held-out velocity map with a trained Meta-Flow-Map prior — camp-A flow steering.

    uv run python experiments/0002_flow_map/inversion.py ckpt=<run.py EMA ckpt> method=mfm_g
    uv run python experiments/0002_flow_map/inversion.py ckpt=<...> method=flow_tilt
    uv run python experiments/0002_flow_map/inversion.py ckpt=<...> method=unguided
    uv run python experiments/0002_flow_map/inversion.py --multirun ckpt=<...> method=mfm_g,flow_tilt,unguided

Loads a flow-map prior trained by ``run.py`` (do not retrain here), simulates seismic data ``d``
from a held-out (validation-split) OpenFWI map with the Deepwave forward operator, then steers the
prior toward ``d`` to recover the velocity. Three methods:

  * ``mfm_g`` — Meta Flow Maps' gradient-based (IWAE) steering (paper Eq. 22; ``prior-work.html``
    §5.5). Estimates ∇V_t by backpropping the data log-likelihood through ``mc_samples`` one-step
    posterior draws ``v(0,1,ε | t_cond=t, x_cond=x_t)`` — the flow map's *own* time-conditional
    posterior, not a Tweedie mean. Our differentiable Deepwave reward makes this the estimator of
    choice. ``mfm_gf`` is the gradient-free fallback (self-normalised, Eq. 20).
  * ``flow_tilt`` — DPS baseline: steers with the single-point Tweedie mean ``x1≈x_t+(1-t)v`` at
    ``t_cond=0`` (same scheme as ``0001_flow_matching/inversion.py``; does *not* use the posterior).
  * ``unguided`` — prior sample, no physics (the control).

Diffusion counterpart: ``0003_baselines/inversion.py`` — so the flow-map and diffusion camps
compare head-to-head. Scoring + figure are shared via ``inversion.single_target``; this entry only
supplies how to invert. The MFM-G path reuses ``inversion.FlowMapSteerModule`` (the native steering
that wraps mfm's drift/sampler), adapted to the shared ``[-1, 1]`` ``Inverter`` contract.

Caveat: ``d_obs`` comes from the same noiseless forward operator used inside the guidance term
(an "inverse crime"), so recovery is optimistic.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field, model_validator

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    OpenFWIDatasetConfig,
)
from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.inversion import FlowMapSteerModule
from physics_informed_flow_map.inversion.bridge import mps_to_norm, seismic_forward
from physics_informed_flow_map.inversion.single_target import (
    Inverter,
    invert_and_report,
)
from physics_informed_flow_map.physics.tilt import guided_sample

EXPERIMENT = "0002_flow_map"

_MFM_METHODS = {"mfm_g", "mfm_gf"}  # native flow-map steering (FlowMapSteerModule)
_TILT_METHODS = {"flow_tilt", "unguided"}  # DPS Tweedie baseline + control


class MethodConfig(Config):
    name: str = "flow_tilt"  # flow_tilt | unguided | mfm_g | mfm_gf
    # Guidance scale: the DPS step size for flow_tilt, the steering scale for mfm_g/mfm_gf.
    # 0 => unguided (the control invert_and_report runs for the misfit ratio).
    guidance_strength: float = 1.0
    normalize_grad: bool = True  # flow_tilt only: unit-normalise the DPS gradient
    # MFM-G / MFM-GF knobs (ignored by flow_tilt/unguided):
    drift_estimator: str = (
        "iwae"  # iwae = MFM-G (gradient); sne = MFM-GF (gradient-free)
    )
    mc_samples: int = Field(
        4, gt=0
    )  # posterior draws/step; wave solves/step scale with it
    sigma: float = Field(1.0, gt=0.0)  # likelihood temperature in r=-||F(v)-d||^2/(2σ²)
    renorm: bool = True  # pin steering magnitude to the base-drift norm each step
    sde: bool = True  # SDE (Euler-Maruyama) sampler; false => ODE (Euler)


class ModelConfig(Config):
    hidden: int = 256
    depth: int = 6
    num_heads: int = 8
    patch_size: int = 4


class InversionConfig(Config):
    seed: int = 0
    ckpt: str = ""  # trained flow-map prior (a run.py checkpoint); empty = untrained, plumbing only
    target_index: int = Field(0, ge=0)  # index into the seed-0 validation split
    n_samples: int = Field(4, gt=0)
    steps: int = Field(100, gt=0)
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = OpenFWIDatasetConfig()
    method: MethodConfig = MethodConfig()

    @model_validator(mode="after")
    def _check(self) -> "InversionConfig":
        if not isinstance(self.dataset, OpenFWIDatasetConfig):
            raise ValueError(
                "inversion targets OpenFWI velocity maps (dataset=openfwi)"
            )
        if self.method.name not in _MFM_METHODS | _TILT_METHODS:
            raise ValueError(
                f"unknown method '{self.method.name}' "
                "(mfm_g | mfm_gf | flow_tilt | unguided)"
            )
        if self.method.name in _MFM_METHODS and self.method.drift_estimator not in {
            "iwae",
            "sne",
        }:
            raise ValueError(
                f"method={self.method.name} needs drift_estimator iwae (MFM-G) or sne "
                f"(MFM-GF), got '{self.method.drift_estimator}'"
            )
        return self


InversionConfig.model_rebuild()


@hydra.main(version_base=None, config_path="conf", config_name="inversion")
def main(dcfg: DictConfig) -> None:
    cfg = InversionConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, InversionConfig)
    assert isinstance(cfg.dataset, OpenFWIDatasetConfig)  # narrowed by the validator

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name=f"invert-{cfg.method.name}")

    channels, size, _ = cfg.dataset.shape
    prior = build_model(
        cfg.dataset.shape,
        None,
        DiTModelConfig(
            hidden=cfg.model.hidden,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            patch_size=cfg.model.patch_size,
        ),
    ).to(dev)
    if cfg.ckpt:
        if not Path(cfg.ckpt).is_file():
            raise SystemExit(
                f"ckpt not found: {cfg.ckpt}\nTrain a prior first (uv run python "
                "experiments/0002_flow_map/run.py experiment=openfwi) and point ckpt= at its "
                "best-val EMA checkpoint (runs/0002_flow_map/<ts>/checkpoints/step_<N>_ema.pt)."
            )
        prior.load_state_dict(
            torch.load(cfg.ckpt, map_location=dev, weights_only=False)["model"]
        )
    else:
        print("[inversion] no ckpt= given: using an UNTRAINED prior (plumbing only)")
    prior.eval()

    n = cfg.n_samples
    n_solves = 0  # forward (PDE) solves consumed by the guided pass (for matched-cost reporting)

    if cfg.method.name in _TILT_METHODS:
        # DPS Tweedie baseline: tilt from a fixed noise context x0 (reused guided/unguided),
        # using the unconditional diagonal velocity v(t,t,x|noise) at t_cond=0.
        x0 = torch.randn(n, channels, size, size, device=dev)
        t_cond = torch.zeros(n, device=dev)

        def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            tb = torch.full((x.shape[0],), t, device=dev)
            return cast(torch.Tensor, prior.v(tb, tb, x, t_cond, x0))

        def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
            return guided_sample(
                velocity_fn,
                x0,
                seismic_forward,
                d_obs,
                sampler_steps=cfg.steps,
                guidance_strength=guidance_strength,
                normalize_grad=cfg.method.normalize_grad,
            )

        n_solves = cfg.steps * n  # one Tweedie sim per step per sample
    else:
        # MFM-G/MFM-GF: native flow-map steering through FlowMapSteerModule, adapted to the
        # shared [-1, 1] Inverter contract (its v_hat is m/s; map back so invert_and_report
        # scores and figures it like the others). guidance=0 => base drift (the unguided control,
        # estimator "base": no posterior draws, no wave solves).
        def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
            estimator = (
                cfg.method.drift_estimator if guidance_strength != 0.0 else "base"
            )
            module = FlowMapSteerModule(
                prior,
                drift_estimator=estimator,
                mc_samples=cfg.method.mc_samples,
                sigma=cfg.method.sigma,
                n_steps=cfg.steps,
                n_samples=n,
                device=dev,
                guidance_scale=guidance_strength,
                renorm=cfg.method.renorm,
                sde=cfg.method.sde,
                resolution=size,
            )
            return mps_to_norm(module.invert(d_obs).v_hat)[:, None]

        n_solves = cfg.steps * cfg.method.mc_samples * n  # mc posterior solves per step

    invert_fn: Inverter = invert
    out = run.ckpt_dir.parent / "inversion.png"
    summary, caption = invert_and_report(
        invert_fn,
        dataset_cfg=cfg.dataset,
        target_index=cfg.target_index,
        method_name=cfg.method.name,
        guidance=cfg.method.guidance_strength,
        steps=cfg.steps,
        device=dev,
        out_png=out,
    )
    summary["inv/n_solves"] = float(0 if cfg.method.name == "unguided" else n_solves)
    run.log_image("inversion", out, caption=caption)
    run.finish(**summary)


if __name__ == "__main__":
    main()
