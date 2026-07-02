"""Invert a held-out velocity map with a trained prior — the unified camp-A inversion entry.

    uv run python experiments/0004_inversion/run.py prior=flow_matching method=flow_tilt ckpt=<...>
    uv run python experiments/0004_inversion/run.py prior=flow_map     method=mfm_g     ckpt=<...>
    uv run python experiments/0004_inversion/run.py prior=diffusion    method=dps       ckpt=<...>
    uv run python experiments/0004_inversion/run.py --multirun prior=flow_map method=mfm_g,flow_tilt,unguided ckpt=<...>

Loads a prior trained by one of the training frameworks (``0001_flow_matching``, ``0002_flow_map``,
``0003_diffusion`` — do not retrain here), simulates seismic data ``d`` from a held-out
(validation-split) OpenFWI map with the Deepwave forward operator, then steers the prior toward
``d`` to recover the velocity. The ``prior`` group picks the prior family/loader and the ``method``
group picks the inference-time scheme:

  * ``flow_tilt`` — DPS-style wave-equation tilting with the single-point Tweedie mean (flow priors).
  * ``mfm_g`` / ``mfm_gf`` — Meta Flow Maps' gradient / gradient-free steering through the flow
    map's own time-conditional posterior (paper Eq. 22 / Eq. 20; ``prior-work.html`` §5.5). flow_map.
  * ``dps`` — canonical Diffusion Posterior Sampling (diffusion prior).
  * ``unguided`` — prior sample, no physics (the control that shows the wave equation does the work).

Scoring + figure are shared across all methods via ``inversion.single_target``; this entry only
supplies how to load each prior and how to invert. ``eval.py`` is the multi-map sweep counterpart.

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
from physics_informed_flow_map.flow_matching.openfwi import viz_velocity
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    OpenFWIDatasetConfig,
)
from physics_informed_flow_map.inversion import FlowMapSteerModule
from physics_informed_flow_map.inversion.bridge import mps_to_norm, seismic_forward
from physics_informed_flow_map.inversion.single_target import (
    Inverter,
    invert_and_report,
)
from physics_informed_flow_map.physics.tilt import guided_sample

import sys

sys.path.insert(0, str(Path(__file__).parent))
from loaders import (  # noqa: E402
    FLOW_PRIORS,
    load_diffusion_prior,
    load_flow_prior,
)

EXPERIMENT = "0004_inversion"

# prior family -> the inference-time methods it supports (the validated compatibility matrix).
_COMPAT: dict[str, set[str]] = {
    "flow_matching": {"unguided", "flow_tilt"},
    "flow_map": {"unguided", "flow_tilt", "mfm_g", "mfm_gf"},
    "diffusion": {"unguided", "dps", "red_diffeq"},
    "none": {"classical_fwi", "realistic_fwi"},  # no learned prior: the FWI baselines
}
_PRIORS = set(_COMPAT)
_METHODS = {m for ms in _COMPAT.values() for m in ms}
_MFM_METHODS = {"mfm_g", "mfm_gf"}


def check_compatible(prior: str, method: str) -> None:
    """Raise if ``prior`` cannot run ``method`` (the approved compatibility matrix)."""
    if prior not in _PRIORS:
        raise ValueError(f"unknown prior '{prior}' ({' | '.join(sorted(_PRIORS))})")
    if method not in _METHODS:
        raise ValueError(f"unknown method '{method}' ({' | '.join(sorted(_METHODS))})")
    if method not in _COMPAT[prior]:
        ok = ", ".join(p for p in sorted(_PRIORS) if method in _COMPAT[p])
        raise ValueError(
            f"method '{method}' is incompatible with prior '{prior}' "
            f"(valid priors for '{method}': {ok})"
        )


class PriorConfig(Config):
    name: str = "flow_matching"  # flow_matching | flow_map | diffusion
    denoiser_kind: str = "unet"  # diffusion only: unet | dit
    num_train_timesteps: int = Field(1000, gt=0)  # diffusion only: DDPM schedule


class MethodConfig(Config):
    name: str = "flow_tilt"  # flow_tilt | unguided | mfm_g | mfm_gf | dps
    # Guidance scale: DPS step size (flow_tilt/dps), steering scale (mfm_g/mfm_gf).
    # 0 => unguided (the control invert_and_report runs for the misfit ratio).
    guidance_strength: float = 1.0
    normalize_grad: bool = True  # flow_tilt/dps only: unit-normalise the DPS gradient
    # MFM-G / MFM-GF knobs (ignored by the other methods):
    drift_estimator: str = (
        "iwae"  # iwae = MFM-G (gradient); sne = MFM-GF (gradient-free)
    )
    mc_samples: int = Field(
        4, gt=0
    )  # posterior draws/step; wave solves/step scale with it
    sigma: float = Field(
        1000.0, gt=0.0
    )  # likelihood temperature in r=-||F(v)-d||^2/(2σ²)
    renorm: bool = False  # pin steering magnitude to the base-drift norm each step
    sde: bool = True  # SDE (Euler-Maruyama) sampler; false => ODE (Euler)
    # RED-DiffEq knobs (diffusion prior; ignored by the other methods).
    eta_data: float = Field(
        0.1, ge=0.0
    )  # data-misfit step (state-space when normalize_grad)
    eta_reg: float = Field(0.05, ge=0.0)  # RED prior-residual step
    t_denoise: int = Field(100, gt=0)  # fixed DDPM level the Tweedie denoiser runs at
    iters: int = Field(
        0, ge=0
    )  # RED-DiffEq optimisation steps; 0 => use the global `steps`
    # Classical / realistic FWI knobs (prior=none; ignored by the other methods).
    lr: float = Field(0.02, gt=0.0)  # step size (Adam ~0.02; L-BFGS ~1.0)
    reg: str = "tikhonov"  # roughness penalty: tikhonov | tv
    reg_weight: float = Field(0.0, ge=0.0)  # weight on R(v); 0 = pure least-squares
    init: str = "smooth"  # classical_fwi start: smooth (1-D gradient) | random (structure-free)
    optimizer: str = "adam"  # realistic_fwi optimiser: lbfgs | adam
    iters_per_stage: int = Field(40, gt=0)  # realistic_fwi: inner iterations per frequency band
    freqs_hz: list[float] = Field(  # realistic_fwi: multiscale cutoffs (Hz), ascending
        default_factory=lambda: [4.0, 8.0, 15.0]
    )


class ModelConfig(Config):
    hidden: int = 256
    depth: int = 6
    num_heads: int = 8
    patch_size: int = 4


class EvalEntry(Config):
    """One (prior, method, ckpt) inverter in an ``eval.py`` sweep (see ``conf/experiment/eval``)."""

    label: str
    prior: PriorConfig = PriorConfig()
    ckpt: str = ""
    method: MethodConfig = MethodConfig()
    model: ModelConfig = ModelConfig()


class EvalConfig(Config):
    """The ``eval.py`` multi-map sweep params (unused by ``run.py``)."""

    families: list[str] = Field(default_factory=lambda: ["FlatVel_A"])
    n_targets: int = Field(8, gt=0)
    methods: list[EvalEntry] = Field(default_factory=list)


class InversionConfig(Config):
    seed: int = 0
    ckpt: str = ""  # trained prior (a training-framework checkpoint); empty = untrained, plumbing only
    diff_ckpt: str = ""  # sweep-only: a second checkpoint for diffusion entries to interpolate (${diff_ckpt})
    target_index: int = Field(0, ge=0)  # index into the seed-0 validation split
    n_samples: int = Field(4, gt=0)
    steps: int = Field(200, gt=0)  # sampler / reverse steps
    n_frames: int = Field(3, ge=0)  # trajectory snapshots per inversion run (0 = off)
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = OpenFWIDatasetConfig()
    prior: PriorConfig = PriorConfig()
    method: MethodConfig = MethodConfig()
    evaluation: EvalConfig = EvalConfig()  # eval.py sweep params (experiment=eval)

    @model_validator(mode="after")
    def _check(self) -> "InversionConfig":
        if not isinstance(self.dataset, OpenFWIDatasetConfig):
            raise ValueError(
                "inversion targets OpenFWI velocity maps (dataset=openfwi)"
            )
        check_compatible(self.prior.name, self.method.name)
        for entry in self.evaluation.methods:  # validate the sweep entries too
            check_compatible(entry.prior.name, entry.method.name)
        return self


InversionConfig.model_rebuild()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = InversionConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, InversionConfig)
    assert isinstance(cfg.dataset, OpenFWIDatasetConfig)  # narrowed by the validator

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    name = f"invert-{cfg.prior.name}-{cfg.method.name}"
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name=name)

    channels, size, _ = cfg.dataset.shape
    n = cfg.n_samples
    n_solves = (
        0  # forward (PDE) solves consumed by the guided pass (matched-cost reporting)
    )
    solves = {"n": 0}  # actual guided-pass solve count (FWI fills this; used for the figure banner)

    if cfg.prior.name == "none":
        # Classical / realistic FWI: no learned prior. Optimise the velocity map directly against
        # the data misfit at native 70x70 (the forward operator's grid — dataset.resolution is the
        # prior's training size, irrelevant here). guidance=0 => the starting model (control).
        from physics_informed_flow_map.inversion.bridge import NATIVE
        from physics_informed_flow_map.physics.classical import (
            multiscale_fwi,
            regularized_fwi,
        )
        from physics_informed_flow_map.physics.forward import simulate

        realistic = cfg.method.name == "realistic_fwi"

        def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
            run_it = guidance_strength != 0.0
            if realistic:
                v_mps, ns = multiscale_fwi(
                    simulate,
                    d_obs,
                    shape=(NATIVE, NATIVE),
                    n_samples=n,
                    freqs_hz=cfg.method.freqs_hz,
                    iters_per_stage=cfg.method.iters_per_stage if run_it else 0,
                    lr=cfg.method.lr,
                    reg=cfg.method.reg,
                    reg_weight=cfg.method.reg_weight,
                    optimizer=cfg.method.optimizer,
                    device=dev,
                )
            else:
                v_mps, ns = regularized_fwi(
                    simulate,  # forward_fn: (H, W) m/s -> data
                    d_obs,
                    shape=(NATIVE, NATIVE),
                    n_samples=n,
                    iters=cfg.steps if run_it else 0,
                    lr=cfg.method.lr,
                    reg=cfg.method.reg,
                    reg_weight=cfg.method.reg_weight,
                    init=cfg.method.init,
                    device=dev,
                )
            if run_it:
                solves["n"] = ns  # true solve count (L-BFGS line searches included)
            return mps_to_norm(v_mps)[:, None]  # (n, 1, 70, 70) in [-1, 1]

        n_solves = 0  # set from solves["n"] after the guided pass runs
    elif cfg.prior.name in FLOW_PRIORS:
        prior = load_flow_prior(
            cfg.dataset.shape,
            hidden=cfg.model.hidden,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            patch_size=cfg.model.patch_size,
            ckpt=cfg.ckpt,
            device=dev,
            prior=cfg.prior.name,
        )

        if cfg.method.name in _MFM_METHODS:
            # Native flow-map steering (FlowMapSteerModule), adapted to the shared [-1, 1] Inverter
            # contract. guidance=0 => base drift (unguided control: estimator "base", no solves).
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

            n_solves = (
                cfg.steps * cfg.method.mc_samples * n
            )  # mc posterior solves per step
        else:
            # DPS Tweedie baseline (flow_tilt) / unguided control: tilt from a fixed noise context
            # x0 (reused guided/unguided), using v(t,t,x|noise) at t_cond=0.
            x0 = torch.randn(n, channels, size, size, device=dev)
            t_cond = torch.zeros(n, device=dev)

            def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
                tb = torch.full((x.shape[0],), t, device=dev)
                return cast(torch.Tensor, prior.v(tb, tb, x, t_cond, x0))

            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    step_cb = run.make_step_saver(
                        f"flow_tilt_g{guidance_strength:.2g}",
                        lambda x, p: viz_velocity(x[:, None] if x.ndim == 3 else x, p),
                        total_steps=cfg.steps,
                        n_frames=cfg.n_frames,
                    )
                return guided_sample(
                    velocity_fn,
                    x0,
                    seismic_forward,
                    d_obs,
                    sampler_steps=cfg.steps,
                    guidance_strength=guidance_strength,
                    normalize_grad=cfg.method.normalize_grad,
                    on_step=step_cb,
                )

            n_solves = cfg.steps * n  # one Tweedie sim per step per sample
    else:  # diffusion prior + dps / red_diffeq / unguided
        from physics_informed_flow_map.baselines import dps_sample, red_diffeq_sample

        denoiser, scheduler = load_diffusion_prior(
            cfg.dataset.shape,
            denoiser_kind=cfg.prior.denoiser_kind,
            hidden=cfg.model.hidden,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            patch_size=cfg.model.patch_size,
            num_train_timesteps=cfg.prior.num_train_timesteps,
            ckpt=cfg.ckpt,
            device=dev,
        )

        if cfg.method.name == "red_diffeq":
            # RED-DiffEq optimisation: eta_data/eta_reg are the knobs; guidance=0 still flags the
            # control (invert_and_report runs it for the misfit ratio), so gate eta_data on it.
            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    step_cb = run.make_step_saver(
                        f"red_diffeq_g{guidance_strength:.2g}",
                        lambda x, p: viz_velocity(x, p),
                        total_steps=cfg.method.iters or cfg.steps,
                        n_frames=cfg.n_frames,
                    )
                return red_diffeq_sample(
                    denoiser,
                    scheduler,
                    cfg.dataset.shape,
                    seismic_forward,
                    d_obs,
                    n_samples=n,
                    iters=cfg.method.iters or cfg.steps,
                    eta_data=cfg.method.eta_data if guidance_strength != 0.0 else 0.0,
                    eta_reg=cfg.method.eta_reg,
                    t_denoise=cfg.method.t_denoise,
                    device=dev,
                    normalize_grad=cfg.method.normalize_grad,
                    on_step=step_cb,
                )
        else:  # dps / unguided

            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    step_cb = run.make_step_saver(
                        f"dps_g{guidance_strength:.2g}",
                        lambda x, p: viz_velocity(x, p),
                        total_steps=cfg.steps,
                        n_frames=cfg.n_frames,
                    )
                return dps_sample(
                    denoiser,
                    scheduler,
                    cfg.dataset.shape,
                    seismic_forward,
                    d_obs,
                    n_samples=n,
                    num_steps=cfg.steps,
                    guidance_strength=guidance_strength,
                    device=dev,
                    normalize_grad=cfg.method.normalize_grad,
                    on_step=step_cb,
                )

        n_solves = (
            cfg.method.iters or cfg.steps
            if cfg.method.name == "red_diffeq"
            else cfg.steps
        ) * n

    # Fixed-seed sampling for reproducible figures: invert_and_report runs a guided then an
    # unguided pass, so re-seed before each so both start from identical noise (the guided/unguided
    # panels track the same prior sample). Every sampler here draws from the global RNG, so a single
    # re-seed covers all prior families/methods; safe because there is no training loop to perturb.
    _base_invert: Inverter = invert

    def invert_fn(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
        torch.manual_seed(cfg.seed)
        return _base_invert(d_obs, guidance_strength)

    out = run.ckpt_dir.parent / "inversion.png"
    obs_out = run.ckpt_dir.parent / "d_obs.png"

    def solve_count() -> float:
        # Total forward solves of the guided pass, resolved after it runs (FWI fills solves["n"];
        # the other methods know it up front). unguided does no physics -> 0.
        if cfg.method.name == "unguided":
            return 0.0
        return float(solves["n"]) if cfg.prior.name == "none" else float(n_solves)

    summary, caption = invert_and_report(
        invert_fn,
        dataset_cfg=cfg.dataset,
        target_index=cfg.target_index,
        method_name=cfg.method.name,
        guidance=cfg.method.guidance_strength,
        steps=cfg.steps,
        device=dev,
        out_png=out,
        out_obs_png=obs_out,
        cost=solve_count,
    )
    if cfg.prior.name == "none":
        n_solves = solves["n"]  # actual forward solves from the guided FWI pass
    summary["inv/n_solves"] = float(0 if cfg.method.name == "unguided" else n_solves)
    run.log_image("inversion", out, caption=caption)
    run.log_image("d_obs", obs_out, caption=f"observed seismic · {caption}")
    run.log(**summary)  # mirror to metrics.jsonl on the network volume
    run.finish(**summary)


if __name__ == "__main__":
    main()
