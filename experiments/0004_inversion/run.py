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

for fmrg:
To run a real inversion:
uv run python experiments/0004_inversion/run.py prior=flow_map method=fmrg_e ckpt=<path> steps=200

To sweep n_opt or guidance_strength:
uv run python experiments/0004_inversion/run.py prior=flow_map method=fmrg_e ckpt=<path> method.n_opt=5 method.guidance_strength=0.5
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
from physics_informed_flow_map.inversion import FlowMapSteerModule
from physics_informed_flow_map.inversion.bridge import mps_to_norm, seismic_forward
from physics_informed_flow_map.inversion.fmrg import fmrg_e_sample
from physics_informed_flow_map.inversion.single_target import (
    Inverter,
    invert_and_report,
)
from physics_informed_flow_map.physics.misfit import MISFITS, make_misfit
from physics_informed_flow_map.physics.observation import ObservationConfig
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
    "flow_map": {"unguided", "flow_tilt", "mfm_g", "mfm_gf", "fmrg_e"},
    "diffusion": {"unguided", "dps", "red_diffeq"},
    "none": {"classical_fwi", "realistic_fwi"},  # no learned prior: the FWI baselines
}
_PRIORS = set(_COMPAT)
_METHODS = {m for ms in _COMPAT.values() for m in ms}
_MFM_METHODS = {"mfm_g", "mfm_gf"}

# Human-readable pieces for the data-space figure title ("<prior> + <estimator> + <misfit>").
_PRIOR_DISP = {
    "flow_matching": "flow-matching",
    "flow_map": "flow-map",
    "diffusion": "diffusion",
    "none": "FWI",
}
_EST_DISP = {
    "flow_tilt": "Tweedie",
    "dps": "Tweedie",
    "mfm_g": "MFM-G",
    "mfm_gf": "MFM-GF",
    "fmrg_e": "FMRG-E",
    "red_diffeq": "RED-DiffEq",
    "unguided": "unguided",
    "classical_fwi": "classical FWI",
    "realistic_fwi": "multiscale FWI",
}


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
    # Guidance data-misfit: l2 (pointwise, the Gaussian log-likelihood) | ot (Peng et al.
    # 2026 weighted+normalized trace-wise Wasserstein-2 potential — anti-cycle-skipping,
    # amplitude-balanced; see physics.misfit). Evaluation metrics stay L2 regardless.
    misfit: str = "l2"
    ot_k: float = Field(100.0, ge=0.0)  # ot only: bounded amplitude-weighting strength
    # How the drift toward the data is estimated. "tweedie" documents the single-point Tweedie
    # estimate hard-wired into flow_tilt/dps (those code paths never read this field); only
    # mfm_g/mfm_gf consume it, and must override it with an estimator mfm.utils.steering
    # accepts: iwae = MFM-G (gradient), sne = MFM-GF (gradient-free), dps = Tweedie baseline.
    drift_estimator: str = "tweedie"
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
    # FMRG-E knobs (flow_map prior only; ignored by the other methods).
    n_opt: int = Field(1, gt=0)  # inner gradient steps in x1-space per outer Euler step
    # Classical / realistic FWI knobs (prior=none; ignored by the other methods).
    lr: float = Field(0.02, gt=0.0)  # step size (Adam ~0.02; L-BFGS ~1.0)
    reg: str = "tikhonov"  # roughness penalty: tikhonov | tv
    reg_weight: float = Field(0.0, ge=0.0)  # weight on R(v); 0 = pure least-squares
    init: str = (
        "smooth"  # classical_fwi start: smooth (1-D gradient) | random (structure-free)
    )
    optimizer: str = "adam"  # realistic_fwi optimiser: lbfgs | adam
    iters_per_stage: int = Field(
        40, gt=0
    )  # realistic_fwi: inner iterations per frequency band
    freqs_hz: list[float] = Field(  # realistic_fwi: multiscale cutoffs (Hz), ascending
        default_factory=lambda: [4.0, 8.0, 15.0]
    )

    @model_validator(mode="after")
    def _check_drift_estimator(self) -> "MethodConfig":
        # mfm.utils.steering raises on unknown estimators only deep inside sampling; catch a
        # forgotten override (e.g. the documentation-only "tweedie" default) at config time.
        if self.name in _MFM_METHODS and self.drift_estimator not in {
            "dps",
            "iwae",
            "sne",
        }:
            raise ValueError(
                f"method '{self.name}' needs drift_estimator dps | iwae | sne, "
                f"got '{self.drift_estimator}'"
            )
        if self.misfit not in MISFITS:
            raise ValueError(f"unknown misfit '{self.misfit}' ({' | '.join(MISFITS)})")
        if self.misfit != "l2" and self.name in {"classical_fwi", "realistic_fwi"}:
            raise ValueError(
                f"misfit '{self.misfit}' is not threaded through the prior-free FWI "
                "baselines yet — they hard-code the (frequency-filtered) L2 objective"
            )
        return self


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
    # Benchmark target id (e.g. style_a_03) from data/inversion_bench/manifest.json;
    # overrides target_index and needs no bulk data. Empty = legacy target_index path.
    target: str = ""
    n_samples: int = Field(4, gt=0)
    steps: int = Field(200, gt=0)  # sampler / reverse steps
    n_frames: int = Field(6, ge=0)  # trajectory snapshots per inversion run (0 = off)
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = OpenFWIDatasetConfig()
    prior: PriorConfig = PriorConfig()
    method: MethodConfig = MethodConfig()
    # Benchmark track (conf/obs group): clean (legacy default) | calib | hard_calib |
    # robust_mild | robust — band limit / frozen matched-σ noise / operator mismatch.
    obs: ObservationConfig = ObservationConfig()
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
    if cfg.method.misfit != "l2":
        name += f"-{cfg.method.misfit}"
    if not cfg.obs.is_clean:
        name += f"-obs{cfg.obs.min_freq_hz:g}hz" if cfg.obs.min_freq_hz else "-obsnoise"
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name=name)

    # Guidance data-misfit, built lazily from d_obs (the OT potential precomputes its
    # weights/CDFs from the observation). None keeps the samplers' hard-wired L2 path
    # byte-identical for existing runs; a band-limited benchmark needs the factory even
    # for L2 so predictions are filtered before comparison (the guidance-side half of F).
    misfit_factory = (
        None
        if cfg.method.misfit == "l2" and cfg.obs.min_freq_hz <= 0.0
        else lambda d: make_misfit(
            cfg.method.misfit,
            d,
            ot_k=cfg.method.ot_k,
            min_freq_hz=cfg.obs.min_freq_hz,
        )
    )

    # Per-step diagnostic misfits: always log OT alongside the L2 data_fidelity (built lazily
    # from d_obs like misfit_factory), so any run's OT-vs-step trajectory can be plotted from
    # steps.jsonl without re-simulation. Cheap — reuses the prediction already computed each step.
    def build_diag_misfits(d: torch.Tensor) -> dict[str, object]:
        return {
            "ot": make_misfit(
                "ot", d, ot_k=cfg.method.ot_k, min_freq_hz=cfg.obs.min_freq_hz
            )
        }

    # Composite trajectory grid (rows = samples, cols = t=0 → t=1) — the same renderer
    # 0001/0002 log as "trajectory". make_step_saver stacks the snapshots on dim 0 and,
    # for samplers that report their noisy state, interleaves (x_t, estimate) row pairs
    # (0003's layout); frames arrive [n_frames, B, C, H, W] (or without C from the
    # native-grid FWI path).
    def viz_traj(frames: torch.Tensor, path: Path) -> None:
        cfg.dataset.visualize_trajectory(
            frames if frames.ndim == 5 else frames[:, :, None], path
        )

    channels, size, _ = cfg.dataset.shape
    n = cfg.n_samples

    # Shared t=0 noise bank: draw every prior's initial noise from an explicit device
    # Generator seeded with cfg.seed so the n samples are byte-identical across prior
    # families/methods (flow vs diffusion) on *any* device. Previously this held only by
    # accident on CUDA — model construction consumed the CPU RNG stream while the noise was
    # drawn from the untouched CUDA stream; on CPU (or if init RNG usage changed) the two
    # priors diverged. Drawn once here and reused for the guided and unguided passes.
    _noise_gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    def init_noise() -> torch.Tensor:
        return torch.randn(n, channels, size, size, device=dev, generator=_noise_gen)

    n_solves = (
        0  # forward (PDE) solves consumed by the guided pass (matched-cost reporting)
    )
    solves = {
        "n": 0
    }  # actual guided-pass solve count (FWI fills this; used for the figure banner)
    traj_cap: dict[
        str, object
    ] = {}  # sample-0 trajectory frames, filled by the guided step saver

    if cfg.prior.name == "none":
        # Classical / realistic FWI: no learned prior. Optimise the velocity map directly against
        # the data misfit at native 70x70 (the forward operator's grid — dataset.resolution is the
        # prior's training size, irrelevant here). guidance=0 => the starting model (control).
        from physics_informed_flow_map.inversion.bridge import NATIVE
        from physics_informed_flow_map.physics.classical import (
            multiscale_fwi,
            regularized_fwi,
        )
        from physics_informed_flow_map.physics.filters import highpass
        from physics_informed_flow_map.physics.forward import simulate

        realistic = cfg.method.name == "realistic_fwi"

        # Band-limited benchmark: the classical loops compare their own forward solves to
        # d_obs, so their operator gets the same guidance-side high-pass (part of F).
        if cfg.obs.min_freq_hz > 0.0:

            def forward_op(v: torch.Tensor) -> torch.Tensor:
                return highpass(simulate(v), cfg.obs.min_freq_hz, 1e-3)
        else:
            forward_op = simulate

        def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
            run_it = guidance_strength != 0.0
            step_cb = None
            if run_it and cfg.n_frames > 0:
                # multiscale/L-BFGS logs per objective evaluation, so line searches can
                # overrun this total; the frame schedule still hits every planned index.
                total = (
                    len(cfg.method.freqs_hz) * cfg.method.iters_per_stage
                    if realistic
                    else cfg.steps
                )
                step_cb = run.make_step_saver(
                    f"{cfg.method.name}_g{guidance_strength:.2g}",
                    viz_traj,
                    total_steps=total,
                    n_frames=cfg.n_frames,
                    capture=traj_cap,
                )
            if realistic:
                v_mps, ns = multiscale_fwi(
                    forward_op,
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
                    on_step=step_cb,
                )
            else:
                v_mps, ns = regularized_fwi(
                    forward_op,  # forward_fn: (H, W) m/s -> data (band-limited if obs is)
                    d_obs,
                    shape=(NATIVE, NATIVE),
                    n_samples=n,
                    iters=cfg.steps if run_it else 0,
                    lr=cfg.method.lr,
                    reg=cfg.method.reg,
                    reg_weight=cfg.method.reg_weight,
                    init=cfg.method.init,
                    device=dev,
                    on_step=step_cb,
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
            x0_mfm = (
                init_noise()
            )  # shared t=0 noise, reused across guided/unguided passes
            # Per-step captures for the MC-posterior grids: the noisy state x_t and the
            # mc_samples posterior draws x1~p(x1|x_t) the estimator used, EVERY step of the
            # guided pass (fp16 CPU to keep 30x4x200 frames light). Rendered after inversion.
            mc_cap: dict[str, list[torch.Tensor]] = {"xt": [], "mc": []}

            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                estimator = (
                    cfg.method.drift_estimator if guidance_strength != 0.0 else "base"
                )
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    traj_saver = run.make_step_saver(
                        f"{cfg.method.name}_g{guidance_strength:.2g}_mc{cfg.method.mc_samples}",
                        viz_traj,
                        total_steps=cfg.steps,
                        n_frames=cfg.n_frames,
                        capture=traj_cap,
                    )
                    mc_cap["xt"].clear()
                    mc_cap["mc"].clear()

                    def step_cb(
                        step: int,
                        estimate: torch.Tensor,
                        xt: torch.Tensor | None = None,
                        mc_samples: torch.Tensor | None = None,
                        **scalars: float,
                    ) -> None:
                        # Keep the existing trajectory grid + steps.jsonl (mc_samples is a
                        # tensor, so it is captured here and NOT forwarded into scalars).
                        traj_saver(step, estimate, xt=xt, **scalars)
                        if xt is not None:
                            mc_cap["xt"].append(xt.detach().to(torch.float16).cpu())
                        if mc_samples is not None:
                            mc_cap["mc"].append(
                                mc_samples.detach().to(torch.float16).cpu()
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
                    misfit_factory=misfit_factory,
                    diag_misfit_factory=build_diag_misfits,
                    on_step=step_cb,
                    x0=x0_mfm,
                )
                return mps_to_norm(module.invert(d_obs).v_hat)[:, None]

            n_solves = (
                cfg.steps * cfg.method.mc_samples * n
            )  # mc posterior solves per step
        elif cfg.method.name == "fmrg_e":
            # FMRG-E: n_opt inner gradient steps in x1-space per outer Euler step.
            # Uses marginal velocity v(t,t,xt,0,0) — explicit zeros so intent is clear.
            x0 = init_noise()  # shared t=0 noise bank (see init_noise)
            t_cond = torch.zeros(n, device=dev)
            x_cond = torch.zeros(n, channels, size, size, device=dev)

            def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
                tb = torch.full((x.shape[0],), t, device=dev)
                return cast(torch.Tensor, prior.v(tb, tb, x, t_cond, x_cond))

            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    step_cb = run.make_step_saver(
                        f"fmrg_e_g{guidance_strength:.2g}_n{cfg.method.n_opt}",
                        viz_traj,
                        total_steps=cfg.steps,
                        n_frames=cfg.n_frames,
                        capture=traj_cap,
                    )
                return fmrg_e_sample(
                    velocity_fn,
                    x0,
                    seismic_forward,
                    d_obs,
                    sampler_steps=cfg.steps,
                    guidance_strength=guidance_strength,
                    n_opt=cfg.method.n_opt,
                    normalize_grad=cfg.method.normalize_grad,
                    misfit_fn=misfit_factory(d_obs) if misfit_factory else None,
                    on_step=step_cb,
                )

            n_solves = (
                cfg.steps * cfg.method.n_opt * n
            )  # n_opt wave solves per step per sample
        else:
            # DPS Tweedie baseline (flow_tilt) / unguided control: tilt from a fixed noise context
            # x0 (reused guided/unguided), using v(t,t,x|noise) at t_cond=0.
            x0 = init_noise()  # shared t=0 noise bank (see init_noise)
            t_cond = torch.zeros(n, device=dev)

            def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
                tb = torch.full((x.shape[0],), t, device=dev)
                return cast(torch.Tensor, prior.v(tb, tb, x, t_cond, x0))

            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    step_cb = run.make_step_saver(
                        f"flow_tilt_g{guidance_strength:.2g}",
                        viz_traj,
                        total_steps=cfg.steps,
                        n_frames=cfg.n_frames,
                        capture=traj_cap,
                    )
                return guided_sample(
                    velocity_fn,
                    x0,
                    seismic_forward,
                    d_obs,
                    sampler_steps=cfg.steps,
                    guidance_strength=guidance_strength,
                    normalize_grad=cfg.method.normalize_grad,
                    misfit_fn=misfit_factory(d_obs) if misfit_factory else None,
                    diag_misfit_fns=build_diag_misfits(d_obs),
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
                        viz_traj,
                        total_steps=cfg.method.iters or cfg.steps,
                        n_frames=cfg.n_frames,
                        capture=traj_cap,
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
                    misfit_fn=misfit_factory(d_obs) if misfit_factory else None,
                    on_step=step_cb,
                )
        else:  # dps / unguided
            x_init = (
                init_noise()
            )  # shared t=T noise, reused across guided/unguided passes

            def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
                step_cb = None
                if guidance_strength != 0.0 and cfg.n_frames > 0:
                    step_cb = run.make_step_saver(
                        f"dps_g{guidance_strength:.2g}",
                        viz_traj,
                        total_steps=cfg.steps,
                        n_frames=cfg.n_frames,
                        capture=traj_cap,
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
                    misfit_fn=misfit_factory(d_obs) if misfit_factory else None,
                    diag_misfit_fns=build_diag_misfits(d_obs),
                    on_step=step_cb,
                    x_init=x_init,
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
    obs_out = (
        run.ckpt_dir.parent / "d_obs"
    )  # base; invert_and_report writes _linear/_log.png
    recon_out = run.ckpt_dir.parent / "reconstructions.npz"
    dobs_cmp_out = run.ckpt_dir.parent / "d_obs_inverted_vs_true.png"
    dobs_cmp_npz = run.ckpt_dir.parent / "d_obs_inverted_sample0.npz"
    cmp_label = (
        f"{_PRIOR_DISP.get(cfg.prior.name, cfg.prior.name)} + "
        f"{_EST_DISP.get(cfg.method.name, cfg.method.name)} + {cfg.method.misfit.upper()}"
    )

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
        target=cfg.target or None,
        method_name=cfg.method.name,
        guidance=cfg.method.guidance_strength,
        steps=cfg.steps,
        device=dev,
        out_png=out,
        out_obs_png=obs_out,
        out_npy=recon_out,
        out_dobs_cmp_png=dobs_cmp_out,
        out_dobs_cmp_npz=dobs_cmp_npz,
        cmp_label=cmp_label,
        cost=solve_count,
        misfit_factory=misfit_factory,
        obs_cfg=cfg.obs,
        traj_capture=traj_cap,
        dobs_scales=("linear", "log"),
    )
    if cfg.prior.name == "none":
        n_solves = solves["n"]  # actual forward solves from the guided FWI pass
    summary["inv/n_solves"] = float(0 if cfg.method.name == "unguided" else n_solves)
    run.log_image("inversion", out, caption=caption)
    run.log_image(
        "d_obs_inverted_vs_true", dobs_cmp_out, caption=f"data-space fit · {caption}"
    )
    # d_obs and the Tweedie-d_obs trajectory grid, each in linear + symlog (invert_and_report
    # wrote {stem}_{scale}.png; names mirrored here for wandb logging).
    gs_log = cfg.method.guidance_strength if cfg.method.name != "unguided" else 0.0
    for sc in ("linear", "log"):
        dobs_png = run.ckpt_dir.parent / f"d_obs_{sc}.png"
        if dobs_png.exists():
            run.log_image(
                f"d_obs_{sc}", dobs_png, caption=f"observed seismic ({sc}) · {caption}"
            )
        traj_png = (
            run.ckpt_dir.parent / f"{cfg.method.name}_g{gs_log:.2g}_dobs_traj_{sc}.png"
        )
        if traj_png.exists():
            run.log_image(
                f"traj/dobs_{sc}",
                traj_png,
                caption=f"d_obs trajectory ({sc}) · {caption}",
            )

    # MFM MC-posterior grids: one tall static grid per particle (rows = every step, cols =
    # noisy x_t + the mc_samples posterior draws that step). Only the iwae/sne estimators
    # populate mc_cap; dps/base draw no MC set, so this is skipped for them.
    if cfg.method.name in _MFM_METHODS and mc_cap["mc"]:
        from physics_informed_flow_map.inversion.mc_posterior_viz import (
            render_mc_posterior_grids,
        )

        xt = torch.stack(mc_cap["xt"]).float().numpy()  # (steps, n, H, W)
        mc = torch.stack(mc_cap["mc"]).float().numpy()  # (steps, n, mc, H, W)
        mc_dir = run.ckpt_dir.parent / "mc_posterior_viz"
        paths = render_mc_posterior_grids(xt, mc, mc_dir, title_prefix=cmp_label)
        print(f"[0004] wrote {len(paths)} MC-posterior grids -> {mc_dir}")
        run.log_image(
            "mc_posterior/particle_00", paths[0], caption=f"MC grid · {caption}"
        )

    run.log(**summary)  # mirror to metrics.jsonl on the network volume
    run.finish(**summary)


if __name__ == "__main__":
    main()
