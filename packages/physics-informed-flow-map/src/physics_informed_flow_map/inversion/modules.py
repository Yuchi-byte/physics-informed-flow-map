"""Concrete inversion modules: the flow-tilting and diffusion-DPS inverters as one uniform
interface for the evaluator. Each wraps an already-loaded prior (I/O stays at the edges — the
caller loads the checkpoint) plus its guidance scheme and hyperparameters.
"""

from __future__ import annotations

from typing import cast

import torch
from diffusers import DDPMScheduler
from mfm.models.base_model import BaseModel
from torch import Tensor, nn

from ..baselines import dps_sample, red_diffeq_sample
from ..physics.classical import multiscale_fwi, regularized_fwi
from ..physics.forward import simulate
from ..physics.tilt import guided_sample
from .base import InversionResult
from .bridge import seismic_forward, to_mps_native

NATIVE = 70  # OpenFWI velocity-map resolution the wave solver runs on


class FlowTiltModule:
    """DPS-style tilting of a flow / flow-map prior (the 0001 & 0002 inversion method)."""

    def __init__(
        self,
        prior: BaseModel,
        *,
        guidance: float,
        steps: int,
        n_samples: int,
        device: torch.device,
        resolution: int = 64,
        normalize_grad: bool = True,
    ) -> None:
        self.name = f"flow_tilt·g{guidance:g}"
        self.prior = prior
        self.guidance = guidance
        self.steps = steps
        self.n_samples = n_samples
        self.device = device
        self.resolution = resolution
        self.normalize_grad = normalize_grad

    def invert(self, d_obs: Tensor) -> InversionResult:
        b = self.n_samples
        x0 = torch.randn(b, 1, self.resolution, self.resolution, device=self.device)
        t_cond = torch.zeros(b, device=self.device)

        def velocity_fn(x: Tensor, t: float) -> Tensor:
            tb = torch.full((x.shape[0],), t, device=self.device)
            return cast(Tensor, self.prior.v(tb, tb, x, t_cond, x0))

        samples = guided_sample(
            velocity_fn,
            x0,
            seismic_forward,
            d_obs,
            sampler_steps=self.steps,
            guidance_strength=self.guidance,
            normalize_grad=self.normalize_grad,
        )
        return InversionResult(to_mps_native(samples), n_solves=self.steps * b)


class ClassicalFWIModule:
    """Classical regularized FWI (no learned prior) — gradient descent on the velocity model
    with a Tikhonov or TV penalty, steered by the same wave equation."""

    def __init__(
        self,
        *,
        reg: str,
        reg_weight: float,
        iters: int,
        lr: float,
        n_samples: int,
        device: torch.device,
        init: str = "smooth",
        native: int = NATIVE,
    ) -> None:
        self.name = f"classical_{reg}·w{reg_weight:g}"
        self.reg = reg
        self.reg_weight = reg_weight
        self.iters = iters
        self.lr = lr
        self.n_samples = n_samples
        self.device = device
        self.init = init
        self.native = native

    def invert(self, d_obs: Tensor) -> InversionResult:
        v_mps, n_solves = regularized_fwi(
            simulate,
            d_obs,
            shape=(self.native, self.native),
            n_samples=self.n_samples,
            iters=self.iters,
            lr=self.lr,
            reg=self.reg,
            reg_weight=self.reg_weight,
            init=self.init,
            device=self.device,
        )
        return InversionResult(v_mps, n_solves=n_solves)


class RealisticFWIModule:
    """Properly-run classical FWI (no learned prior) — smooth 1-D start, multiscale frequency
    continuation, regularisation, L-BFGS with line search; steered by the same wave equation."""

    def __init__(
        self,
        *,
        freqs_hz: list[float],
        iters_per_stage: int,
        lr: float,
        reg: str,
        reg_weight: float,
        optimizer: str,
        n_samples: int,
        device: torch.device,
        native: int = NATIVE,
    ) -> None:
        self.name = f"realistic_{optimizer}·w{reg_weight:g}"
        self.freqs_hz = freqs_hz
        self.iters_per_stage = iters_per_stage
        self.lr = lr
        self.reg = reg
        self.reg_weight = reg_weight
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.device = device
        self.native = native

    def invert(self, d_obs: Tensor) -> InversionResult:
        v_mps, n_solves = multiscale_fwi(
            simulate,
            d_obs,
            shape=(self.native, self.native),
            n_samples=self.n_samples,
            freqs_hz=self.freqs_hz,
            iters_per_stage=self.iters_per_stage,
            lr=self.lr,
            reg=self.reg,
            reg_weight=self.reg_weight,
            optimizer=self.optimizer,
            device=self.device,
        )
        return InversionResult(v_mps, n_solves=n_solves)


class DiffusionDPSModule:
    """Canonical DPS over a diffusion prior (the ``0003`` baseline)."""

    def __init__(
        self,
        denoiser: nn.Module,
        scheduler: DDPMScheduler,
        *,
        guidance: float,
        steps: int,
        n_samples: int,
        device: torch.device,
        resolution: int = 64,
        normalize_grad: bool = True,
    ) -> None:
        self.name = f"diffusion_dps·g{guidance:g}"
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.guidance = guidance
        self.steps = steps
        self.n_samples = n_samples
        self.device = device
        self.resolution = resolution
        self.normalize_grad = normalize_grad

    def invert(self, d_obs: Tensor) -> InversionResult:
        samples = dps_sample(
            self.denoiser,
            self.scheduler,
            (1, self.resolution, self.resolution),
            seismic_forward,
            d_obs,
            n_samples=self.n_samples,
            num_steps=self.steps,
            guidance_strength=self.guidance,
            device=self.device,
            normalize_grad=self.normalize_grad,
        )
        return InversionResult(
            to_mps_native(samples), n_solves=self.steps * self.n_samples
        )


class REDDiffEqModule:
    """RED-DiffEq: the diffusion prior as a Regularization-by-Denoising term in a
    wave-equation-steered optimization (an alternative inference scheme to DPS)."""

    def __init__(
        self,
        denoiser: nn.Module,
        scheduler: DDPMScheduler,
        *,
        eta_data: float,
        eta_reg: float,
        t_denoise: int,
        iters: int,
        n_samples: int,
        device: torch.device,
        resolution: int = 64,
        normalize_grad: bool = True,
    ) -> None:
        self.name = f"red_diffeq·d{eta_data:g}·r{eta_reg:g}"
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.eta_data = eta_data
        self.eta_reg = eta_reg
        self.t_denoise = t_denoise
        self.iters = iters
        self.n_samples = n_samples
        self.device = device
        self.resolution = resolution
        self.normalize_grad = normalize_grad

    def invert(self, d_obs: Tensor) -> InversionResult:
        samples = red_diffeq_sample(
            self.denoiser,
            self.scheduler,
            (1, self.resolution, self.resolution),
            seismic_forward,
            d_obs,
            n_samples=self.n_samples,
            iters=self.iters,
            eta_data=self.eta_data,
            eta_reg=self.eta_reg,
            t_denoise=self.t_denoise,
            device=self.device,
            normalize_grad=self.normalize_grad,
        )
        return InversionResult(
            to_mps_native(samples), n_solves=self.iters * self.n_samples
        )
