"""Native Meta-Flow-Map steering for FWI — the inversion that avoids the Tweedie approximation.

Our :class:`~.modules.FlowTiltModule` is the DPS baseline: it steers with the single-point Tweedie
estimate ``x1_hat = xt + (1-t) v``. MFM's contribution is to replace that with a Monte-Carlo
estimate of the reward-tilted drift, where the *flow map* draws genuine samples ``x1 ~ p(x1 | xt)``
by pushing fresh noise through the conditional map ``v(0, 1, noise | t_cond=t, x_cond=xt)`` — no
Tweedie mean. The reward for FWI is the data log-likelihood ``-||F(x1) - d_obs||^2 / (2 sigma^2)``.

This reuses mfm's steering helpers verbatim (``get_conditional_drift_fn`` + the Euler / Euler-
Maruyama samplers); the only FWI-specific glue is the reward and the norm->m/s ``inverse_scaler``.
``drift_estimator`` selects the estimator: ``dps`` (Tweedie baseline), ``iwae`` (importance-
weighted, backprops the reward), or ``sne`` (self-normalized, no backprop). ``mc_samples`` is the
number of ``xt -> x1`` draws per step.
"""

from __future__ import annotations

from typing import Callable

import torch
from mfm.models.base_model import BaseModel
from mfm.utils.steering import (
    euler_maruyama_sampler,
    euler_sampler,
    get_conditional_drift_fn,
)
from torch import Tensor

from ..physics.forward import simulate
from .base import InversionResult
from .bridge import to_mps_native


def make_misfit_reward(d_obs: Tensor, sigma: float) -> Callable[[Tensor], Tensor]:
    """Reward ``x1 (m/s) -> -||F(x1) - d_obs||^2 / (2 sigma^2)`` for a ``(N, H, W)`` batch.

    ``sigma`` is the likelihood temperature: smaller ties the reward harder to the data (and
    collapses the importance weights onto the single best sample sooner)."""

    def reward_fn(x1_mps: Tensor) -> Tensor:
        pred = torch.stack([simulate(x1_mps[i]) for i in range(x1_mps.shape[0])])
        return -((pred - d_obs) ** 2).flatten(1).sum(1) / (2.0 * sigma**2)

    return reward_fn


class FlowMapSteerModule:
    """MFM-G steering of a flow-map prior toward seismic data (the native, non-Tweedie inverter).

    Reuses ``mfm.utils.steering`` for the drift and sampler; ``drift_estimator`` chooses
    ``dps``/``iwae``/``sne`` and ``mc_samples`` the per-step ``xt -> x1`` draw count.
    """

    def __init__(
        self,
        prior: BaseModel,
        *,
        drift_estimator: str,
        mc_samples: int,
        sigma: float,
        n_steps: int,
        n_samples: int,
        device: torch.device,
        guidance_scale: float = 1.0,
        renorm: bool = True,
        sde: bool = True,
        resolution: int = 64,
    ) -> None:
        self.name = f"flowmap_steer_{drift_estimator}·mc{mc_samples}"
        self.prior = prior
        self.drift_estimator = drift_estimator
        self.mc_samples = mc_samples
        self.sigma = sigma
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.device = device
        self.guidance_scale = guidance_scale
        # renorm=True pins the steering to guidance_scale*||base_drift|| every step (stable but
        # prevents self-attenuation); renorm=False lets the raw reward-gradient magnitude (set by
        # sigma) decay naturally as samples fit the data.
        self.renorm = renorm
        self.sde = sde
        self.resolution = resolution

    def invert(self, d_obs: Tensor) -> InversionResult:
        drift_fn = get_conditional_drift_fn(
            self.prior,
            make_misfit_reward(d_obs, self.sigma),
            to_mps_native,  # inverse_scaler: normalized [-1,1] -> m/s (and resize to native)
            type="sde" if self.sde else "ode",
            drift_estimator=self.drift_estimator,
            mc_samples=self.mc_samples,
            guidance_scale=self.guidance_scale,
            renorm_gradient=self.renorm,
        )
        x0 = torch.randn(
            self.n_samples, 1, self.resolution, self.resolution, device=self.device
        )
        sampler = euler_maruyama_sampler if self.sde else euler_sampler
        x1 = sampler(x0, drift_fn, t_start=0.01, n_steps=self.n_steps)
        # Wave solves: mc_samples reward sims per step per posterior sample (base drift is a
        # network eval, not a solve). dps uses one Tweedie sim per step.
        per = 1 if self.drift_estimator == "dps" else self.mc_samples
        return InversionResult(
            to_mps_native(x1), n_solves=self.n_steps * per * self.n_samples
        )
