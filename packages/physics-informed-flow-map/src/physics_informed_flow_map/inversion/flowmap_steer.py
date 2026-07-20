"""Native Meta-Flow-Map steering for FWI — the inversion that avoids the Tweedie approximation.

Our :class:`~.modules.FlowTiltModule` is the DPS baseline: it steers with the single-point Tweedie
estimate ``x1_hat = xt + (1-t) v``. MFM's contribution is to replace that with a Monte-Carlo
estimate of the reward-tilted drift, where the *flow map* draws genuine samples ``x1 ~ p(x1 | xt)``
by pushing fresh noise through the conditional map ``v(0, 1, noise | t_cond=t, x_cond=xt)`` — no
Tweedie mean. The reward for FWI is the data log-likelihood ``-||F(x1) - d_obs||^2 / (2 sigma^2)``.

This reuses mfm's steering drift verbatim (``get_conditional_drift_fn``); the Euler / Euler-
Maruyama stepping loop is inlined here (numerically identical to mfm's samplers, same RNG draw
order) so an ``on_step`` hook can observe each step. The only FWI-specific glue is the reward and
the norm->m/s ``inverse_scaler``. ``drift_estimator`` selects the estimator: ``dps`` (Tweedie
baseline), ``iwae`` (importance-weighted, backprops the reward), or ``sne`` (self-normalized, no
backprop). ``mc_samples`` is the number of ``xt -> x1`` draws per step.
"""

from __future__ import annotations

from typing import Callable

import torch
from mfm.models.base_model import BaseModel
from mfm.utils.steering import (
    broadcast_to_shape,
    get_conditional_drift_fn,
    sigma_t_sq,
)
from torch import Tensor
from tqdm import tqdm

from ..physics.forward import simulate
from ..physics.misfit import MisfitFn, l2_misfit
from .base import InversionResult
from .bridge import mps_to_norm, to_mps_native


def make_misfit_reward(
    d_obs: Tensor, sigma: float, misfit_fn: MisfitFn | None = None
) -> Callable[[Tensor], Tensor]:
    """Reward ``x1 (m/s) -> -misfit(F(x1)) / (2 sigma^2)`` for a ``(N, H, W)`` batch.

    ``sigma`` is the likelihood temperature: smaller ties the reward harder to the data (and
    collapses the importance weights onto the single best sample sooner). ``misfit_fn``
    (see ``physics.misfit``) defaults to the pointwise L2 ``||F(x1) - d_obs||^2`` — the exact
    Gaussian log-likelihood; a non-L2 misfit makes this a generalized (Gibbs) posterior
    reward, and note the numerical scale changes (the OT potential is O(1) where the L2
    misfit is huge, so ``sigma`` needs retuning)."""
    if misfit_fn is None:
        misfit_fn = l2_misfit(d_obs)

    def reward_fn(x1_mps: Tensor) -> Tensor:
        pred = torch.stack([simulate(x1_mps[i]) for i in range(x1_mps.shape[0])])
        return -misfit_fn(pred) / (2.0 * sigma**2)

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
        misfit_factory: Callable[[Tensor], MisfitFn] | None = None,
        diag_misfit_factory: Callable[[Tensor], dict[str, MisfitFn]] | None = None,
        on_step: Callable[..., None] | None = None,
        x0: Tensor | None = None,
    ) -> None:
        self.name = f"flowmap_steer_{drift_estimator}·mc{mc_samples}"
        # d_obs arrives per-invert, so the misfit (which precomputes from it) is built lazily.
        self.misfit_factory = misfit_factory
        # Named diagnostic misfits (e.g. OT) built lazily from d_obs and logged per step.
        self.diag_misfit_factory = diag_misfit_factory
        self._diag: dict[str, MisfitFn] = {}
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
        # Optional shared t=0 noise bank (device-robust reproducibility across priors);
        # None => draw fresh noise from the global RNG in invert().
        self.x0 = x0
        # on_step(step, x_est_norm, data_fidelity=..., drift_norm=..., steering_norm=...) —
        # x_est_norm is the per-step Tweedie estimate in [-1, 1] at native resolution. The
        # data_fidelity diagnostic stays raw L2 (comparable across misfits/methods) and costs
        # one extra wave solve per posterior sample per step, only when the hook is set.
        self.on_step = on_step

    def invert(self, d_obs: Tensor, invert_scalar=None) -> InversionResult:
        misfit_fn = self.misfit_factory(d_obs) if self.misfit_factory else None
        self._diag = self.diag_misfit_factory(d_obs) if self.diag_misfit_factory else {}
        drift_fn = get_conditional_drift_fn(
            self.prior,
            make_misfit_reward(d_obs, self.sigma, misfit_fn),
            to_mps_native,  # inverse_scaler: normalized [-1,1] -> m/s (and resize to native)
            type="sde" if self.sde else "ode",
            drift_estimator=self.drift_estimator,
            mc_samples=self.mc_samples,
            guidance_scale=self.guidance_scale,
            renorm_gradient=self.renorm,
        )
        x0 = (
            self.x0
            if self.x0 is not None
            else torch.randn(
                self.n_samples, 1, self.resolution, self.resolution, device=self.device
            )
        )
        x1 = self._sample(x0, drift_fn, d_obs)
        # Wave solves: mc_samples reward sims per step per posterior sample (base drift is a
        # network eval, not a solve). dps uses one Tweedie sim per step.

        return InversionResult(torch.squeeze(x1), to_mps_native(x1))

    def _sample(
        self,
        x0: Tensor,
        drift_fn: Callable[..., tuple],
        d_obs: Tensor,
        t_start: float = 0.01,
    ) -> Tensor:
        """Euler (ODE) / Euler-Maruyama (SDE) stepping, identical to mfm's samplers (same
        ``linspace`` grid and, for the SDE, the same one-``randn_like``-per-step draw order,
        so trajectories match the originals bitwise), plus the :attr:`on_step` hook."""
        n = x0.shape[0]
        x = x0
        t_steps = torch.linspace(t_start, 1.0, self.n_steps + 1, device=x.device)

        desc = "SDE sampling" if self.sde else "ODE sampling"
        for i in tqdm(range(self.n_steps), desc=desc):
            t_cur = t_steps[i]
            dt = t_steps[i + 1] - t_cur
            t_batched = torch.full((n,), t_cur, device=x.device)
            drift, ret = drift_fn(x, t_batched)

            if self.on_step is not None:
                est_mps = ret["tweedie_estimate"]  # (n, H, W) m/s, native resolution
                pred = torch.stack([simulate(est_mps[b]) for b in range(n)])
                # Noisy sampler state for the trajectory viz, resized (model resolution
                # -> native) to sit alongside the (n, H, W) Tweedie-estimate rows.
                xt = torch.nn.functional.interpolate(
                    x.detach(), size=est_mps.shape[-2:], mode="bilinear"
                )[:, 0]
                # The mc_samples posterior draws x1~p(x1|xt) the iwae/sne estimator used this
                # step (native-res m/s -> [-1,1]); (n, mc, H, W). None for dps/base.
                mc = ret.get("mc_x1_data")
                mc_norm = mps_to_norm(mc) if mc is not None else None
                diag = {
                    f"misfit_{k}": float(fn(pred).mean())
                    for k, fn in self._diag.items()
                }
                self.on_step(
                    i,
                    mps_to_norm(est_mps),
                    xt=xt,
                    mc_samples=mc_norm,
                    data_fidelity=float(((pred - d_obs) ** 2).mean()),
                    drift_norm=float(ret["uncond_drift"].flatten(1).norm(dim=1).mean()),
                    steering_norm=float(
                        ret["steering_drift_scaled"].flatten(1).norm(dim=1).mean()
                    ),
                    **diag,
                )

            x = x + drift * dt
            if self.sde:
                diffusion = broadcast_to_shape(
                    torch.sqrt(sigma_t_sq(t_batched)), x.shape
                )
                x = x + diffusion * torch.sqrt(dt) * torch.randn_like(x)
        return x
