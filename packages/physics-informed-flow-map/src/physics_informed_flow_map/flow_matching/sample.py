"""Sampling from a flow prior: the multi-step ODE sampler, and the few-step flow-map
(consistency) sampler that only a meta flow map supports."""

from __future__ import annotations

from typing import Sequence, cast

import torch
from torch import Tensor
from torchdiffeq import odeint

from mfm.SI.samplers import consistency_sampler_fn, ode_sampler_fn
from mfm.models.base_model import BaseModel


@torch.no_grad()
def sample(
    model: BaseModel,
    n_samples: int,
    shape: tuple[int, ...],
    *,
    sampler_steps: int,
    device: torch.device,
    x_noise: Tensor | None = None,
    return_states_at: Sequence[float] | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sample from the flow prior via Euler ODE integration (torchdiffeq).

    Pass ``x_noise`` to fix the starting noise across calls (e.g. for reproducible
    per-epoch visualizations that track the same samples through training).

    Pass ``return_states_at`` (times that must land on the Euler grid, i.e. integer
    multiples of ``1/sampler_steps``) to also get the intermediate ODE states at those
    times — the return becomes ``(samples, states)`` with ``states`` of shape
    ``[len(times), B, *shape]``. Same integrator and grid either way, so the samples
    are identical to the plain call.
    """
    model.eval()
    if x_noise is None:
        x_noise = torch.randn(n_samples, *shape, device=device)
    else:
        x_noise = x_noise.to(device)
    t_cond = torch.zeros(x_noise.shape[0], device=device)
    if return_states_at is None:
        return cast(
            Tensor,
            ode_sampler_fn(
                model,
                xt_cond=x_noise,
                t_cond=t_cond,
                n_steps=sampler_steps,
                solver="euler",
                eps_start=x_noise,
                v_type="standard",
            ),
        )

    indices = []
    for t in return_states_at:
        idx = t * sampler_steps
        if abs(idx - round(idx)) > 1e-9:
            raise ValueError(
                f"return_states_at time {t} is not on the Euler grid "
                f"(sampler_steps={sampler_steps})"
            )
        indices.append(round(idx))

    def ode_func(t: Tensor, x: Tensor) -> Tensor:
        tb = t.expand(x.shape[0])
        return model.v(tb, tb, x, t_cond, x_noise)

    times = torch.linspace(0, 1, sampler_steps + 1, device=device)
    hist = odeint(ode_func, x_noise, times, method="euler", atol=1e-5, rtol=1e-5)
    return hist[-1], hist[indices]


@torch.no_grad()
def sample_trajectory(
    model: BaseModel,
    x_noise: Tensor,
    *,
    sampler_steps: int,
    device: torch.device,
    n_frames: int = 6,
) -> tuple[Tensor, Tensor]:
    """Euler ODE integration returning ``n_frames`` evenly-spaced intermediate snapshots.

    Returns a pair ``(states, x1hats)``, each of shape ``[n_frames, B, *shape]``, from t=0
    (pure noise) to t=1 (generated sample): the transported ODE state ``x_t``, and the
    one-step clean estimate ``x1hat = x_t + (1-t) * v(t, t, x_t)`` (Euler extrapolation to
    the data endpoint — the flow analogue of the DDPM Tweedie ``x0hat`` in
    :func:`baselines.ddpm_sample_trajectory`; at t=1 it equals the state). Pass a fixed
    ``x_noise`` across epochs to track the same starting points as the model improves.
    """
    model.eval()
    x_noise = x_noise.to(device)
    t_cond = torch.zeros(x_noise.shape[0], device=device)

    def ode_func(t: Tensor, x: Tensor) -> Tensor:
        tb = t.expand(x.shape[0])
        return model.v(tb, tb, x, t_cond, x_noise)

    times = torch.linspace(0, 1, sampler_steps + 1, device=device)
    hist = odeint(ode_func, x_noise, times, method="euler", atol=1e-5, rtol=1e-5)
    # hist: [sampler_steps+1, B, *shape] — pick n_frames evenly-spaced indices
    indices = [round(i * sampler_steps / (n_frames - 1)) for i in range(n_frames)]
    states = hist[indices]  # [n_frames, B, *shape]
    x1hats = [hist[i] + (1.0 - times[i]) * ode_func(times[i], hist[i]) for i in indices]
    return states, torch.stack(x1hats)


@torch.no_grad()
def sample_few_step(
    model: BaseModel,
    n_samples: int,
    shape: tuple[int, ...],
    *,
    n_steps: int,
    device: torch.device,
    x_noise: Tensor | None = None,
    return_hist: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Few-step flow-map sampling: walk ``s→u`` in ``n_steps`` jumps via ``model.X(s,u,x,v)``.

    Meaningful only for a trained meta flow map (``0002``); on a pure flow-matching prior the
    off-diagonal ``v(s,u)`` is unconstrained, so few-step samples will be poor.

    Pass ``x_noise`` to fix the starting noise across calls. Sharing the same ``x_noise`` with
    :func:`sample` makes the few-step grid directly comparable, cell-for-cell, to the ODE grid
    (same starting points, different sampler) and reproducible across epochs.

    Pass ``return_hist=True`` to also get the intermediate jump states — the return becomes
    ``(samples, hist)`` with ``hist`` of shape ``[n_steps+1, B, *shape]`` at the junction
    times ``k/n_steps``. Same jump chain either way, so the samples are identical.
    """
    model.eval()
    if x_noise is None:
        x_noise = torch.randn(n_samples, *shape, device=device)
    else:
        x_noise = x_noise.to(device)
    t_cond = torch.zeros(x_noise.shape[0], device=device)
    if not return_hist:
        return cast(
            Tensor,
            consistency_sampler_fn(
                model,
                xt_cond=x_noise,
                t_cond=t_cond,
                n_steps=n_steps,
                eps_start=x_noise,
            ),
        )

    times = torch.linspace(0, 1, n_steps + 1, device=device)
    hist = torch.empty((n_steps + 1, *x_noise.shape), device=device)
    hist[0] = x_noise
    for i in range(n_steps):
        s = times[i].expand(x_noise.shape[0])
        u = times[i + 1].expand(x_noise.shape[0])
        hist[i + 1] = model(s, u, hist[i], t_cond, x_noise)
    return hist[-1], hist


def _rel_sq_err(pred: Tensor, target: Tensor) -> float:
    """Batch mean of the per-sample relative squared error ``||pred-target||²/||target||²``."""
    err = (pred - target).flatten(1).pow(2).sum(dim=1)
    norm = target.flatten(1).pow(2).sum(dim=1).clamp_min(1e-12)
    return float((err / norm).mean())


@torch.no_grad()
def flow_map_consistency(
    model: BaseModel,
    x_noise: Tensor,
    ode_states: Tensor,
    few_hist: Tensor,
) -> dict[str, float]:
    """Quantify off-diagonal ``v(s,u)`` quality by self-consistency against the fine ODE.

    Both inputs are trajectories from the *same* ``x_noise``: ``ode_states`` are the fine-ODE
    (diagonal ``v(t,t)``) reference states and ``few_hist`` the chained flow-map jumps, each of
    shape ``[K+1, B, *shape]`` at the junction times ``k/K`` (``sample(...,
    return_states_at=...)`` and ``sample_few_step(..., return_hist=True)``). Internal
    consistency only — how well the off-diagonal matches integrating the model's own diagonal —
    not distance to the true velocity field; that is exactly the semantics of the ``s<u``
    training term, so it is its validation counterpart.

    Returned metrics (all relative squared errors):
      * ``fewstep_ode_gap`` — chained few-step endpoint vs ODE endpoint (headline; the
        quantitative gap between the few-step and ODE sample grids), with per-junction
        ``fewstep_ode_gap_t{k/K}`` for the intermediate times. Accumulates over jumps.
      * ``jump_consistency`` — mean teacher-forced single-jump error: ``X(t_k, t_{k+1}, ·)``
        applied to the *ODE* state, vs the next ODE state. Isolates per-jump error.
      * ``jump_consistency_full`` — the single ``0→1`` jump from noise vs the ODE endpoint
        (the pure one-step-generation case, the hardest jump).
    """
    model.eval()
    n_jumps = ode_states.shape[0] - 1
    batch = x_noise.shape[0]
    device = x_noise.device
    t_cond = torch.zeros(batch, device=device)
    times = torch.linspace(0, 1, n_jumps + 1, device=device)

    metrics: dict[str, float] = {}
    for k in range(1, n_jumps):
        metrics[f"fewstep_ode_gap_t{k / n_jumps:g}"] = _rel_sq_err(
            few_hist[k], ode_states[k]
        )
    metrics["fewstep_ode_gap"] = _rel_sq_err(few_hist[n_jumps], ode_states[n_jumps])

    jump_errs = []
    for k in range(n_jumps):
        s = times[k].expand(batch)
        u = times[k + 1].expand(batch)
        pred = model(s, u, ode_states[k], t_cond, x_noise)
        jump_errs.append(_rel_sq_err(pred, ode_states[k + 1]))
    metrics["jump_consistency"] = sum(jump_errs) / len(jump_errs)

    zeros = torch.zeros(batch, device=device)
    ones = torch.ones(batch, device=device)
    pred_full = model(zeros, ones, x_noise, t_cond, x_noise)
    metrics["jump_consistency_full"] = _rel_sq_err(pred_full, ode_states[n_jumps])
    return metrics


@torch.no_grad()
def sample_posterior(
    model: BaseModel,
    refs: Tensor,
    t_cond: float,
    *,
    n_steps: int,
    device: torch.device,
) -> Tensor:
    """Draw a posterior sample $x_1\\sim p(x_1\\mid x_{t})$ per reference.

    Builds the partially-noised intermediate state $x_t=(1-t)\\epsilon+t\\,x_1$ at ``t_cond``
    from each reference, then conditions the flow map on it. A **time-conditional** flow map
    (trained with ``t_cond>0``) reconstructs the reference — sharper as ``t_cond``→1; an
    unconditional map ignores the conditioning and returns generic samples. The test that the
    ``t_cond>0`` training actually took.
    """
    model.eval()
    refs = refs.to(device)
    tb = torch.full((refs.shape[0],) + (1,) * (refs.ndim - 1), t_cond, device=device)
    xt_cond = (1 - tb) * torch.randn_like(refs) + tb * refs
    t_cond_vec = torch.full((refs.shape[0],), t_cond, device=device)
    return cast(
        Tensor,
        consistency_sampler_fn(
            model,
            xt_cond=xt_cond,
            t_cond=t_cond_vec,
            n_steps=n_steps,
            eps_start=torch.randn_like(refs),
        ),
    )
