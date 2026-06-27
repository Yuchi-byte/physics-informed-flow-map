"""Classical regularized FWI — the no-learned-prior baseline.

Recovers a velocity model by gradient descent on the model itself: minimise the data misfit
``||forward(v) - d_obs||^2`` plus a hand-designed smoothness regulariser ``reg_weight * R(v)``,
using the differentiable Deepwave operator for the gradient. This is the classical camp the
generative priors are meant to beat — same wave-equation steering, but the only prior is a
Tikhonov (squared-gradient) or total-variation penalty rather than a learned distribution.

The model is parametrised in the normalised ``[-1, 1]`` velocity space (so the data and
regularisation terms are both O(1) and the step sizes are interpretable) and mapped to m/s for
the forward operator. The optimiser core is pure: it takes a ``forward_fn`` callable, so it is
testable without the wave solver.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ..flow_matching.openfwi import VMAX, VMIN

ForwardFn = Callable[[Tensor], Tensor]  # (H, W) m/s -> data

REGULARIZERS = ("tikhonov", "tv")


def _grad_xy(v: Tensor) -> tuple[Tensor, Tensor]:
    """Forward finite differences of ``(..., H, W)``, each cropped to the ``(H-1, W-1)``
    interior so the two directions align."""
    dvy = (v[..., 1:, :] - v[..., :-1, :])[..., :, :-1]
    dvx = (v[..., :, 1:] - v[..., :, :-1])[..., :-1, :]
    return dvx, dvy


def regularization(v: Tensor, kind: str, *, tv_eps: float = 1e-3) -> Tensor:
    """Scalar roughness penalty summed over a ``(B, H, W)`` batch.

    ``tikhonov`` is the squared gradient magnitude (penalises all roughness, smooths edges);
    ``tv`` is the isotropic total variation ``sqrt(dvx^2 + dvy^2 + eps^2)`` (edge-preserving).
    """
    dvx, dvy = _grad_xy(v)
    if kind == "tikhonov":
        return (dvx**2 + dvy**2).sum()
    if kind == "tv":
        return torch.sqrt(dvx**2 + dvy**2 + tv_eps**2).sum()
    raise ValueError(f"unknown regularizer {kind!r} (one of {REGULARIZERS})")


def _to_mps(x_norm: Tensor) -> Tensor:
    """``[-1, 1]`` -> m/s (the forward operator's units)."""
    return (x_norm + 1.0) / 2.0 * (VMAX - VMIN) + VMIN


def linear_gradient_init(
    n: int,
    h: int,
    w: int,
    device: torch.device,
    *,
    top: float = -0.7,
    bottom: float = 0.3,
) -> Tensor:
    """``(n, H, W)`` starting models: a smooth vertical velocity gradient (normalised),
    the standard cycle-skipping-averse FWI initial model. Each of the ``n`` restarts gets a
    small random offset so deterministic FWI still yields a (narrow) sample spread."""
    col = torch.linspace(top, bottom, h, device=device).reshape(h, 1).expand(h, w)
    base = col.unsqueeze(0).expand(n, h, w).clone()
    base += 0.02 * torch.randn(n, 1, 1, device=device)  # per-restart constant offset
    return base.clamp(-1.0, 1.0)


def regularized_fwi(
    forward_fn: ForwardFn,
    d_obs: Tensor,
    *,
    shape: tuple[int, int],
    n_samples: int,
    iters: int,
    lr: float,
    reg: str = "tikhonov",
    reg_weight: float = 0.0,
    tv_eps: float = 1e-3,
    device: torch.device,
) -> tuple[Tensor, int]:
    """Gradient-descent FWI from a smooth start; returns ``(v_mps (n, H, W), n_solves)``.

    Optimises normalised models with Adam against the relative data misfit
    ``||forward(v) - d_obs||^2 / ||d_obs||^2`` plus ``reg_weight * R(v)``, clamping to
    ``[-1, 1]`` each step. ``n_solves = iters * n_samples`` (one forward solve per model per
    iteration — the dominant cost, matching how the generative samplers are counted).
    """
    h, w = shape
    x = linear_gradient_init(n_samples, h, w, device).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    denom = (d_obs**2).sum().clamp_min(1e-12)

    for _ in range(iters):
        opt.zero_grad()
        v_mps = _to_mps(x)
        pred = torch.stack([forward_fn(v_mps[b]) for b in range(n_samples)])
        data = ((pred - d_obs) ** 2).sum() / denom
        loss = data + reg_weight * regularization(x, reg, tv_eps=tv_eps)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)

    return _to_mps(x.detach()), iters * n_samples
