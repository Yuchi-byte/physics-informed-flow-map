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

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..flow_matching.openfwi import VMAX, VMIN

ForwardFn = Callable[[Tensor], Tensor]  # (H, W) m/s -> data

REGULARIZERS = ("tikhonov", "tv")
INITS = ("smooth", "random")


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


def random_smooth_init(
    n: int, h: int, w: int, device: torch.device, *, coarse: int = 8
) -> Tensor:
    """``(n, H, W)`` structure-free starting models in ``[-1, 1]``: a random constant background
    spanning the velocity range plus a smooth low-frequency perturbation (upsampled coarse noise).
    Deliberately biased toward no geology — the naive FWI start that exposes non-uniqueness. Draws
    from the global RNG so a caller's ``manual_seed`` makes the restarts reproducible."""
    level = torch.empty(n, 1, 1, device=device).uniform_(-0.6, 0.6)
    noise = torch.randn(n, 1, coarse, coarse, device=device)
    smooth = F.interpolate(noise, size=(h, w), mode="bicubic", align_corners=False)[:, 0]
    return (level + 0.4 * smooth).clamp(-1.0, 1.0)


def _make_init(kind: str, n: int, h: int, w: int, device: torch.device) -> Tensor:
    if kind == "smooth":
        return linear_gradient_init(n, h, w, device)
    if kind == "random":
        return random_smooth_init(n, h, w, device)
    raise ValueError(f"unknown init {kind!r} (one of {INITS})")


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
    init: str = "smooth",
    device: torch.device,
) -> tuple[Tensor, int]:
    """Gradient-descent FWI; returns ``(v_mps (n, H, W), n_solves)``.

    Optimises normalised models with Adam against the relative data misfit
    ``||forward(v) - d_obs||^2 / ||d_obs||^2`` plus ``reg_weight * R(v)``, clamping to
    ``[-1, 1]`` each step. ``init`` picks the starting model: ``smooth`` (a 1-D gradient, the
    cycle-skipping-averse classical start) or ``random`` (structure-free, the naive baseline that
    exposes non-uniqueness). ``n_solves = iters * n_samples`` (one forward solve per model per
    iteration — the dominant cost, matching how the generative samplers are counted).
    """
    h, w = shape
    x = _make_init(init, n_samples, h, w, device).requires_grad_(True)
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


def lowpass_time(x: Tensor, fmax_hz: float, *, dt: float = 1e-3, taper: float = 1.3) -> Tensor:
    """Differentiable low-pass along the last (time) axis via rFFT, with a raised-cosine taper
    from ``fmax_hz`` to ``taper * fmax_hz`` (no ringing). Used for multiscale FWI: filtering both
    the observed and modelled data to the same band lets the low frequencies (which are far less
    prone to cycle-skipping) drive the early inversion."""
    nt = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(nt, d=dt, device=x.device)
    f1, f2 = fmax_hz, taper * fmax_hz
    ramp = ((f - f1) / (f2 - f1)).clamp(0.0, 1.0)
    win = 0.5 * (1.0 + torch.cos(math.pi * ramp))  # 1 below f1, cos taper to 0 at f2
    return torch.fft.irfft(spec * win, n=nt, dim=-1)


def multiscale_fwi(
    forward_fn: ForwardFn,
    d_obs: Tensor,
    *,
    shape: tuple[int, int],
    n_samples: int,
    freqs_hz: list[float],
    iters_per_stage: int,
    lr: float,
    reg: str = "tikhonov",
    reg_weight: float = 1e-4,
    tv_eps: float = 1e-3,
    dt: float = 1e-3,
    optimizer: str = "lbfgs",
    device: torch.device,
) -> tuple[Tensor, int]:
    """Realistic FWI: smooth 1-D start + multiscale frequency continuation + regularisation,
    optimised with L-BFGS (default) or Adam. Returns ``(v_mps (n, H, W), n_solves)``.

    For each cutoff in ``freqs_hz`` (ascending), low-pass both observed and modelled data to that
    band and minimise ``||low(F(v)) - low(d_obs)||^2 / ||low(d_obs)||^2 + reg_weight * R(v)``,
    warm-starting the next (higher) band from the current model. This is the standard cure for
    cycle-skipping. ``n_solves`` is the *actual* total forward solves — including every L-BFGS
    line-search evaluation — so it stays a faithful cost metric.
    """
    h, w = shape
    x = _make_init("smooth", n_samples, h, w, device).requires_grad_(True)
    n_solves = 0

    def loss_at(fmax: float, d_filt: Tensor, denom: Tensor) -> Tensor:
        nonlocal n_solves
        v_mps = _to_mps(x)
        pred = torch.stack([forward_fn(v_mps[b]) for b in range(n_samples)])
        n_solves += n_samples
        data = ((lowpass_time(pred, fmax, dt=dt) - d_filt) ** 2).sum() / denom
        return data + reg_weight * regularization(x, reg, tv_eps=tv_eps)

    for fmax in freqs_hz:
        d_filt = lowpass_time(d_obs, fmax, dt=dt)
        denom = (d_filt**2).sum().clamp_min(1e-12)
        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [x], lr=lr, max_iter=iters_per_stage, line_search_fn="strong_wolfe"
            )

            def closure() -> Tensor:
                with torch.no_grad():
                    x.clamp_(-1.0, 1.0)  # keep velocity in [VMIN, VMAX] at every trial point
                opt.zero_grad()
                loss = loss_at(fmax, d_filt, denom)
                loss.backward()  # type: ignore[no-untyped-call]
                return loss

            opt.step(closure)  # runs up to iters_per_stage inner iterations
        else:  # adam
            opt = torch.optim.Adam([x], lr=lr)
            for _ in range(iters_per_stage):
                opt.zero_grad()
                loss = loss_at(fmax, d_filt, denom)
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
                with torch.no_grad():
                    x.clamp_(-1.0, 1.0)
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)

    return _to_mps(x.detach()), n_solves
