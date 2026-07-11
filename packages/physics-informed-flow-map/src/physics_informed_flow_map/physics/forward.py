"""Differentiable acoustic forward operator for FWI, via Deepwave.

``simulate`` maps a velocity model (m/s) to the surface seismic data an OpenFWI-style
survey would record — a thin wrapper around ``deepwave.scalar``. It is differentiable,
so gradients of any data functional flow back to the velocity model; that vector-Jacobian
product is what inference-time tilting / posterior sampling will use.

Defaults match the OpenFWI Vel/Fault/Style acquisition: a 70x70 grid at 10 m spacing,
5 surface Ricker sources at 15 Hz, 70 surface receivers, 1001 samples at 1 ms.

Hardened-benchmark knobs (design spec 2026-07-11; all default to today's behavior):
``grid_scale`` / ``freq_scale`` implement *generation-operator mismatch* (kill the inverse
crime) — solve on a ``grid_scale``-times refined discretization (same physical survey,
different numerical dispersion) and/or perturb the wavelet centre frequency, then return
data on the standard ``(n_sources, n_receivers, nt)`` axes. The guidance operator keeps
the defaults — that difference is the point. Band-limiting ("missing lows") is *not* a
``simulate`` knob: it is applied to data by ``physics.filters.highpass``, identically on
the generation side (``observe``) and the guidance/eval side, so the band limit is part of
the operator F on both sides exactly by construction.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from deepwave import scalar
from deepwave.wavelets import ricker
from torch import Tensor


def simulate(
    velocity_mps: Tensor,
    *,
    dx: float = 10.0,
    dt: float = 1e-3,
    nt: int = 1001,
    n_sources: int = 5,
    n_receivers: int = 70,
    freq: float = 15.0,
    grid_scale: int = 1,
    freq_scale: float = 1.0,
) -> Tensor:
    """Forward-model surface seismic data from a velocity map.

    Args:
        velocity_mps: ``(ny, nx)`` P-wave velocity in m/s.
        dx: grid spacing in metres (isotropic).
        dt: time step in seconds.
        nt: number of time samples.
        n_sources: surface sources, evenly spaced across the top row (one shot each).
        n_receivers: surface receivers, one per column from the left edge.
        freq: Ricker wavelet centre frequency in Hz.
        grid_scale: if > 1, solve on a refined grid (``dx/s``, ``dt/s``, same physical
            extent and survey positions) and decimate the output back to ``nt`` samples —
            a deliberately mismatched generation operator. Courant number is unchanged.
        freq_scale: multiplier on the wavelet centre frequency (generation-side source
            mismatch).

    Returns:
        ``(n_sources, n_receivers, nt)`` receiver amplitudes. Differentiable w.r.t.
        ``velocity_mps``.
    """
    device, dtype = velocity_mps.device, velocity_mps.dtype
    ny, nx = velocity_mps.shape
    s = int(grid_scale)
    if s < 1:
        raise ValueError(f"grid_scale must be >= 1, got {grid_scale}")

    # Survey geometry on the *coarse* grid (identical physical positions at every scale).
    src_col = torch.linspace(0, nx - 1, n_sources, device=device).round().long()
    rec_col = torch.arange(n_receivers, device=device)

    if s > 1:
        # Refine: (n-1)*s + 1 nodes keeps the physical extent (n-1)*dx exact under
        # align_corners=True; dt/s keeps the Courant number (stability) unchanged.
        velocity_mps = F.interpolate(
            velocity_mps[None, None],
            size=((ny - 1) * s + 1, (nx - 1) * s + 1),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        dx, dt, nt = dx / s, dt / s, (nt - 1) * s + 1
        src_col, rec_col = src_col * s, rec_col * s

    source_locations = torch.stack(
        [torch.zeros_like(src_col), src_col], dim=-1
    ).unsqueeze(1)  # (n_sources, 1, 2)
    receiver_locations = (
        torch.stack([torch.zeros_like(rec_col), rec_col], dim=-1)
        .unsqueeze(0)
        .expand(n_sources, -1, -1)
        .contiguous()
    )  # (n_sources, n_receivers, 2)

    f0 = freq * freq_scale
    wavelet = ricker(f0, nt, dt, 1.5 / f0).to(device=device, dtype=dtype)
    source_amplitudes = wavelet.reshape(1, 1, nt).repeat(n_sources, 1, 1)

    out = scalar(
        velocity_mps,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=f0,
    )
    data = cast(Tensor, out[-1])  # (n_sources, n_receivers, nt) receiver amplitudes
    return data[..., ::s] if s > 1 else data
