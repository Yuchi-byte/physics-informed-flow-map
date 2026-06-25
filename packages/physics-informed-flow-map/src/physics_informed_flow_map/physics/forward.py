"""Differentiable acoustic forward operator for FWI, via Deepwave.

``simulate`` maps a velocity model (m/s) to the surface seismic data an OpenFWI-style
survey would record — a thin wrapper around ``deepwave.scalar``. It is differentiable,
so gradients of any data functional flow back to the velocity model; that vector-Jacobian
product is what inference-time tilting / posterior sampling will use.

Defaults match the OpenFWI Vel/Fault/Style acquisition: a 70x70 grid at 10 m spacing,
5 surface Ricker sources at 15 Hz, 70 surface receivers, 1001 samples at 1 ms.
"""

from __future__ import annotations

from typing import cast

import torch
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

    Returns:
        ``(n_sources, n_receivers, nt)`` receiver amplitudes. Differentiable w.r.t.
        ``velocity_mps``.
    """
    device, dtype = velocity_mps.device, velocity_mps.dtype
    _, nx = velocity_mps.shape

    # Sources and receivers sit on the surface (row 0); columns indexed [depth, x].
    src_x = torch.linspace(0, nx - 1, n_sources, device=device).round().long()
    source_locations = torch.stack([torch.zeros_like(src_x), src_x], dim=-1).unsqueeze(
        1
    )  # (n_sources, 1, 2)

    rec_x = torch.arange(n_receivers, device=device)
    receiver_locations = (
        torch.stack([torch.zeros_like(rec_x), rec_x], dim=-1)
        .unsqueeze(0)
        .expand(n_sources, -1, -1)
        .contiguous()
    )  # (n_sources, n_receivers, 2)

    wavelet = ricker(freq, nt, dt, 1.5 / freq).to(device=device, dtype=dtype)
    source_amplitudes = wavelet.reshape(1, 1, nt).repeat(n_sources, 1, 1)

    out = scalar(
        velocity_mps,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_freq=freq,
    )
    return cast(Tensor, out[-1])  # receiver amplitudes
