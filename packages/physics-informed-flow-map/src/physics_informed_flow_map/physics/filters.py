"""Zero-phase frequency filters for seismic traces (time along the last axis).

One shared implementation of the band-limiting used across the hardened benchmark
(design spec 2026-07-11 §2): the *generation* side high-passes the clean data inside
``observe`` and the *guidance/eval* side high-passes predicted data with the **same
function** — so the band limit is part of the forward operator F on both sides exactly by
construction, and the Bayesian target stays well defined ("hard calibration" in the
research proposal §1.1).

The response is flat above ``1.5 * min_freq_hz`` with a raised-cosine roll-off to zero at
``0.5 * min_freq_hz`` — the mirror image of the ``lowpass`` in
``FWI_problem_exploration/cycle_skipping_landscape.py``. Zero-phase (real, non-negative
transfer function), so arrival times are not shifted. The input is zero-padded to twice
its length before the FFT so the filtering is a linear (not circular) convolution —
energy near the record end cannot wrap into early times.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor


def highpass(x: Tensor, min_freq_hz: float, dt: float) -> Tensor:
    """Zero-phase high-pass along the last axis; identity for ``min_freq_hz <= 0``.

    Kills content below ``0.5 * min_freq_hz``, passes content above ``1.5 * min_freq_hz``
    untouched, raised-cosine in between (so "no lows below ~f" means the taper is centred
    on f). Differentiable; works on any leading batch shape.
    """
    if min_freq_hz <= 0.0:
        return x
    nt = x.shape[-1]
    padded = F.pad(x, (0, nt))  # linear, not circular, convolution
    f = torch.fft.rfftfreq(2 * nt, dt).to(device=x.device)
    # 0 below 0.5*f0, 1 above 1.5*f0, raised cosine between.
    ramp = torch.clamp((f - 0.5 * min_freq_hz) / min_freq_hz, 0.0, 1.0)
    mask = 0.5 - 0.5 * torch.cos(math.pi * ramp)
    spec = torch.fft.rfft(padded, dim=-1) * mask.to(dtype=x.dtype)
    return cast(Tensor, torch.fft.irfft(spec, n=2 * nt, dim=-1)[..., :nt])
