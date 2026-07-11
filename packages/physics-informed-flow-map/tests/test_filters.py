"""Zero-phase high-pass: band behavior, phase neutrality, and wrap-freedom (the padding
that makes the filtering a linear, not circular, convolution — record-end energy must not
leak into early times, since guidance gradients read the whole record)."""

import torch

from physics_informed_flow_map.physics.filters import highpass

DT = 1e-3


def _band_energy(x: torch.Tensor, lo: float, hi: float, dt: float) -> float:
    f = torch.fft.rfftfreq(x.shape[-1], dt)
    spec = torch.fft.rfft(x, dim=-1).abs() ** 2
    return float(spec[..., (f >= lo) & (f < hi)].sum())


def test_noop_below_zero() -> None:
    x = torch.randn(3, 128)
    assert highpass(x, 0.0, DT) is x


def test_kills_lows_preserves_highs() -> None:
    """Transfer function on smooth (edge-free) signals: a 10 Hz tone dies under a 40 Hz
    high-pass, a 100 Hz tone passes untouched. (Raw white noise would show ~-30 dB crop-edge
    leakage instead — a finite record cannot be exactly band-limited; the smooth-signal
    bound is the meaningful one, and seismic records are causal/decaying, not edge-hard.)"""
    import math

    t = torch.arange(2048) * DT
    w = torch.hann_window(2048)
    lo = torch.sin(2 * math.pi * 10 * t) * w
    hi = torch.sin(2 * math.pi * 100 * t) * w
    assert float(highpass(lo, 40.0, DT).pow(2).sum()) < 1e-9 * float(lo.pow(2).sum())
    ratio = float(highpass(hi, 40.0, DT).pow(2).sum() / hi.pow(2).sum())
    assert abs(ratio - 1.0) < 1e-6


def test_zero_phase() -> None:
    """A symmetric pulse keeps its peak position and stays symmetric about it."""
    t = torch.arange(512, dtype=torch.float64) * DT
    pulse = torch.exp(-(((t - 0.25) / 0.01) ** 2))  # centred at sample 250, edge-free
    y = highpass(pulse, 30.0, DT)
    assert int(y.abs().argmax()) == 250
    # Reflection about sample 250: i -> 500 - i, i.e. flip (i -> 511-i) then roll(-11);
    # compare on the interior (edges hold cropped filter tails).
    refl = y.flip(-1).roll(-11)
    err = float((y - refl)[64:448].abs().max() / y.abs().max())
    assert err < 1e-10


def test_no_circular_wraparound() -> None:
    """A pulse near the record end must not leak to the record start (linear conv via
    padding). The kernel's smooth 1/t² tail reaches ~1e-4 relative; a circular FFT filter
    would wrap the near-lag tail there at ~1e-2 — an order-of-magnitude discriminator."""
    x = torch.zeros(512)
    x[480] = 1.0
    y = highpass(x, 40.0, DT)
    head = float(y[:128].abs().max())
    peak = float(y.abs().max())
    assert head < 1e-4 * peak


def test_differentiable() -> None:
    x = torch.randn(2, 64, requires_grad=True)
    highpass(x, 40.0, DT).pow(2).sum().backward()  # type: ignore[no-untyped-call]
    assert x.grad is not None and torch.isfinite(x.grad).all()
