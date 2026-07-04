"""Guidance misfits: L2 parity, and the OT potential's defining properties.

Hermetic — synthetic Ricker-like traces, no wave solver. The load-bearing property is the
anti-cycle-skipping one: the OT misfit keeps growing with the time shift between two pulses
long after the pointwise L2 misfit has saturated (non-overlapping supports all look the same
to L2 — the flat/oscillatory landscape that causes cycle skipping).
"""

import torch

from physics_informed_flow_map.physics.misfit import OTMisfit, l2_misfit, make_misfit


def ricker(nt: int, t0: float, fp: float = 12.0) -> torch.Tensor:
    """Ricker wavelet centred at ``t0`` (both in units of the [0, 1] trace window)."""
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    arg = (torch.pi * fp * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * torch.exp(-arg)


def obs_from(trace: torch.Tensor) -> torch.Tensor:
    """Wrap a single trace as a (1, 1, nt) observation (one source-receiver pair)."""
    return trace.reshape(1, 1, -1)


def test_l2_matches_hardwired_expression() -> None:
    torch.manual_seed(0)
    d_obs = torch.randn(2, 3, 32)
    pred = torch.randn(4, 2, 3, 32)
    expected = ((pred - d_obs) ** 2).flatten(1).sum(1)
    assert torch.allclose(l2_misfit(d_obs)(pred), expected)


def test_make_misfit_dispatch() -> None:
    d_obs = obs_from(ricker(64, 0.5))
    assert isinstance(make_misfit("ot", d_obs), OTMisfit)
    try:
        make_misfit("huber", d_obs)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown misfit name must raise")


def test_ot_zero_at_observation() -> None:
    d_obs = obs_from(ricker(128, 0.4))
    j = OTMisfit(d_obs)(d_obs.unsqueeze(0))
    assert j.shape == (1,)
    assert float(j) < 1e-4  # J is O(1) by normalization; exact fit is numerically ~0


def test_ot_batch_shape() -> None:
    nt = 96
    d_obs = torch.stack([ricker(nt, 0.3), ricker(nt, 0.6)]).reshape(2, 1, nt)
    pred = torch.stack(
        [torch.stack([ricker(nt, c), ricker(nt, c + 0.2)]).reshape(2, 1, nt) for c in (0.3, 0.4, 0.5)]
    )
    j = OTMisfit(d_obs)(pred)
    assert j.shape == (3,)
    assert torch.isfinite(j).all()
    assert float(j[0]) < float(j[1]) < float(j[2])


def test_ot_linear_in_translation_of_positive_pulse() -> None:
    # The defining OT property, and a numerical check of the quantile-form W2 (Eq. 21):
    # for a translated *nonnegative* pulse the 1-D W2 is exactly the shift, so J must be
    # linear in the translation. (c' is ~0 for a positive signal, so the density map is
    # a pure normalization here.)
    nt = 800
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)

    def gauss(t0: float, s: float = 0.02) -> torch.Tensor:
        return torch.exp(-0.5 * ((t - t0) / s) ** 2).reshape(1, 1, nt)

    ot = OTMisfit(gauss(0.2), k=0.0)
    shifts = [0.05, 0.1, 0.2, 0.4]
    vals = ot(torch.stack([gauss(0.2 + s) for s in shifts]))
    for s, v in zip(shifts[1:], vals[1:]):
        assert abs(float(v / vals[0]) - s / shifts[0]) < 0.02


def test_ot_discriminates_where_l2_saturates() -> None:
    # A zero-mean Ricker at t0=0.15 observed, predictions shifted beyond overlap. There L2
    # is exactly flat — every wrong shift looks equally wrong (cycle skipping's landscape) —
    # while the weighted OT misfit still (a) separates near from far shifts and (b) retains
    # a usable spread across far shifts. Note it is NOT globally monotone in the shift: the
    # shift-to-positive normalization lets a zero-mean wavelet's positive-lobe excess be
    # absorbed by the adjacent negative-lobe deficit (transport stays local), so the
    # long-range signal comes from the observation-frozen amplitude weighting, not from
    # pulse-to-pulse transport. See the module docstring caveat.
    nt = 400
    d_obs = obs_from(ricker(nt, 0.15))
    ot = OTMisfit(d_obs)
    l2 = l2_misfit(d_obs)
    near, far = [0.02, 0.05], [0.2, 0.35, 0.5, 0.65]
    ot_near = ot(torch.stack([obs_from(ricker(nt, 0.15 + s)) for s in near]))
    far_preds = torch.stack([obs_from(ricker(nt, 0.15 + s)) for s in far])
    ot_far = ot(far_preds)

    assert float(ot_near.max()) < float(ot_far.min())

    l2_far = l2(far_preds)
    l2_rel_spread = float((l2_far.max() - l2_far.min()) / l2_far.max())
    assert l2_rel_spread < 0.01, f"L2 unexpectedly discriminates shifts: {l2_far}"
    ot_rel_spread = float((ot_far.max() - ot_far.min()) / ot_far.max())
    assert ot_rel_spread > 0.1


def test_ot_gradient_flows_and_matches_finite_difference() -> None:
    torch.manual_seed(1)
    nt = 64
    d_obs = obs_from(ricker(nt, 0.5))
    ot = OTMisfit(d_obs)
    pred = obs_from(ricker(nt, 0.62)).unsqueeze(0).clone().requires_grad_(True)

    j = ot(pred).sum()
    j.backward()
    grad = pred.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0.0

    # Directional finite-difference check (float64): the analytic directional derivative
    # must match (d/dh) J(pred + h v) despite the piecewise-linear quantile interpolation.
    v = torch.randn_like(pred.detach())
    h = 1e-6
    with torch.no_grad():
        j_plus = float(ot(pred.detach() + h * v).sum())
        j_minus = float(ot(pred.detach() - h * v).sum())
    fd = (j_plus - j_minus) / (2.0 * h)
    analytic = float((grad * v).sum())
    assert abs(fd - analytic) < 1e-4 * max(1.0, abs(fd))


def test_ot_amplitude_weighting_bounds() -> None:
    k = 100.0
    d_obs = obs_from(ricker(128, 0.5)) * 37.0  # arbitrary physical scale
    ot = OTMisfit(d_obs, k=k)
    assert float(ot.weight.min()) >= 1.0 / (1.0 + k) - 1e-9
    assert float(ot.weight.max()) <= 1.0 + 1e-9
    # The weight floor is attained exactly at the global amplitude maximum.
    amax_pos = d_obs.abs().argmax()
    assert abs(float(ot.weight.flatten()[amax_pos]) - 1.0 / (1.0 + k)) < 1e-9


def test_ot_shape_mismatch_raises() -> None:
    d_obs = obs_from(ricker(64, 0.5))
    ot = OTMisfit(d_obs)
    try:
        ot(torch.zeros(2, 1, 1, 32, dtype=torch.float64))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched trace shape must raise")
