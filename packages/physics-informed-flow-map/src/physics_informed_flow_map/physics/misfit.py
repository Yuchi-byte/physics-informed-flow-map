"""Data-misfit functionals for guidance: pointwise L2 and the Peng et al. OT potential.

Every guidance path in this package steers with the gradient of a data misfit through the
wave operator. This module factors that misfit out into interchangeable callables
``pred (B, *traces, nt) -> (B,)`` so the misfit becomes an inference-time knob
(``method.misfit`` in ``0004_inversion``) rather than a hard-wired ``((pred - d_obs)**2)``.

``OTMisfit`` implements the OT-based data-consistency potential of Peng et al. 2026
("Robust Physics-Guided Diffusion for Full-Waveform Inversion", §5.1 Eqs. 18-22, §6.2),
*without* their preconditioned guidance (Algorithm 2's rho_i * D_i) — the potential alone,
so L2-vs-OT comparisons isolate the misfit as the single variable. Three ingredients, all
observation-fixed so no model-dependent nonlinearity enters the frozen parts:

1. bounded amplitude weighting ``w = 1 / (1 + k |d_obs| / max|d_obs|)`` applied to both
   observed and synthetic traces (Eq. 18-19) — early high-amplitude arrivals stop
   dominating the gradient;
2. per-trace shift-and-normalize to probability densities (Eq. 20) followed by the 1-D
   Wasserstein-2 distance — misfit becomes convex in time shifts (anti-cycle-skipping);
3. observation-only scale normalization (Eq. 22) — the potential is dimensionless and
   O(1), stabilizing guidance-strength tuning.

The W2 integral is evaluated in the composition form ``W2^2 = ∫ (τ - G(τ))^2 ρ_syn(τ) dτ``
with ``G = F_obs^{-1} ∘ F_syn``: the *fixed* observed quantile is interpolated at the
differentiable synthetic CDF values, so autograd never differentiates through an inverse
(the standard trace-wise OT-FWI implementation; equivalent to the paper's quantile form
Eq. 21 by the substitution ξ = F_syn(τ)).

Note the Bayesian caveat: swapping L2 for OT changes the *target* posterior to a
generalized/Gibbs posterior ``p(v|d) ∝ exp(-J(v)/T) p(v)`` — deliberate robustification,
not exact Bayes under a noise model.

Known limitation of the shift-to-positive construction (empirically confirmed in
``test_misfit``): for an isolated *zero-mean* wavelet the positive lobe's excess mass is
absorbed by the adjacent negative-lobe deficit, so transport stays local and the plain
(k=0) misfit saturates in the shift almost as fast as L2 — the textbook W2
convexity-in-shift holds for nonnegative signals, not linearly shifted oscillatory ones
(Métivier et al.'s critique of this normalization). In this construction the long-range
shift signal comes from the observation-frozen amplitude weighting (a shifted synthetic
arrival lands where ``w ≈ 1`` while the observed arrival is crushed at its own location),
which discriminates non-overlapping shifts where L2 is exactly flat but is not globally
monotone. Alternatives (envelope/squaring encodings, graph-space OT) are future knobs.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

# A misfit functional: predicted data (B, *trace_dims, nt) -> per-sample misfit (B,).
# Sum over the batch for a guidance loss; per-sample values feed the MFM steering reward.
MisfitFn = Callable[[Tensor], Tensor]

MISFITS = ("l2", "ot")


def make_misfit(name: str, d_obs: Tensor, *, ot_k: float = 100.0) -> MisfitFn:
    """Build the named misfit against a fixed observation (``l2`` | ``ot``)."""
    if name == "l2":
        return l2_misfit(d_obs)
    if name == "ot":
        return OTMisfit(d_obs, k=ot_k)
    raise ValueError(f"unknown misfit '{name}' ({' | '.join(MISFITS)})")


def l2_misfit(d_obs: Tensor) -> MisfitFn:
    """Pointwise squared-L2 misfit ``pred -> sum (pred - d_obs)^2`` per sample — the
    exact negative log-likelihood under additive iid Gaussian noise, and the historical
    hard-wired default of every guidance path here."""

    def fn(pred: Tensor) -> Tensor:
        return ((pred - d_obs) ** 2).flatten(1).sum(1)

    return fn


def _cumtrapz(y: Tensor, dx: float) -> Tensor:
    """Cumulative trapezoid along the last dim, anchored at 0 (shape-preserving)."""
    inc = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    zero = torch.zeros_like(y[..., :1])
    return torch.cat([zero, torch.cumsum(inc, dim=-1)], dim=-1)


class OTMisfit:
    """Peng et al.'s weighted + normalized trace-wise Wasserstein-2 potential ``J(v)``.

    Everything derivable from the observation (weights ``w``, shift ``c'``, observed CDFs,
    scale ``S_obs``) is precomputed once and frozen. ``__call__`` is differentiable through
    ``pred`` and costs a few batched cumsums — negligible next to a wave solve.

    Args:
        d_obs: observed data ``(*trace_dims, nt)``; every leading dim indexes a trace
            (source x receiver in the OpenFWI acquisition).
        k: bounded-weighting strength (Eq. 18); ``0`` disables amplitude balancing.
            The paper illustrates ``k = 100``.
        eps: the paper's ``eps' = 1e-9`` mass-normalization guard.
    """

    def __init__(self, d_obs: Tensor, *, k: float = 100.0, eps: float = 1e-9) -> None:
        if d_obs.ndim < 2:
            raise ValueError(f"d_obs needs (*trace_dims, nt), got {tuple(d_obs.shape)}")
        d_obs = d_obs.detach()
        self.trace_shape = d_obs.shape[:-1]
        nt = d_obs.shape[-1]
        self.nt = nt
        self.eps = eps
        # Normalized time grid tau in [0, 1]; physical units cancel in J (Eq. 22 divides
        # a sum of W2 values by an observation scale with the same units).
        self.dtau = 1.0 / (nt - 1)
        self.tau = torch.linspace(0.0, 1.0, nt, device=d_obs.device, dtype=d_obs.dtype)

        # (I) Bounded amplitude weighting, computed from the observation and frozen.
        amax = d_obs.abs().max().clamp_min(eps)
        self.weight = 1.0 / (1.0 + k * d_obs.abs() / amax)  # in [1/(1+k), 1]
        dw_obs = self.weight * d_obs

        # (II) Fixed global shift c' = 1.1 |min dw_obs| (§6.2): the observed densities are
        # strictly positive (floor 0.1 |min|), so the observed CDFs are strictly increasing
        # and the quantile interpolation below never divides by a flat segment.
        self.shift = 1.1 * dw_obs.min().abs().clamp_min(eps)

        flat = (dw_obs + self.shift).reshape(-1, nt)  # (T, nt) positive trace densities
        mass = torch.trapezoid(flat, dx=self.dtau, dim=-1) + eps
        rho_obs = flat / mass[:, None]
        cdf = _cumtrapz(rho_obs, self.dtau)
        self.cdf_obs = cdf / cdf[:, -1:].clamp_min(eps)  # (T, nt), F(0)=0, F(1)=1

        # (III) Observation-only scale S_obs = sum_traces (∫ Q_obs(ξ)^2 dξ)^(1/2); by the
        # substitution ξ = F_obs(τ) this is ∫ τ^2 ρ_obs(τ) dτ per trace.
        q2 = torch.trapezoid(self.tau**2 * rho_obs, dx=self.dtau, dim=-1)
        self.scale = q2.clamp_min(0).sqrt().sum().clamp_min(eps)

    def _density_cdf(self, traces: Tensor) -> tuple[Tensor, Tensor]:
        """Weighted/shifted traces ``(..., nt)`` -> unit-mass density and CDF at the nodes.

        Synthetic traces can dip below ``-c'`` early in sampling (the paper assumes c' is
        "sufficiently large"); clamping at 0 keeps the density valid with a zero subgradient
        on the clamped region."""
        pos = traces.clamp_min(0.0)
        mass = torch.trapezoid(pos, dx=self.dtau, dim=-1) + self.eps
        rho = pos / mass[..., None]
        cdf = _cumtrapz(rho, self.dtau)
        return rho, cdf / cdf[..., -1:].clamp_min(self.eps)

    def __call__(self, pred: Tensor) -> Tensor:
        """Normalized OT misfit ``J`` (Eq. 22) per sample: ``(B, *trace_dims, nt) -> (B,)``."""
        if pred.shape[1:] != (*self.trace_shape, self.nt):
            raise ValueError(
                f"pred shape {tuple(pred.shape)} does not match observation "
                f"(B, {', '.join(map(str, (*self.trace_shape, self.nt)))})"
            )
        bsz = pred.shape[0]
        n_traces = self.cdf_obs.shape[0]
        dw_syn = (self.weight * pred + self.shift).reshape(bsz, n_traces, self.nt)
        rho_syn, cdf_syn = self._density_cdf(dw_syn)

        # G = F_obs^{-1}(F_syn(tau)): piecewise-linear interpolation of the fixed observed
        # quantile at the (differentiable) synthetic CDF values. searchsorted batches over
        # the trace dim, so queries are regrouped trace-major.
        xi = cdf_syn.permute(1, 0, 2).reshape(n_traces, bsz * self.nt)
        idx = torch.searchsorted(self.cdf_obs, xi.detach().contiguous())
        idx = idx.clamp(1, self.nt - 1)
        f_lo = torch.gather(self.cdf_obs, 1, idx - 1)
        f_hi = torch.gather(self.cdf_obs, 1, idx)
        tau_lo = (idx - 1).to(xi.dtype) * self.dtau
        g = tau_lo + (xi - f_lo) / (f_hi - f_lo).clamp_min(self.eps) * self.dtau
        g = g.reshape(n_traces, bsz, self.nt).permute(1, 0, 2).clamp(0.0, 1.0)

        w2sq = torch.trapezoid(
            rho_syn * (self.tau - g) ** 2, dx=self.dtau, dim=-1
        )  # (B, T)
        w2 = w2sq.clamp_min(0).add(1e-12).sqrt()  # eps: dJ/dv finite at perfect fit
        return w2.sum(dim=-1) / self.scale
