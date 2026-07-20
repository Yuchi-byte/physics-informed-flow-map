"""The observation model of the hardened inversion benchmark (design spec 2026-07-11).

``observe`` turns a true velocity map into the fixed observation an inversion is judged
against, under one config that names the two benchmark tracks from the research proposal
(§1/§1.1):

- **calibration track** (``noise_frac > 0``, everything else default): ``d_obs`` is the
  band-limited clean data plus a *frozen* white-Gaussian realization whose σ is known by
  construction — the setting where the Bayesian posterior exists exactly and calibration
  claims are falsifiable. σ is matched, not tuned.
- **robustness track** (``min_freq_hz``/``grid_scale``/``wavelet_freq_scale`` engaged):
  missing lows and a deliberately mismatched *generation* operator (the guidance operator
  never changes — that difference is what kills the inverse crime).

The noise realization is deterministic in ``(key, noise_seed)`` and never stored: the same
target under the same config sees the identical ``d_obs`` in every run, method, and step —
like a field recording — while the benchmark directory stays untouched (regeneration is
pure). Noise is added *after* band-limiting and stays broadband: the field situation
(signal has no lows, noise everywhere), and the matched likelihood stays plain white L2.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from torch import Tensor

from .forward import simulate


class ObservationConfig(BaseModel):
    """How ``d_obs`` is generated from a true velocity map."""

    min_freq_hz: float = Field(
        0.0, ge=0.0, description="high-pass ('missing lows'); 0=off"
    )
    noise_frac: float = Field(
        0.0, ge=0.0, description="sigma as a fraction of clean-data RMS"
    )
    noise_seed: int = Field(
        0, description="combined with the target key -> frozen realization"
    )
    grid_scale: int = Field(
        1, ge=1, description="generation-operator grid refinement (mismatch)"
    )
    wavelet_freq_scale: float = Field(
        1.0, gt=0.0, description="generation wavelet centre-freq multiplier"
    )

    @property
    def is_clean(self) -> bool:
        """True when every knob is at the legacy (inverse-crime) default."""
        return (
            self.min_freq_hz == 0.0
            and self.noise_frac == 0.0
            and self.grid_scale == 1
            and self.wavelet_freq_scale == 1.0
        )


@dataclass
class Observation:
    """A fixed observation: the data, and (when noise is on) the matched-σ bookkeeping."""

    d_obs: Tensor
    sigma: (
        float | None
    )  # observation-noise std; None on noiseless configs (no principled value)
    noise_floor: (
        float | None
    )  # E ||F(v_true) - d_obs||^2 = sigma^2 * numel (chi^2 mean)


def observe(
    v_true: Tensor, cfg: ObservationConfig, **sim_kwargs: object
) -> Observation:
    """Generate the fixed observation for one target.

    Args:
        v_true: ``(ny, nx)`` velocity in m/s.
        cfg: the observation model.
        **sim_kwargs: forwarded to :func:`simulate` (survey geometry overrides; tests use
            a tiny survey, production uses the OpenFWI defaults).
    """
    d_clean = simulate(
        v_true,
        grid_scale=cfg.grid_scale,
        freq_scale=cfg.wavelet_freq_scale,
        **sim_kwargs,  # type: ignore[arg-type]
    ).detach()
    return Observation(d_obs=d_clean, sigma=None, noise_floor=None)
