"""The inversion-module interface the evaluator scores.

An :class:`InversionModule` is a fully-configured inverter (prior + guidance scheme +
hyperparameters baked in). Given observed seismic data it returns posterior velocity
samples in physical units, so the evaluator can score flow-tilting and diffusion-DPS (and
anything else) through one uniform interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@dataclass
class InversionResult:
    """What an :class:`InversionModule` returns for one observation.

    Attributes:
        v_hat: ``(n_samples, H, W)`` posterior velocity samples in m/s. Same resolution as training dataloader's velocity map (i.e. that used for training the model).
    """

    v_hat_resolution: Tensor
    v_hat_native: Tensor


@runtime_checkable
class InversionModule(Protocol):
    """Anything that inverts seismic data to velocity-map samples.

    ``name`` labels the method/config in result tables; ``invert`` maps observed data to
    posterior samples in m/s (native resolution), owning its own prior and guidance.
    """

    name: str

    def invert(self, d_obs: Tensor) -> InversionResult: ...
