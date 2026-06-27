"""Inversion evaluation: a uniform module interface + an evaluator that scores how well a
module recovers held-out velocity maps.
"""

from .base import InversionModule, InversionResult
from .bridge import held_out_targets, mps_to_norm, seismic_forward, to_mps_native
from .evaluate import Evaluator, InversionStats, score_target, ssim
from .modules import DiffusionDPSModule, FlowTiltModule

__all__ = [
    "InversionModule",
    "InversionResult",
    "Evaluator",
    "InversionStats",
    "score_target",
    "ssim",
    "FlowTiltModule",
    "DiffusionDPSModule",
    "held_out_targets",
    "mps_to_norm",
    "seismic_forward",
    "to_mps_native",
]
