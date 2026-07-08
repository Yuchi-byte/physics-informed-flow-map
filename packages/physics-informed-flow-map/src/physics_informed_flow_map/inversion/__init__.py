"""Inversion evaluation: a uniform module interface + an evaluator that scores how well a
module recovers held-out velocity maps.
"""

from .base import InversionModule, InversionResult
from .benchmark import InversionBenchmark
from .bridge import held_out_targets, mps_to_norm, seismic_forward, to_mps_native
from .evaluate import Evaluator, InversionStats, score_target, ssim
from .flowmap_steer import FlowMapSteerModule, make_misfit_reward
from .fmrg import FmrgEModule, fmrg_e_sample
from .modules import (
    ClassicalFWIModule,
    DiffusionDPSModule,
    FlowTiltModule,
    RealisticFWIModule,
    REDDiffEqModule,
)

__all__ = [
    "InversionBenchmark",
    "InversionModule",
    "InversionResult",
    "Evaluator",
    "InversionStats",
    "score_target",
    "ssim",
    "FlowTiltModule",
    "DiffusionDPSModule",
    "ClassicalFWIModule",
    "RealisticFWIModule",
    "REDDiffEqModule",
    "FlowMapSteerModule",
    "make_misfit_reward",
    "FmrgEModule",
    "fmrg_e_sample",
    "held_out_targets",
    "mps_to_norm",
    "seismic_forward",
    "to_mps_native",
]
