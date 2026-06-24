"""Shared experiment harness: typed configs + manifest-pinned runs.

Frameworks under ``experiments/`` import from here:

    from physics_informed_flow_map.experiment import Config, Run, start_run
"""

from physics_informed_flow_map.experiment.config import Config
from physics_informed_flow_map.experiment.run import Run, start_run

__all__ = ["Config", "Run", "start_run"]
