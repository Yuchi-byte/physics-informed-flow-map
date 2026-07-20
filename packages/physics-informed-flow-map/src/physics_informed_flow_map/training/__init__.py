"""Shared training lifecycle (warmup, EMA, per-epoch logging, eval/ckpt cadence)."""

from physics_informed_flow_map.training.loop import train_loop
from physics_informed_flow_map.training.validate import assess_overfit

__all__ = ["train_loop", "assess_overfit"]
