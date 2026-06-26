"""Shared training lifecycle (warmup, EMA, per-epoch logging, eval/ckpt cadence)."""

from physics_informed_flow_map.training.loop import train_loop

__all__ = ["train_loop"]
