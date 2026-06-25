"""Diffusion-prior + DPS baseline for FWI posterior sampling (camp A comparison)."""

from physics_informed_flow_map.baselines.diffusion_prior import (
    build_denoiser,
    train_diffusion_prior,
)

__all__ = ["build_denoiser", "train_diffusion_prior"]
