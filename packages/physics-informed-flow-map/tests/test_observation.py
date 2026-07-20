"""Observation model: the continuity guarantee (clean config == today's
`simulate(v).detach()` bit-for-bit) and the generation-operator mismatch knobs."""

from typing import Any

import torch

from physics_informed_flow_map.physics.forward import simulate
from physics_informed_flow_map.physics.observation import (
    Observation,
    ObservationConfig,
    observe,
)

# Small fast survey on a 16x16 map (the test_forward pattern).
_KW: dict[str, Any] = dict(
    dx=10.0, dt=1e-3, nt=160, n_sources=1, n_receivers=12, freq=40.0
)
_V = torch.full((16, 16), 1500.0)
_V[8:] = 2000.0


def _obs(cfg: ObservationConfig) -> Observation:
    return observe(_V, cfg, **_KW)


def test_clean_config_is_passthrough() -> None:
    cfg = ObservationConfig()
    assert cfg.is_clean
    obs = _obs(cfg)
    assert obs.sigma is None and obs.noise_floor is None
    assert torch.equal(obs.d_obs, simulate(_V, **_KW).detach())


def test_mismatch_reaches_generation() -> None:
    clean = _obs(ObservationConfig()).d_obs
    refined = _obs(ObservationConfig(grid_scale=2)).d_obs
    shifted = _obs(ObservationConfig(wavelet_freq_scale=0.95)).d_obs
    assert clean.shape == refined.shape == shifted.shape
    for variant in (refined, shifted):
        assert not torch.equal(variant, clean)
    # Refined grid: same physics, different discretization (large on this deliberately
    # under-resolved toy survey — see test_forward for the rationale).
    rel = float((refined - clean).norm() / clean.norm())
    assert 1e-6 < rel < 2.0
