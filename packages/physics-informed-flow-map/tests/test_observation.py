"""Observation model: frozen-noise determinism, matched-σ bookkeeping, and the continuity
guarantee (clean config == today's `simulate(v).detach()` bit-for-bit)."""

import torch

from physics_informed_flow_map.physics.forward import simulate
from physics_informed_flow_map.physics.observation import (
    Observation,
    ObservationConfig,
    observe,
)

# Small fast survey on a 16x16 map (the test_forward pattern).
_KW = dict(dx=10.0, dt=1e-3, nt=160, n_sources=1, n_receivers=12, freq=40.0)
_V = torch.full((16, 16), 1500.0)
_V[8:] = 2000.0


def _obs(cfg: ObservationConfig, key: str = "t0") -> Observation:
    return observe(_V, cfg, key, **_KW)


def test_clean_config_is_passthrough() -> None:
    cfg = ObservationConfig()
    assert cfg.is_clean
    obs = _obs(cfg)
    assert obs.sigma is None and obs.noise_floor is None
    assert torch.equal(obs.d_obs, simulate(_V, **_KW).detach())


def test_noise_frozen_and_key_dependent() -> None:
    cfg = ObservationConfig(noise_frac=0.05)
    a1, a2 = _obs(cfg, "map_a"), _obs(cfg, "map_a")
    b = _obs(cfg, "map_b")
    reseeded = _obs(ObservationConfig(noise_frac=0.05, noise_seed=1), "map_a")
    assert torch.equal(a1.d_obs, a2.d_obs)  # frozen realization
    assert not torch.equal(a1.d_obs, b.d_obs)  # per-target
    assert not torch.equal(a1.d_obs, reseeded.d_obs)  # seed-sensitive


def test_sigma_matched_and_floor_consistent() -> None:
    obs = _obs(ObservationConfig(noise_frac=0.05))
    d_clean = simulate(_V, **_KW).detach()
    rms = float(d_clean.pow(2).mean().sqrt())
    assert obs.sigma is not None and obs.noise_floor is not None
    assert abs(obs.sigma - 0.05 * rms) < 1e-9
    # ||eta||^2 / floor ~ chi^2_N / N: mean 1, std sqrt(2/N); N = 1*12*160 -> std ~ 0.032.
    realized = float((obs.d_obs - d_clean).pow(2).sum()) / obs.noise_floor
    assert abs(realized - 1.0) < 0.15
    # sigma scales linearly with noise_frac.
    obs2 = _obs(ObservationConfig(noise_frac=0.10))
    assert obs2.sigma is not None
    assert abs(obs2.sigma - 2 * obs.sigma) < 1e-9


def test_bandlimit_and_mismatch_reach_generation() -> None:
    clean = _obs(ObservationConfig()).d_obs
    banded = _obs(ObservationConfig(min_freq_hz=20.0)).d_obs
    refined = _obs(ObservationConfig(grid_scale=2)).d_obs
    shifted = _obs(ObservationConfig(wavelet_freq_scale=0.95)).d_obs
    assert banded.shape == clean.shape == refined.shape == shifted.shape
    for variant in (banded, refined, shifted):
        assert not torch.equal(variant, clean)
    # Refined grid: same physics, different discretization (large on this deliberately
    # under-resolved toy survey — see test_forward for the rationale).
    rel = float((refined - clean).norm() / clean.norm())
    assert 1e-6 < rel < 2.0
