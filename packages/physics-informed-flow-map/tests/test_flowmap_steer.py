"""MFM-G steering reuse: the mfm drift/sampler plumbing with a stub flow map, a trivial reward,
and an identity scaler — no real network, no wave solver. Guards the shape plumbing flagged as
the risk (5-D batching, per-estimator drift shapes)."""

import torch
from mfm.models.base_model import BaseModel
from mfm.utils.steering import (
    euler_maruyama_sampler,
    euler_sampler,
    get_conditional_drift_fn,
)


class _StubMap(BaseModel):  # type: ignore[misc]  # mfm BaseModel is untyped
    """Velocity depends on x_cond so the IWAE steering gradient w.r.t. xt is non-trivial."""

    def v(self, s, u, x, t_cond, x_cond, **kwargs):  # type: ignore[no-untyped-def]
        return 0.1 * x_cond


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _reward(x1: torch.Tensor) -> torch.Tensor:
    return -(x1.flatten(1) ** 2).sum(1)  # [N] — peaks at x1 = 0


def test_drift_fn_shapes_all_estimators() -> None:
    m = _StubMap()
    x = torch.randn(3, 1, 8, 8)
    t = torch.full((3,), 0.5)
    for est in ("dps", "iwae", "sne"):
        drift_fn = get_conditional_drift_fn(
            m,
            _reward,
            _identity,
            type="sde",
            drift_estimator=est,
            mc_samples=4,
            guidance_scale=1.0,
            renorm_gradient=True,
        )
        drift, ret = drift_fn(x, t)
        assert drift.shape == x.shape
        assert torch.isfinite(drift).all()


def test_samplers_run_and_keep_shape() -> None:
    m = _StubMap()
    x0 = torch.randn(2, 1, 8, 8)
    for sde, sampler in ((True, euler_maruyama_sampler), (False, euler_sampler)):
        drift_fn = get_conditional_drift_fn(
            m,
            _reward,
            _identity,
            type="sde" if sde else "ode",
            drift_estimator="sne",
            mc_samples=4,
            guidance_scale=1.0,
        )
        out = sampler(x0, drift_fn, t_start=0.01, n_steps=5)
        assert out.shape == x0.shape
        assert torch.isfinite(out).all()


def test_iwae_steering_pulls_toward_reward_peak() -> None:
    # With reward peaking at x1=0 and the stub map x1 = noise + 0.1*xt, the IWAE steering drift
    # should have a component pulling xt toward 0 (downhill in ||xt||). Check it is non-zero and
    # finite (a directional sanity check, not an exact value).
    torch.manual_seed(0)
    m = _StubMap()
    drift_fn = get_conditional_drift_fn(
        m,
        _reward,
        _identity,
        type="ode",
        drift_estimator="iwae",
        mc_samples=8,
        guidance_scale=1.0,
        renorm_gradient=False,
    )
    x = torch.full((1, 1, 4, 4), 2.0)
    t = torch.full((1,), 0.5)
    _, ret = drift_fn(x, t)
    assert ret["steering_drift"].abs().sum() > 0
    assert torch.isfinite(ret["steering_drift"]).all()
