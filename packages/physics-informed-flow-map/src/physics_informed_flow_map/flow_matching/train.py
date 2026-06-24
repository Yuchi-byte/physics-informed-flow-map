"""Generic flow-matching training loop, wrapping mfm's FM loss (pure-FM config)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from mfm.SI import Linear
from mfm.losses.losses import get_consistency_loss_fn
from mfm.models.base_model import BaseModel


class _Cfg:
    """Attribute bag mirroring the Hydra DictConfig mfm's loss expects."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _fm_loss_cfg(label_dim: int) -> _Cfg:
    return _Cfg(
        SI=_Cfg(t_max=1.0),
        trainer=_Cfg(
            t_cond_warmup_steps=0,
            t_cond_0_rate=1.0,  # always condition on pure noise → standard FM
            t_cond_power=1.0,
            # mfm's off-diagonal consistency/distillation term is gated by
            # `step > num_warmup_steps` (independent of distill_fm). Park the
            # warmup beyond any run length so pure FM keeps only the diagonal term.
            num_warmup_steps=10**12,
            anneal_end_step=10**12,
            class_dropout_prob=0.0,
        ),
        model=_Cfg(
            label_dim=label_dim,
            learn_loss_weighting=False,
            model_guidance_class_ws=[],
            model_guidance_x_cond_ws=[],
            init="dmf",
        ),
        loss=_Cfg(
            data_fm=True,
            distill_fm=False,
            distillation_type="mf",
            model_guidance=False,
            model_guidance_base_prob=0.5,
            fm_loss_type="l2",
            distillation_loss_type="l2",
            distill_fm_loss_type="l2",
            distill_teacher_stop_grad=True,
            fm_adaptive_loss_p=None,
            fm_adaptive_loss_c=None,
            distill_adaptive_loss_p=None,
            distill_adaptive_loss_c=None,
        ),
    )


def train(
    model: BaseModel,
    loader: DataLoader,
    *,
    n_steps: int,
    lr: float,
    device: torch.device,
    log: Callable[..., None] | None = None,
) -> list[dict[str, float]]:
    label_dim = 0  # unconditional FM; class_labels are ignored by the velocity models
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    data_iter = iter(loader)
    history: list[dict[str, float]] = []
    for step in range(n_steps):
        try:
            x1, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1, labels = next(data_iter)
        x1 = x1.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        opt_losses, _ = loss_fn(model, None, x1, labels, step=step)
        total = sum(opt_losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        rec = {
            "step": float(step),
            "fm_loss": float(opt_losses["fm_loss"].item()),
            "total": float(total.item()),
        }
        history.append(rec)
        if log is not None:
            log(**rec)
    return history
