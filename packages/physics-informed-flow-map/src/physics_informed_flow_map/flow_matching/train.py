"""Flow-matching trainer: builds mfm's pure-FM per-step loss and runs the shared train loop."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mfm.SI import Linear
from mfm.losses.losses import get_consistency_loss_fn
from mfm.models.base_model import BaseModel

from physics_informed_flow_map.training.loop import train_loop


class _Cfg:
    """Attribute bag mirroring the Hydra DictConfig mfm's loss expects."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _fm_loss_cfg(label_dim: int) -> _Cfg:
    class_dropout_prob = 0.1 if label_dim > 0 else 0.0
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
            class_dropout_prob=class_dropout_prob,
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


def make_loss_fn(num_classes: int | None) -> Callable[..., Any]:
    """The pure-FM consistency loss used for both training and validation."""
    return cast(
        Callable[..., Any],
        get_consistency_loss_fn(_fm_loss_cfg(num_classes or 0), Linear(t_max=1.0)),
    )


def train(
    model: BaseModel,
    loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    num_classes: int | None = None,
    log: Callable[..., None] | None = None,
    warmup_steps: int = 0,
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[[BaseModel, int], float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> tuple[list[dict[str, float]], BaseModel | None]:
    """Train a flow-matching model; thin wrapper over the shared ``train_loop``.

    The FM-specific per-step loss is mfm's pure-FM consistency loss (the off-diagonal
    distill terms are identically zero in this config, so the returned ``total`` is the FM
    loss). History records ``{step, epoch, total}`` per step; the rest of the lifecycle
    (warmup, EMA, per-epoch logging, eval/ckpt cadence) is the shared loop's.
    """
    loss_fn = make_loss_fn(num_classes)

    def compute_loss(m: BaseModel, batch: Any, step: int) -> Tensor:
        x1, labels = batch
        x1 = x1.to(device)
        labels = labels.to(device)
        opt_losses, _ = loss_fn(m, None, x1, labels, step=step)
        return cast(Tensor, sum(opt_losses.values()))

    history, ema = train_loop(
        model,
        loader,
        compute_loss,
        n_epochs=n_epochs,
        lr=lr,
        device=device,
        log=log,
        warmup_steps=warmup_steps,
        ema_enabled=ema_enabled,
        ema_decay=ema_decay,
        ema_warmup_steps=ema_warmup_steps,
        eval_every_epochs=eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
    return history, cast("BaseModel | None", ema)
