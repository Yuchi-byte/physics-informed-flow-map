"""Flow-matching / flow-map trainer: builds mfm's SI consistency loss and runs the shared
train loop.

Pure flow matching (the ``0001`` default) keeps only the diagonal ``s==u`` term — mfm's
off-diagonal consistency/distillation loss is gated by ``step > num_warmup_steps``, so parking
the warmup beyond the run length disables it. Passing a finite ``flow_map_warmup_steps`` turns
on the off-diagonal ``s<u`` term (the ``mf`` mutual-flow objective, self-contained — its target
is the data velocity plus a JVP correction, no teacher), training a meta flow map from scratch.
"""

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


# Off-diagonal warmup parked beyond any run length => pure flow matching (diagonal only).
_DISABLED = 10**12


def _loss_cfg(
    label_dim: int,
    *,
    num_warmup_steps: int = _DISABLED,
    anneal_end_step: int = _DISABLED,
    distillation_type: str = "mf",
    distillation_loss_type: str = "l2",
) -> _Cfg:
    class_dropout_prob = 0.1 if label_dim > 0 else 0.0
    return _Cfg(
        SI=_Cfg(t_max=1.0),
        trainer=_Cfg(
            t_cond_warmup_steps=0,
            t_cond_0_rate=1.0,  # always condition on pure noise → unconditional prior
            t_cond_power=1.0,
            # The off-diagonal consistency term turns on at `step > num_warmup_steps`;
            # _DISABLED keeps pure FM, a finite value trains a flow map.
            num_warmup_steps=num_warmup_steps,
            anneal_end_step=anneal_end_step,
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
            distillation_type=distillation_type,
            model_guidance=False,
            model_guidance_base_prob=0.5,
            fm_loss_type="l2",
            distillation_loss_type=distillation_loss_type,
            distill_fm_loss_type="l2",
            distill_teacher_stop_grad=True,
            fm_adaptive_loss_p=None,
            fm_adaptive_loss_c=None,
            distill_adaptive_loss_p=None,
            distill_adaptive_loss_c=None,
        ),
    )


def make_loss_fn(
    num_classes: int | None,
    *,
    num_warmup_steps: int = _DISABLED,
    anneal_end_step: int = _DISABLED,
    distillation_type: str = "mf",
) -> Callable[..., Any]:
    """mfm's SI consistency loss. Defaults to pure FM (off-diagonal disabled); a finite
    ``num_warmup_steps`` enables the ``s<u`` flow-map term. Used for train and (at ``step=0``,
    so always pure-FM) validation."""
    return cast(
        Callable[..., Any],
        get_consistency_loss_fn(
            _loss_cfg(
                num_classes or 0,
                num_warmup_steps=num_warmup_steps,
                anneal_end_step=anneal_end_step,
                distillation_type=distillation_type,
            ),
            Linear(t_max=1.0),
        ),
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
    flow_map_warmup_steps: int = _DISABLED,
    flow_map_anneal_end: int = _DISABLED,
    distillation_type: str = "mf",
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[[BaseModel, int], float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> tuple[list[dict[str, float]], BaseModel | None]:
    """Train a flow-matching model; thin wrapper over the shared ``train_loop``.

    The per-step loss is mfm's SI consistency loss. By default it is pure flow matching (the
    off-diagonal terms are identically zero, so ``total`` is the FM loss). Passing a finite
    ``flow_map_warmup_steps`` enables the ``s<u`` mutual-flow term after that many steps,
    training a meta flow map. History records ``{step, epoch, total}`` per step; the rest of
    the lifecycle (warmup, EMA, per-epoch logging, eval/ckpt cadence) is the shared loop's.
    """
    loss_fn = make_loss_fn(
        num_classes,
        num_warmup_steps=flow_map_warmup_steps,
        anneal_end_step=flow_map_anneal_end,
        distillation_type=distillation_type,
    )

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
