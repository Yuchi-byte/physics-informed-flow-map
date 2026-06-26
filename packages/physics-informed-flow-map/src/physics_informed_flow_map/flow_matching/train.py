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
    loss_weighting: str = "l2",
    t_cond_0_rate: float = 1.0,
    t_cond_power: float = 1.0,
    t_cond_warmup_steps: int = 0,
) -> _Cfg:
    class_dropout_prob = 0.1 if label_dim > 0 else 0.0
    # mfm's adaptive weighting w = 1/(||err||^2 + c)^p (detached) normalizes each term's
    # magnitude — essential for balancing the hard off-diagonal consistency loss against the
    # easy diagonal FM (p=1.0 tames the distillation term harder than the FM's p=0.5). "l2"
    # keeps the pure-FM behaviour (0001).
    adaptive = loss_weighting == "adaptive"
    loss_type = "adaptive" if adaptive else "l2"
    return _Cfg(
        SI=_Cfg(t_max=1.0),
        trainer=_Cfg(
            # t_cond is the conditioning time: t_cond=0 conditions on pure noise (unconditional);
            # t_cond>0 conditions on a partially-noised reference, the posterior p(x1|x_t) substrate.
            # t_cond_0_rate=1.0 stays fully unconditional. Teacher-free (mf) gets its conditioning
            # signal only from high t_cond, so power=1 (uniform), not mfm's teacher-recipe power=2.
            t_cond_warmup_steps=t_cond_warmup_steps,
            t_cond_0_rate=t_cond_0_rate,
            t_cond_power=t_cond_power,
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
            fm_loss_type=loss_type,
            distillation_loss_type=loss_type,
            distill_fm_loss_type=loss_type,
            distill_teacher_stop_grad=True,
            fm_adaptive_loss_p=0.5 if adaptive else None,
            fm_adaptive_loss_c=0.01 if adaptive else None,
            distill_adaptive_loss_p=1.0 if adaptive else None,
            distill_adaptive_loss_c=0.01 if adaptive else None,
        ),
    )


def make_loss_fn(
    num_classes: int | None,
    *,
    num_warmup_steps: int = _DISABLED,
    anneal_end_step: int = _DISABLED,
    distillation_type: str = "mf",
    loss_weighting: str = "l2",
    t_cond_0_rate: float = 1.0,
    t_cond_power: float = 1.0,
    t_cond_warmup_steps: int = 0,
) -> Callable[..., Any]:
    """mfm's SI consistency loss. Defaults to pure FM (off-diagonal disabled); a finite
    ``num_warmup_steps`` enables the ``s<u`` flow-map term. ``loss_weighting="adaptive"`` uses
    mfm's per-sample adaptive reweighting. ``t_cond_0_rate<1`` trains the intermediate-state
    posterior (the time-conditional flow map). Used for train and (at ``step=0``, so always
    pure-FM) validation."""
    return cast(
        Callable[..., Any],
        get_consistency_loss_fn(
            _loss_cfg(
                num_classes or 0,
                num_warmup_steps=num_warmup_steps,
                anneal_end_step=anneal_end_step,
                distillation_type=distillation_type,
                loss_weighting=loss_weighting,
                t_cond_0_rate=t_cond_0_rate,
                t_cond_power=t_cond_power,
                t_cond_warmup_steps=t_cond_warmup_steps,
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
    loss_weighting: str = "l2",
    t_cond_0_rate: float = 1.0,
    t_cond_power: float = 1.0,
    t_cond_warmup_steps: int = 0,
    teacher_model: BaseModel | None = None,
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
    training a meta flow map. Passing ``distillation_type="esd_teacher"`` with a frozen
    ``teacher_model`` distills the off-diagonal target from that teacher (the diagonal stays on
    data); the teacher path is unconditional only. History records ``{step, epoch, total}`` per
    step; the rest of the lifecycle is the shared loop's.
    """
    loss_fn = make_loss_fn(
        num_classes,
        num_warmup_steps=flow_map_warmup_steps,
        anneal_end_step=flow_map_anneal_end,
        distillation_type=distillation_type,
        loss_weighting=loss_weighting,
        t_cond_0_rate=t_cond_0_rate,
        t_cond_power=t_cond_power,
        t_cond_warmup_steps=t_cond_warmup_steps,
    )

    def compute_loss(m: BaseModel, batch: Any, step: int) -> Tensor:
        x1, labels = batch
        x1 = x1.to(device)
        # Unconditional datasets pass labels=None so the teacher's posterior-velocity extraction
        # takes the label-free path (mfm's conditional path is ImageNet-specific).
        lbl = labels.to(device) if num_classes else None
        opt_losses, _ = loss_fn(
            m, None, x1, lbl, step=step, teacher_model=teacher_model
        )
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
