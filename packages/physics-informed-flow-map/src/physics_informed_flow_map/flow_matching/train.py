"""Generic flow-matching training loop, wrapping mfm's FM loss (pure-FM config)."""

from __future__ import annotations

import math
from typing import Any, Callable, cast

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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


def train(
    model: BaseModel,
    loader: DataLoader,
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    num_classes: int | None = None,
    log: Callable[..., None] | None = None,
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[[BaseModel, int], float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> tuple[list[dict[str, float]], BaseModel | None]:
    label_dim = num_classes or 0
    loss_fn = get_consistency_loss_fn(_fm_loss_cfg(label_dim), Linear(t_max=1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    model.train()

    ema: AveragedModel | None = None
    if ema_enabled:
        ema = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(ema_decay),  # type: ignore[no-untyped-call]
            use_buffers=True,
        )

    def ema_module() -> BaseModel | None:
        """The averaged model once at least one EMA update has happened, else None."""
        if ema is not None and int(ema.n_averaged.item()) > 0:
            return cast(BaseModel, ema.module)
        return None

    history: list[dict[str, float]] = []
    best_metric = math.inf
    step = 0
    for epoch in range(n_epochs):
        for x1, labels in tqdm(
            loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False
        ):
            x1 = x1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            opt_losses, _ = loss_fn(model, None, x1, labels, step=step)
            total = sum(opt_losses.values())
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema is not None and step >= ema_warmup_steps:
                ema.update_parameters(model)

            rec: dict[str, float] = {
                "step": float(step),
                "epoch": float(epoch),
                "total": float(total.item()),
            }
            for name, value in opt_losses.items():
                rec[name] = float(value.item())
            history.append(rec)
            if log is not None:
                log(**rec)
            step += 1

        is_best = False
        if (
            on_eval is not None
            and eval_every_epochs
            and (epoch + 1) % eval_every_epochs == 0
        ):
            eval_model = ema_module() or model
            metric = on_eval(eval_model, epoch)
            model.train()
            if metric is not None and metric < best_metric:
                best_metric = metric
                is_best = True

        if on_checkpoint is not None and (
            is_best or (ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0)
        ):
            on_checkpoint(
                model, epoch, is_best=is_best, is_final=False, ema_model=ema_module()
            )

    if on_checkpoint is not None:
        on_checkpoint(
            model, n_epochs - 1, is_best=False, is_final=True, ema_model=ema_module()
        )
    return history, ema_module()
