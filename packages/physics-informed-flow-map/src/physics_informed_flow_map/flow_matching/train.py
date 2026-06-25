"""Generic flow-matching training loop, wrapping mfm's FM loss (pure-FM config)."""

from __future__ import annotations

import math
from typing import Any, Callable, cast

import torch
from torch.optim.lr_scheduler import LinearLR
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
    loss_fn = make_loss_fn(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mirror mfm: linear warmup 0.1x -> 1x over warmup_steps, then constant (no decay).
    # A single LinearLR holds lr at end_factor=1x past total_iters, so it needs no
    # follow-on ConstantLR. Stepped once per optimizer step.
    scheduler: LinearLR | None = None
    if warmup_steps > 0:
        scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )

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
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_steps = 0
        for x1, labels in tqdm(
            loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False
        ):
            x1 = x1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            opt_losses, _ = loss_fn(model, None, x1, labels, step=step)
            total = sum(opt_losses.values())
            total.backward()
            # clip_grad_norm_ returns the *pre-clip* total grad norm — log it.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if ema is not None and step >= ema_warmup_steps:
                ema.update_parameters(model)

            total_f = float(total.item())
            # In pure-FM mode the distill terms are identically 0; `total` is the FM loss.
            rec: dict[str, float] = {
                "step": float(step),
                "epoch": float(epoch),
                "total": total_f,
                "fm_loss": float(opt_losses["fm_loss"].item()),
            }
            history.append(rec)
            epoch_loss += total_f
            epoch_grad_norm += grad_norm
            epoch_steps += 1
            step += 1

        # Log once per epoch (mean loss / grad-norm over the epoch, end-of-epoch lr).
        if log is not None and epoch_steps:
            log(
                step=step,
                epoch=epoch,
                **{
                    "train/loss": epoch_loss / epoch_steps,
                    "train/grad_norm": epoch_grad_norm / epoch_steps,
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
            )

        is_best = False
        if (
            on_eval is not None
            and eval_every_epochs
            and (epoch + 1) % eval_every_epochs == 0
        ):
            eval_model = ema_module() or model
            metric = on_eval(eval_model, epoch)
            model.train()
            if metric is not None:
                if log is not None:
                    log(step=step, epoch=epoch, **{"val/loss": metric})
                if metric < best_metric:
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
