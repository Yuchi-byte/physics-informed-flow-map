"""Generic supervised training lifecycle, shared by the flow-matching trainer and the
diffusion-prior baseline.

The only task-specific part — the per-step loss — is supplied as a ``compute_loss(model,
batch, step) -> Tensor`` callback (it owns moving the batch to ``device``). Everything else
(optimizer, mfm-style linear warmup, EMA, per-epoch ``train/`` logging, eval cadence with
``val/loss``, checkpoint cadence + final) lives here so both trainers stay in sync.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    compute_loss: Callable[..., torch.Tensor],
    *,
    n_epochs: int,
    lr: float,
    device: torch.device,
    log: Callable[..., None] | None = None,
    warmup_steps: int = 0,
    grad_clip: float = 1.0,
    ema_enabled: bool = False,
    ema_decay: float = 0.999,
    ema_warmup_steps: int = 0,
    eval_every_epochs: int = 0,
    on_eval: Callable[..., float | None] | None = None,
    ckpt_every_epochs: int = 0,
    on_checkpoint: Callable[..., None] | None = None,
) -> tuple[list[dict[str, float]], nn.Module | None]:
    """Run ``n_epochs`` of training, returning a per-step history and the EMA model (or None).

    Args:
        compute_loss: ``(model, batch, step) -> scalar loss``; moves ``batch`` to ``device``.
        log: receives per-epoch ``train/loss``, ``train/grad_norm``, ``train/lr``,
            ``time/epoch_s``, ``perf/it_per_s`` and, at eval epochs, ``val/loss`` +
            ``time/eval_s`` (each with ``step`` + ``epoch``). A final ``time/total_min``
            is logged once at the end. (GPU memory comes from wandb's system metrics.)
        warmup_steps: linear LR warmup ``0.1x -> 1x`` over this many optimizer steps, then
            constant (mirrors mfm). ``0`` disables it.
        on_eval: ``(eval_model, epoch) -> metric | None`` at the eval cadence; a new best
            metric flags an ``is_best`` checkpoint.
        on_checkpoint: ``(model, epoch, *, is_best, is_final, ema_model)`` at the ckpt cadence,
            on every new best, and once at the end (``is_final=True``).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Linear warmup 0.1x -> 1x over warmup_steps, then constant (LinearLR holds end_factor
    # past total_iters, so no follow-on ConstantLR is needed). Stepped per optimizer step.
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

    def ema_module() -> nn.Module | None:
        """The averaged model once at least one EMA update has happened, else None."""
        if ema is not None and int(ema.n_averaged.item()) > 0:
            return ema.module
        return None

    history: list[dict[str, float]] = []
    best_metric = math.inf
    step = 0
    train_start = time.perf_counter()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        epoch_steps = 0
        epoch_start = time.perf_counter()
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}", leave=False):
            optimizer.zero_grad()
            loss = compute_loss(model, batch, step)
            loss.backward()  # type: ignore[no-untyped-call]
            # clip_grad_norm_ returns the *pre-clip* total grad norm — log it.
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if ema is not None and step >= ema_warmup_steps:
                ema.update_parameters(model)

            loss_f = float(loss.item())
            history.append(
                {"step": float(step), "epoch": float(epoch), "total": loss_f}
            )
            epoch_loss += loss_f
            epoch_grad_norm += grad_norm
            epoch_steps += 1
            step += 1

        # Log once per epoch: mean loss / grad-norm, end-of-epoch lr, and wall-clock
        # timing (epoch seconds, optimizer steps/sec).
        epoch_time = time.perf_counter() - epoch_start
        if log is not None and epoch_steps:
            log(
                step=step,
                epoch=epoch,
                **{
                    "train/loss": epoch_loss / epoch_steps,
                    "train/grad_norm": epoch_grad_norm / epoch_steps,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "time/epoch_s": epoch_time,
                    "perf/it_per_s": epoch_steps / epoch_time if epoch_time else 0.0,
                },
            )

        is_best = False
        if (
            on_eval is not None
            and eval_every_epochs
            and (epoch + 1) % eval_every_epochs == 0
        ):
            eval_model = ema_module() or model
            eval_start = time.perf_counter()
            metric = on_eval(eval_model, epoch)
            model.train()
            if metric is not None:
                if log is not None:
                    log(
                        step=step,
                        epoch=epoch,
                        **{
                            "val/loss": metric,
                            "time/eval_s": time.perf_counter() - eval_start,
                        },
                    )
                if metric < best_metric:
                    best_metric = metric
                    is_best = True

        if on_checkpoint is not None and (
            is_best or (ckpt_every_epochs and (epoch + 1) % ckpt_every_epochs == 0)
        ):
            on_checkpoint(
                model, epoch, is_best=is_best, is_final=False, ema_model=ema_module()
            )

    if log is not None:
        log(
            step=step,
            epoch=n_epochs - 1,
            **{"time/total_min": (time.perf_counter() - train_start) / 60},
        )

    if on_checkpoint is not None:
        on_checkpoint(
            model, n_epochs - 1, is_best=False, is_final=True, ema_model=ema_module()
        )
    return history, ema_module()
