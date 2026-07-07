import math

import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import GaussiansDatasetConfig
from physics_informed_flow_map.flow_matching.models import MLPModelConfig, build_model
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train


def _gaussian_loader(n_samples: int, batch_size: int) -> DataLoader:
    ds = GaussiansDatasetConfig(n_samples=n_samples).build()
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def test_train_runs_and_logs() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(6400, 128)  # 50 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=64, depth=3))

    logged: list[dict] = []
    history, _ = train(
        model,
        loader,
        n_epochs=3,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
    )

    # history stays per-step (150 = 3 epochs x 50 batches); logging is per-epoch.
    assert len(history) == 150
    assert all("total" in r and "epoch" in r for r in history)
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]
    # Each epoch logs the namespaced train metrics + timing (3 epochs -> 3 such logs).
    epoch_logs = [r for r in logged if "train/loss" in r]
    assert len(epoch_logs) == 3
    assert all(
        {"train/loss", "train/grad_norm", "train/lr", "time/epoch_s"} <= r.keys()
        for r in epoch_logs
    )
    # A single end-of-run wall-clock summary is logged last.
    assert sum("time/total_min" in r for r in logged) == 1


def test_train_history_is_per_step() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(96, 32)  # 3 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    history, _ = train(
        model,
        loader,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    # The returned history keeps a per-step record (used for the final-epoch mean loss).
    assert len(history) == 3
    assert {"total", "epoch", "step"} == set(history[0].keys())


def test_train_warmup_ramps_lr_below_base() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    logged: list[dict] = []
    train(
        model,
        _gaussian_loader(96, 32),  # 3 steps in 1 epoch
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        warmup_steps=2000,  # far longer than 3 steps -> lr still ramping (0.1x -> 1x)
        log=lambda **r: logged.append(r),
    )
    lr_logged = next(r["train/lr"] for r in logged if "train/lr" in r)
    # Still in warmup after 3 steps: above the 0.1x floor, below the 1x base lr.
    assert 1e-4 <= lr_logged < 1e-3


def test_train_no_warmup_keeps_base_lr() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    logged: list[dict] = []
    train(
        model,
        _gaussian_loader(96, 32),
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
    )
    lr_logged = next(r["train/lr"] for r in logged if "train/lr" in r)
    assert lr_logged == 1e-3


def test_train_hooks_fire_on_epoch_cadence() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    evals: list[int] = []
    ckpts: list[tuple[int, bool, bool]] = []

    def on_eval(m: object, epoch: int) -> float:
        evals.append(epoch)
        return 1.0 / len(evals)  # strictly decreasing -> every eval is a new best

    def on_checkpoint(
        m: object,
        epoch: int,
        *,
        is_best: bool,
        is_final: bool,
        ema_model: object = None,
        train_state: dict | None = None,
    ) -> None:
        ckpts.append((epoch, is_best, is_final))

    train(
        model,
        _gaussian_loader(64, 16),  # 4 batches/epoch
        n_epochs=4,
        lr=1e-3,
        device=torch.device("cpu"),
        eval_every_epochs=2,
        on_eval=on_eval,
        ckpt_every_epochs=0,
        on_checkpoint=on_checkpoint,
    )

    assert evals == [1, 3]  # (epoch+1) % 2 == 0
    assert [c[0] for c in ckpts if c[1]] == [1, 3]  # best at each eval
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 3  # exactly one final, last epoch


def test_train_checkpoint_cadence_without_eval() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    ckpts: list[tuple[int, bool, bool]] = []

    def on_checkpoint(
        m: object,
        epoch: int,
        *,
        is_best: bool,
        is_final: bool,
        ema_model: object = None,
        train_state: dict | None = None,
    ) -> None:
        ckpts.append((epoch, is_best, is_final))

    train(
        model,
        _gaussian_loader(64, 16),
        n_epochs=6,
        lr=1e-3,
        device=torch.device("cpu"),
        ckpt_every_epochs=3,
        on_checkpoint=on_checkpoint,
    )

    # No eval -> never a best; cadence fires at (epoch+1) % 3 == 0 -> epochs 2, 5.
    assert all(not is_best for _, is_best, _ in ckpts)
    assert [ep for ep, _, is_final in ckpts if not is_final] == [2, 5]
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 5  # exactly one final, last epoch


def test_train_ema_disabled_returns_none() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    history, ema_model = train(
        model,
        _gaussian_loader(96, 32),  # 3 batches -> 3 steps
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
    )
    assert ema_model is None
    assert len(history) == 3


def test_train_ema_enabled_returns_distinct_module() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    _, ema_model = train(
        model,
        _gaussian_loader(96, 32),  # 3 steps -> EMA lags the raw weights
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        ema_enabled=True,
        ema_decay=0.9,
    )
    assert ema_model is not None
    assert ema_model is not model
    raw = dict(model.named_parameters())
    differs = any(
        not torch.equal(p.detach(), raw[name].detach())
        for name, p in ema_model.named_parameters()
        if name in raw
    )
    assert differs


def test_train_ema_warmup_beyond_run_returns_none() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    _, ema_model = train(
        model,
        _gaussian_loader(96, 32),  # 3 steps total
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        ema_enabled=True,
        ema_warmup_steps=100,  # never reached -> EMA never updates
    )
    assert ema_model is None


def test_train_resume_continues_from_checkpoint() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(96, 32)  # 3 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    saved: dict = {}

    def on_checkpoint(
        m: object,
        epoch: int,
        *,
        is_best: bool,
        is_final: bool,
        ema_model: object = None,
        train_state: dict | None = None,
    ) -> None:
        if is_final and train_state is not None:
            saved.update(train_state)

    train(
        model,
        loader,
        n_epochs=2,
        lr=1e-3,
        device=torch.device("cpu"),
        warmup_steps=100,  # so scheduler state is exercised through the round-trip
        on_checkpoint=on_checkpoint,
    )
    assert saved["epoch"] == 1 and saved["step"] == 6
    assert saved["optimizer"] is not None and saved["scheduler"] is not None

    history, _ = train(
        model,
        loader,
        n_epochs=4,  # the TOTAL: resume runs epochs 2 and 3 only
        lr=1e-3,
        device=torch.device("cpu"),
        warmup_steps=100,
        resume=saved,
    )
    assert {h["epoch"] for h in history} == {2.0, 3.0}
    assert history[0]["step"] == 6  # step counter continues, not restarts


def test_train_resume_weights_only_estimates_step() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(96, 32)  # 3 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    # Pre-resume-support checkpoints carry only the epoch: the optimizer restarts fresh and
    # the step counter is estimated as (epoch + 1) * len(loader).
    history, _ = train(
        model,
        loader,
        n_epochs=3,
        lr=1e-3,
        device=torch.device("cpu"),
        resume={"epoch": 1},
    )
    assert {h["epoch"] for h in history} == {2.0}
    assert history[0]["step"] == 6


def test_train_resume_past_n_epochs_rejected() -> None:
    import pytest

    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(96, 32)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    with pytest.raises(ValueError, match="n_epochs"):
        train(
            model,
            loader,
            n_epochs=2,
            lr=1e-3,
            device=torch.device("cpu"),
            resume={"epoch": 1},
        )


def test_make_loss_fn_produces_finite_loss() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    loss_fn = make_loss_fn(None)
    x1, labels = next(iter(_gaussian_loader(32, 16)))
    opt_losses, _ = loss_fn(model, None, x1, labels, step=0)
    total = sum(opt_losses.values())
    assert math.isfinite(float(total.item()))


def test_train_logs_val_loss_when_on_eval_returns_metric() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))
    records: list[dict] = []
    train(
        model,
        _gaussian_loader(96, 32),  # 3 batches / epoch
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: records.append(r),
        eval_every_epochs=1,
        on_eval=lambda m, epoch: 0.5,
    )
    assert any(r.get("val/loss") == 0.5 for r in records)


def test_train_bf16_precision_runs() -> None:
    # bf16 autocast on CPU: loss forward under autocast, fp32 weights/grads; must converge
    # to a finite loss just like fp32.
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(512, 128)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=32, depth=2))
    history, _ = train(
        model,
        loader,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        precision="bf16",
    )
    assert history and all(math.isfinite(h["total"]) for h in history)
    assert next(model.parameters()).dtype == torch.float32


def test_train_rejects_unknown_precision() -> None:
    import pytest

    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(256, 128)
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=32, depth=2))
    with pytest.raises(ValueError, match="precision"):
        train(
            model,
            loader,
            n_epochs=1,
            lr=1e-3,
            device=torch.device("cpu"),
            precision="fp16",
        )
