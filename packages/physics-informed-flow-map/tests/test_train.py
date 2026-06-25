import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import GaussiansDatasetConfig
from physics_informed_flow_map.flow_matching.models import MLPModelConfig, build_model
from physics_informed_flow_map.flow_matching.train import train


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
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
    )

    assert len(history) == len(logged) == 50
    assert all("fm_loss" in r for r in history)
    assert all("epoch" in r for r in history)
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]


def test_train_logs_decomposed_losses() -> None:
    torch.manual_seed(0)
    spec = GaussiansDatasetConfig()
    loader = _gaussian_loader(96, 32)  # 3 batches/epoch
    model = build_model(spec.shape, spec.num_classes, MLPModelConfig(width=16, depth=2))

    records: list[dict] = []
    train(
        model,
        loader,
        n_epochs=1,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: records.append(r),
    )
    assert len(records) == 3
    assert "total" in records[0]
    assert "fm_loss" in records[0]
    assert "epoch" in records[0]


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
