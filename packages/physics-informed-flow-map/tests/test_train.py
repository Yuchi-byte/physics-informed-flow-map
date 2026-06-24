import torch
from torch.utils.data import DataLoader

from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.train import train


def test_train_runs_and_logs() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    ds = spec.make_dataset()
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, mlp_width=64, mlp_depth=3)

    logged: list[dict] = []
    history = train(
        model,
        loader,
        n_steps=50,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: logged.append(r),
    )

    assert len(history) == 50
    assert len(logged) == 50
    assert all("fm_loss" in r for r in history)
    # loss should be finite and trend down over 50 steps on this easy target
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]


def test_train_logs_decomposed_losses() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.make_dataset(), batch_size=32, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, mlp_width=16, mlp_depth=2)

    records: list[dict] = []
    train(
        model,
        loader,
        n_steps=3,
        lr=1e-3,
        device=torch.device("cpu"),
        log=lambda **r: records.append(r),
    )
    assert len(records) == 3
    assert "total" in records[0]
    assert "fm_loss" in records[0]


def test_train_hooks_fire_on_cadence() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.make_dataset(), batch_size=16, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, mlp_width=16, mlp_depth=2)

    evals: list[int] = []
    ckpts: list[tuple[int, bool, bool]] = []

    def on_eval(m: object, step: int) -> float:
        evals.append(step)
        return 1.0 / len(evals)  # strictly decreasing -> every eval is a new best

    def on_checkpoint(m: object, step: int, *, is_best: bool, is_final: bool) -> None:
        ckpts.append((step, is_best, is_final))

    train(
        model,
        loader,
        n_steps=10,
        lr=1e-3,
        device=torch.device("cpu"),
        eval_every=5,
        on_eval=on_eval,
        ckpt_every=0,
        on_checkpoint=on_checkpoint,
    )

    assert evals == [4, 9]  # (step+1) % 5 == 0
    assert [c[0] for c in ckpts if c[1]] == [4, 9]  # best at each eval
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 9  # exactly one final, last step


def test_train_checkpoint_cadence_without_eval() -> None:
    torch.manual_seed(0)
    spec = DATASETS["gaussians"]
    loader = DataLoader(spec.make_dataset(), batch_size=16, shuffle=True)
    model = build_model(spec.shape, spec.num_classes, mlp_width=16, mlp_depth=2)

    ckpts: list[tuple[int, bool, bool]] = []

    def on_checkpoint(m: object, step: int, *, is_best: bool, is_final: bool) -> None:
        ckpts.append((step, is_best, is_final))

    train(
        model,
        loader,
        n_steps=10,
        lr=1e-3,
        device=torch.device("cpu"),
        ckpt_every=4,
        on_checkpoint=on_checkpoint,
    )

    # No eval -> never a best; cadence fires at (step+1) % 4 == 0 -> steps 3, 7.
    assert all(not is_best for _, is_best, _ in ckpts)
    assert [step for step, _, is_final in ckpts if not is_final] == [3, 7]
    finals = [c for c in ckpts if c[2]]
    assert len(finals) == 1 and finals[0][0] == 9  # exactly one final, last step
