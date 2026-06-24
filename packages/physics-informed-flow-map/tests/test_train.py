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
    model = build_model(spec, mlp_width=64, mlp_depth=3)

    logged: list[dict] = []
    history = train(
        model, loader, n_steps=50, lr=1e-3, device=torch.device("cpu"), log=lambda **r: logged.append(r)
    )

    assert len(history) == 50
    assert len(logged) == 50
    assert all("fm_loss" in r for r in history)
    # loss should be finite and trend down over 50 steps on this easy target
    assert torch.isfinite(torch.tensor(history[-1]["total"]))
    assert history[-1]["total"] < history[0]["total"]
