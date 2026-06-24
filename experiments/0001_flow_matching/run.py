"""Train flow matching on swappable datasets (2D Gaussians, MNIST).

    uv run python experiments/0001_flow_matching/run.py gaussians
    uv run python experiments/0001_flow_matching/run.py mnist
    uv run python experiments/0001_flow_matching/run.py smoke

Verdict: gaussians → energy distance < gate; mnist → final FM loss < gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from pydantic import Field

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import DATASETS
from physics_informed_flow_map.flow_matching.models import build_model
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)
from physics_informed_flow_map.flow_matching.train import train


class FlowMatchingConfig(Config):
    seed: int = 0
    dataset: str = "gaussians"
    n_steps: int = Field(2000, gt=0)
    batch_size: int = 256
    lr: float = 1e-3
    sampler_steps: int = Field(100, gt=0)
    n_eval_samples: int = Field(2048, gt=0)
    gate: float = 0.5
    mlp_width: int = 256
    mlp_depth: int = 4
    dit_hidden: int = 128
    dit_depth: int = 4


VARIANTS: dict[str, dict[str, object]] = {
    "gaussians": {"dataset": "gaussians", "n_steps": 2000, "gate": 0.5},
    "mnist": {
        "dataset": "mnist",
        "n_steps": 3000,
        "batch_size": 128,
        "sampler_steps": 50,
        "gate": 240.0,
    },
    "smoke": {"dataset": "gaussians", "n_steps": 20, "n_eval_samples": 256, "gate": 1e9},
}


def main() -> None:
    argv = sys.argv[1:]
    has_variant = bool(argv) and "=" not in argv[0]
    variant = argv[0] if has_variant else "gaussians"
    overrides = argv[1:] if has_variant else argv
    if variant not in VARIANTS:
        sys.exit(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    cfg = FlowMatchingConfig.resolve(VARIANTS[variant], overrides)
    assert isinstance(cfg, FlowMatchingConfig)

    spec = DATASETS[cfg.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run = start_run(Path(__file__).parent, cfg.dump())

    dataset = spec.make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    model = build_model(
        spec,
        mlp_width=cfg.mlp_width,
        mlp_depth=cfg.mlp_depth,
        dit_hidden=cfg.dit_hidden,
        dit_depth=cfg.dit_depth,
    ).to(device)

    history = train(model, loader, n_steps=cfg.n_steps, lr=cfg.lr, device=device, num_classes=spec.num_classes, log=run.log)
    final_loss = history[-1]["total"]

    samples = sample(model, cfg.n_eval_samples, spec.shape, sampler_steps=cfg.sampler_steps, device=device)
    spec.visualize(samples, run.dir / "samples.png")

    if cfg.dataset == "gaussians":
        ref = real_reference(dataset, cfg.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        verdict = "pass" if metric < cfg.gate else "fail"
        run.finish(verdict, energy_distance=metric, final_loss=final_loss, gate=cfg.gate)
    else:
        verdict = "pass" if final_loss < cfg.gate else "fail"
        run.finish(verdict, final_loss=final_loss, gate=cfg.gate)


if __name__ == "__main__":
    main()
