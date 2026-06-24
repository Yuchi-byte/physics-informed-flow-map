"""Train flow matching on swappable datasets (2D Gaussians, MNIST) via Hydra.

    uv run python experiments/0001_flow_matching/run.py                       # gaussians
    uv run python experiments/0001_flow_matching/run.py experiment=mnist
    uv run python experiments/0001_flow_matching/run.py experiment=smoke
    uv run python experiments/0001_flow_matching/run.py experiment=mnist eval_every=500 ckpt_every=1000

Verdict: gaussians → energy distance < gate; mnist → final FM loss < gate.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from mfm.models.base_model import BaseModel
from omegaconf import DictConfig
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

EXPERIMENT = "0001_flow_matching"


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
    eval_every: int = Field(0, ge=0)
    ckpt_every: int = Field(0, ge=0)
    artifact_every: int = Field(0, ge=0)
    n_eval_viz: int = Field(64, gt=0)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = FlowMatchingConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, FlowMatchingConfig)

    spec = DATASETS[cfg.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())

    dataset = spec.make_dataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    model = build_model(
        spec.shape,
        spec.num_classes,
        mlp_width=cfg.mlp_width,
        mlp_depth=cfg.mlp_depth,
        dit_hidden=cfg.dit_hidden,
        dit_depth=cfg.dit_depth,
    ).to(device)

    def on_eval(m: BaseModel, step: int) -> float | None:
        samples = sample(
            m,
            cfg.n_eval_viz,
            spec.shape,
            sampler_steps=cfg.sampler_steps,
            device=device,
        )
        path = run.ckpt_dir.parent / f"samples_{step}.png"
        spec.visualize(samples, path)
        run.log_image("samples", path, step=step)
        if cfg.dataset == "gaussians":
            ref = real_reference(dataset, cfg.n_eval_viz, device)
            return energy_distance(samples, ref)
        return None

    def on_checkpoint(
        m: BaseModel, step: int, *, is_best: bool = False, is_final: bool = False
    ) -> None:
        path = run.save_checkpoint(m, step, dataset=cfg.dataset, config=cfg.dump())
        aliases: list[str] = []
        if is_final:
            aliases.append("final")
        if is_best:
            aliases.append("best")
        if cfg.artifact_every and (step + 1) % cfg.artifact_every == 0:
            aliases.append("periodic")
        if aliases:
            run.log_artifact(path, name=f"{cfg.dataset}-model", aliases=aliases)

    history = train(
        model,
        loader,
        n_steps=cfg.n_steps,
        lr=cfg.lr,
        device=device,
        num_classes=spec.num_classes,
        log=run.log,
        eval_every=cfg.eval_every,
        on_eval=on_eval,
        ckpt_every=cfg.ckpt_every,
        on_checkpoint=on_checkpoint,
    )
    final_loss = history[-1]["total"]

    samples = sample(
        model,
        cfg.n_eval_samples,
        spec.shape,
        sampler_steps=cfg.sampler_steps,
        device=device,
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    spec.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    if cfg.dataset == "gaussians":
        ref = real_reference(dataset, cfg.n_eval_samples, device)
        metric = energy_distance(samples, ref)
        verdict = "pass" if metric < cfg.gate else "fail"
        run.finish(
            verdict, energy_distance=metric, final_loss=final_loss, gate=cfg.gate
        )
    else:
        verdict = "pass" if final_loss < cfg.gate else "fail"
        run.finish(verdict, final_loss=final_loss, gate=cfg.gate)


if __name__ == "__main__":
    main()
