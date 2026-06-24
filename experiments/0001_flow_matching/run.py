"""Train flow matching on swappable datasets (2D Gaussians, MNIST) via Hydra.

    uv run python experiments/0001_flow_matching/run.py                          # gaussians
    uv run python experiments/0001_flow_matching/run.py experiment=mnist
    uv run python experiments/0001_flow_matching/run.py experiment=smoke
    uv run python experiments/0001_flow_matching/run.py experiment=mnist training.n_epochs=80 training.eval_every_epochs=5

Verdict: gaussians -> energy distance < gate; mnist -> final FM loss < gate.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from mfm.models.base_model import BaseModel
from omegaconf import DictConfig
from pydantic import Field, model_validator

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    GaussiansDatasetConfig,
)
from physics_informed_flow_map.flow_matching.models import (
    DiTModelConfig,
    MLPModelConfig,
    ModelConfig,
    build_model,
)
from physics_informed_flow_map.flow_matching.sample import (
    energy_distance,
    real_reference,
    sample,
)
from physics_informed_flow_map.flow_matching.train import train

EXPERIMENT = "0001_flow_matching"


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = 1e-3
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(0, ge=0)
    artifact_every_epochs: int = Field(0, ge=0)


class SamplingConfig(Config):
    sampler_steps: int = Field(100, gt=0)
    n_eval_samples: int = Field(2048, gt=0)
    n_eval_viz: int = Field(64, gt=0)


class FlowMatchingConfig(Config):
    seed: int = 0
    gate: float = 0.5
    model: ModelConfig = MLPModelConfig()
    dataset: DatasetConfig = GaussiansDatasetConfig()
    training: TrainingConfig = TrainingConfig()
    sampling: SamplingConfig = SamplingConfig()

    @model_validator(mode="after")
    def _check_model_dataset_compat(self) -> "FlowMatchingConfig":
        ndim = len(self.dataset.shape)
        if isinstance(self.model, MLPModelConfig) and ndim != 1:
            raise ValueError("mlp model needs a vector dataset (e.g. gaussians)")
        if isinstance(self.model, DiTModelConfig) and ndim != 3:
            raise ValueError("dit model needs an image dataset (e.g. mnist)")
        return self


FlowMatchingConfig.model_rebuild()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = FlowMatchingConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, FlowMatchingConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())

    dataset = cfg.dataset.build()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"empty loader: dataset ({len(dataset)} samples) is smaller than "
            f"batch_size={cfg.training.batch_size} with drop_last=True"
        )
    model = build_model(cfg.dataset.shape, cfg.dataset.num_classes, cfg.model).to(
        device
    )

    def on_eval(m: BaseModel, epoch: int) -> float | None:
        s = sample(
            m,
            cfg.sampling.n_eval_viz,
            cfg.dataset.shape,
            sampler_steps=cfg.sampling.sampler_steps,
            device=device,
        )
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p)
        if isinstance(cfg.dataset, GaussiansDatasetConfig):
            # 'best'-tracking metric: deliberately the cheaper n_eval_viz budget,
            # not the final verdict's n_eval_samples.
            ref = real_reference(dataset, cfg.sampling.n_eval_viz, device)
            return energy_distance(s, ref)
        return None

    def on_checkpoint(
        m: BaseModel, epoch: int, *, is_best: bool = False, is_final: bool = False
    ) -> None:
        path = run.save_checkpoint(
            m, epoch, dataset=cfg.dataset.name, config=cfg.dump()
        )
        aliases: list[str] = []
        if is_final:
            aliases.append("final")
        if is_best:
            aliases.append("best")
        if (
            cfg.training.artifact_every_epochs
            and (epoch + 1) % cfg.training.artifact_every_epochs == 0
        ):
            aliases.append("periodic")
        if aliases:
            run.log_artifact(path, name=f"{cfg.dataset.name}-model", aliases=aliases)

    history = train(
        model,
        loader,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=device,
        num_classes=cfg.dataset.num_classes,
        log=run.log,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
    final_loss = history[-1]["total"]

    samples = sample(
        model,
        cfg.sampling.n_eval_samples,
        cfg.dataset.shape,
        sampler_steps=cfg.sampling.sampler_steps,
        device=device,
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    if isinstance(cfg.dataset, GaussiansDatasetConfig):
        ref = real_reference(dataset, cfg.sampling.n_eval_samples, device)
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
