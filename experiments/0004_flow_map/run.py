"""Train a meta flow map (mfm, from scratch) on swappable datasets via Hydra.

    uv run python experiments/0004_flow_map/run.py                       # gaussians
    uv run python experiments/0004_flow_map/run.py experiment=mnist
    uv run python experiments/0004_flow_map/run.py experiment=openfwi
    uv run python experiments/0004_flow_map/run.py experiment=smoke

Same harness as 0001 (same datasets/model/EMA/eval/checkpoint), but the loss adds mfm's
off-diagonal ``s<u`` mutual-flow consistency term (enabled after ``flow_map_warmup_steps``),
training a flow map that maps between arbitrary interpolant times — not just the
noise→data diagonal. Trained from scratch (the ``mf`` target is self-contained, no teacher).
Validate on gaussians → mnist → openfwi. Logs held-out FM loss as the run summary scalar.
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
from physics_informed_flow_map.flow_matching.sample import sample, sample_few_step
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train

EXPERIMENT = "0004_flow_map"


class EmaConfig(Config):
    enabled: bool = False
    decay: float = Field(0.999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = 1e-3
    warmup_steps: int = Field(0, ge=0)  # LR warmup
    # Off-diagonal mutual-flow term turns on at this global step (lets diagonal FM settle).
    flow_map_warmup_steps: int = Field(1000, ge=0)
    flow_map_anneal_end: int = Field(20000, gt=0)
    distillation_type: str = "mf"
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(0, ge=0)
    ema: EmaConfig = EmaConfig()


TrainingConfig.model_rebuild()


class SamplingConfig(Config):
    sampler_steps: int = Field(100, gt=0)  # ODE reference sampler
    few_steps: int = Field(
        4, gt=0
    )  # flow-map (consistency) sampler — the payoff metric
    n_eval_viz: int = Field(64, gt=0)  # samples drawn for each viz (per-epoch + final)


class FlowMapConfig(Config):
    seed: int = 0
    model: ModelConfig = MLPModelConfig()
    dataset: DatasetConfig = GaussiansDatasetConfig()
    training: TrainingConfig = TrainingConfig()
    sampling: SamplingConfig = SamplingConfig()

    @model_validator(mode="after")
    def _check_model_dataset_compat(self) -> "FlowMapConfig":
        ndim = len(self.dataset.shape)
        if isinstance(self.model, MLPModelConfig) and ndim != 1:
            raise ValueError("mlp model needs a vector dataset (e.g. gaussians)")
        if isinstance(self.model, DiTModelConfig) and ndim != 3:
            raise ValueError("dit model needs an image dataset (e.g. mnist)")
        return self


FlowMapConfig.model_rebuild()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = FlowMapConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, FlowMapConfig)

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

    val_loader = torch.utils.data.DataLoader(
        cfg.dataset.build_val(),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    val_loss_fn = make_loss_fn(cfg.dataset.num_classes)  # step=0 → pure-FM diagonal

    @torch.no_grad()
    def compute_val_loss(m: BaseModel) -> float:
        m.eval()
        total, n = 0.0, 0
        for xb, lb in val_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            opt_losses, _ = val_loss_fn(m, None, xb, lb, step=0)
            total += float(sum(opt_losses.values()).item())
            n += 1
        return total / max(n, 1)

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
        run.log_image("samples", p, caption=f"epoch {epoch + 1} ODE")
        # Few-step flow-map sampling — the capability this framework adds over 0001.
        sf = sample_few_step(
            m,
            cfg.sampling.n_eval_viz,
            cfg.dataset.shape,
            n_steps=cfg.sampling.few_steps,
            device=device,
        )
        pf = run.ckpt_dir.parent / f"samples_fewstep_epoch{epoch}.png"
        cfg.dataset.visualize(sf, pf)
        run.log_image(
            "samples_fewstep",
            pf,
            caption=f"epoch {epoch + 1} {cfg.sampling.few_steps}-step",
        )
        return compute_val_loss(m)

    on_checkpoint = run.checkpoint_callback(
        artifact_name=f"{cfg.dataset.name}-flowmap",
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        dataset=cfg.dataset.name,
        config=cfg.dump(),
    )

    history, ema_model = train(
        model,
        loader,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=device,
        num_classes=cfg.dataset.num_classes,
        log=run.log,
        warmup_steps=cfg.training.warmup_steps,
        flow_map_warmup_steps=cfg.training.flow_map_warmup_steps,
        flow_map_anneal_end=cfg.training.flow_map_anneal_end,
        distillation_type=cfg.training.distillation_type,
        ema_enabled=cfg.training.ema.enabled,
        ema_decay=cfg.training.ema.decay,
        ema_warmup_steps=cfg.training.ema.warmup_steps,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
    # Mean total loss over the final epoch's steps (one minibatch is too noisy).
    last_epoch = history[-1]["epoch"]
    last = [h["total"] for h in history if h["epoch"] == last_epoch]
    final_loss = sum(last) / len(last)
    eval_model = ema_model if ema_model is not None else model

    samples = sample(
        eval_model,
        cfg.sampling.n_eval_viz,
        cfg.dataset.shape,
        sampler_steps=cfg.sampling.sampler_steps,
        device=device,
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    few = sample_few_step(
        eval_model,
        cfg.sampling.n_eval_viz,
        cfg.dataset.shape,
        n_steps=cfg.sampling.few_steps,
        device=device,
    )
    few_png = run.ckpt_dir.parent / "samples_fewstep.png"
    cfg.dataset.visualize(few, few_png)
    run.log_image("samples_fewstep_final", few_png)

    final_val_loss = compute_val_loss(eval_model)
    run.finish(
        **{
            "val/loss": final_val_loss,
            "train/final_loss": final_loss,
        }
    )


if __name__ == "__main__":
    main()
