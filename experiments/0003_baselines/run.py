"""Train an unconditional diffusion prior (the camp-A baseline) on swappable image datasets.

    uv run python experiments/0003_baselines/run.py experiment=smoke
    uv run python experiments/0003_baselines/run.py experiment=openfwi
    uv run python experiments/0003_baselines/run.py experiment=openfwi_full

Mirrors the 0001 flow-matching framework (same Hydra config layout, same harness, same
held-out datasets + val-loss eval), swapping the flow prior for a diffusers DDPM prior.
Trains the denoiser, samples it for monitoring, logs held-out val loss as the run summary,
and uploads checkpoints/artifacts.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field, model_validator
from torch import nn

from physics_informed_flow_map.baselines import (
    build_denoiser,
    ddpm_sample,
    train_diffusion_prior,
)
from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    MNISTDatasetConfig,
)

EXPERIMENT = "0003_baselines"


class EmaConfig(Config):
    enabled: bool = False
    decay: float = Field(0.999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)


class ModelConfig(Config):
    # build_denoiser seam; sample_size/channels derive from the dataset shape.
    kind: str = "unet"


class DiffusionConfig(Config):
    num_train_timesteps: int = Field(1000, gt=0)
    num_sample_steps: int = Field(200, gt=0)  # reverse steps for eval/final sampling


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(64, gt=0)
    lr: float = 1e-4
    warmup_steps: int = Field(0, ge=0)
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(
        0, ge=0
    )  # local save + wandb artifact upload cadence
    ema: EmaConfig = EmaConfig()


TrainingConfig.model_rebuild()


class SamplingConfig(Config):
    n_eval_samples: int = Field(2048, gt=0)
    n_eval_viz: int = Field(64, gt=0)


class DiffusionBaselineConfig(Config):
    seed: int = 0
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = MNISTDatasetConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
    sampling: SamplingConfig = SamplingConfig()

    @model_validator(mode="after")
    def _check_image_dataset(self) -> "DiffusionBaselineConfig":
        if len(self.dataset.shape) != 3:
            raise ValueError("diffusion baseline needs an image dataset (C, H, W)")
        c, h, w = self.dataset.shape
        if h != w:
            raise ValueError(f"denoiser expects square images, got {h}x{w}")
        return self


DiffusionBaselineConfig.model_rebuild()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = DiffusionBaselineConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, DiffusionBaselineConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump())

    channels, size, _ = cfg.dataset.shape
    shape = cfg.dataset.shape
    denoiser = build_denoiser(cfg.model.kind, sample_size=size, channels=channels).to(
        device
    )
    scheduler = DDPMScheduler(num_train_timesteps=cfg.diffusion.num_train_timesteps)
    num_timesteps = cfg.diffusion.num_train_timesteps

    loader = torch.utils.data.DataLoader(
        cfg.dataset.build(),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"empty loader: dataset smaller than batch_size={cfg.training.batch_size}"
        )
    val_loader = torch.utils.data.DataLoader(
        cfg.dataset.build_val(),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    @torch.no_grad()
    def compute_val_loss(model: nn.Module) -> float:
        """Mean predict-noise DDPM loss over the held-out split (the per-eval val metric)."""
        model.eval()
        total, n = 0.0, 0
        for xb, _ in val_loader:
            xb = xb.to(device)
            noise = torch.randn_like(xb)
            t = torch.randint(0, num_timesteps, (xb.shape[0],), device=device)
            x_t = scheduler.add_noise(xb, noise, t)
            pred = model(x_t, t).sample
            total += float(F.mse_loss(pred, noise).item())
            n += 1
        return total / max(n, 1)

    def on_eval(model: nn.Module, epoch: int) -> float | None:
        s = ddpm_sample(
            model,
            scheduler,
            shape,
            n_samples=cfg.sampling.n_eval_viz,
            num_steps=cfg.diffusion.num_sample_steps,
            device=device,
        )
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p, caption=f"epoch {epoch + 1}")
        return compute_val_loss(model)

    on_checkpoint = run.checkpoint_callback(
        artifact_name=f"{cfg.dataset.name}-diffusion",
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        dataset=cfg.dataset.name,
        config=cfg.dump(),
    )

    _, ema_model = train_diffusion_prior(
        denoiser,
        scheduler,
        loader,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=device,
        log=run.log,
        warmup_steps=cfg.training.warmup_steps,
        ema_enabled=cfg.training.ema.enabled,
        ema_decay=cfg.training.ema.decay,
        ema_warmup_steps=cfg.training.ema.warmup_steps,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
    )
    eval_model = ema_model if ema_model is not None else denoiser

    samples = ddpm_sample(
        eval_model,
        scheduler,
        shape,
        n_samples=cfg.sampling.n_eval_samples,
        num_steps=cfg.diffusion.num_sample_steps,
        device=device,
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    final_val_loss = compute_val_loss(eval_model)
    run.finish(**{"val/loss": final_val_loss})


if __name__ == "__main__":
    main()
