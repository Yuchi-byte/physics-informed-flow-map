"""Train an unconditional diffusion prior (the camp-A baseline) on swappable image datasets.

    uv run python experiments/0003_diffusion/run.py experiment=smoke
    uv run python experiments/0003_diffusion/run.py experiment=openfwi
    uv run python experiments/0003_diffusion/run.py experiment=openfwi_full

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
    ddpm_sample_trajectory,
    train_diffusion_prior,
)
from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    MNISTDatasetConfig,
    OpenFWIDatasetConfig,
)
from physics_informed_flow_map.flow_matching.family_eval import (
    N_ENERGY_SAMPLES,
    family_reference_grid,
    per_family_energy_distance,
    per_family_val_loss,
)
from physics_informed_flow_map.flow_matching.models import count_parameters

EXPERIMENT = "0003_diffusion"


class EmaConfig(Config):
    enabled: bool = False
    decay: float = Field(0.999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)


class ModelConfig(Config):
    # build_denoiser seam; sample_size/channels derive from the dataset shape.
    kind: str = "unet"  # unet | dit (the flow priors' DiT backbone as an eps-denoiser)
    hidden: int = (
        256  # DiT only; match the flow prior for an architecture-controlled compare
    )
    depth: int = 6
    num_heads: int = 8
    patch_size: int = 4


class DiffusionConfig(Config):
    num_train_timesteps: int = Field(1000, gt=0)
    num_sample_steps: int = Field(200, gt=0)  # reverse steps for eval/final sampling


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(64, gt=0)
    lr: float = 1e-4
    warmup_steps: int = Field(0, ge=0)
    num_workers: int = Field(4, ge=0)
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(
        0, ge=0
    )  # local save + wandb artifact upload cadence
    precision: str = "fp32"  # "bf16" runs the loss forward under autocast
    ema: EmaConfig = EmaConfig()


TrainingConfig.model_rebuild()


class SamplingConfig(Config):
    n_eval_viz: int = Field(64, gt=0)  # samples drawn for each viz (per-epoch + final)
    n_traj_viz: int = Field(4, gt=0)   # samples shown in the trajectory grid
    traj_every_epochs: int = Field(20, gt=0)  # save a trajectory viz every N eval epochs


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
    # name= denoiser kind + run-dir basename (dataset + timestamp): the kind is the key
    # 0003 variant and the basename maps the wandb run 1:1 to its /workspace/runs folder.
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name=f"{cfg.model.kind}_{run_dir.name}")

    channels, size, _ = cfg.dataset.shape
    shape = cfg.dataset.shape
    denoiser = build_denoiser(
        cfg.model.kind,
        sample_size=size,
        channels=channels,
        hidden=cfg.model.hidden,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        patch_size=cfg.model.patch_size,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
    ).to(device)
    n_params, n_trainable = count_parameters(denoiser)
    print(
        f"[{EXPERIMENT}] model params: {n_params:,} total ({n_trainable:,} trainable)"
    )
    run.update_config(
        **{"model/num_params": n_params, "model/num_trainable_params": n_trainable}
    )
    scheduler = DDPMScheduler(num_train_timesteps=cfg.diffusion.num_train_timesteps)  # type: ignore[no-untyped-call]  # diffusers __init__ is untyped
    num_timesteps = cfg.diffusion.num_train_timesteps

    loader = torch.utils.data.DataLoader(
        cfg.dataset.build(),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
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
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )

    # OpenFWI: per-family val losses + a one-time real-map reference grid (unconditional
    # prior → samples can't be stratified by family; see family_eval module docstring).
    val_by_family = (
        cfg.dataset.build_val_by_family()
        if isinstance(cfg.dataset, OpenFWIDatasetConfig)
        else None
    )
    if val_by_family is not None:
        ref_path = run.log_dir / "val_reference.png"
        family_reference_grid(val_by_family, ref_path)
        run.log_image(
            "val_reference", ref_path, caption="held-out real maps per family"
        )
        run.update_config(dataset_fingerprint=cfg.dataset.fingerprint())

    @torch.no_grad()
    def _batch_val_loss(model: nn.Module, xb: torch.Tensor) -> float:
        xb = xb.to(device)
        noise = torch.randn_like(xb)
        t = torch.randint(0, num_timesteps, (xb.shape[0],), device=device)
        x_t = scheduler.add_noise(xb, noise, t)  # type: ignore[attr-defined]  # diffusers stub omits add_noise
        pred = model(x_t, t).sample
        return float(F.mse_loss(pred, noise).item())

    @torch.no_grad()
    def compute_val_loss(model: nn.Module) -> tuple[float, dict[str, float]]:
        """Mean predict-noise DDPM val loss, global + per family (empty for non-OpenFWI)."""
        model.eval()
        if val_by_family is not None:
            return per_family_val_loss(
                lambda xb, lb: _batch_val_loss(model, xb),
                val_by_family,
                cfg.training.batch_size,
            )
        total, n = 0.0, 0
        for xb, _ in val_loader:
            total += _batch_val_loss(model, xb)
            n += 1
        return total / max(n, 1), {}

    def on_eval(model: nn.Module, epoch: int) -> float | None:
        # Fixed-seed generator so every epoch's grid samples the same noise (initial + ancestral):
        # the grids track one image's evolution as the model improves, not fresh random draws. A
        # local generator (not torch.manual_seed) leaves the training RNG stream untouched.
        s = ddpm_sample(
            model,
            scheduler,
            shape,
            n_samples=cfg.sampling.n_eval_viz,
            num_steps=cfg.diffusion.num_sample_steps,
            device=device,
            generator=torch.Generator(device=device).manual_seed(cfg.seed),
        )
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p, caption=f"epoch {epoch + 1}")
        if (epoch + 1) % cfg.sampling.traj_every_epochs == 0:
            states, x0hats = ddpm_sample_trajectory(
                model,
                scheduler,
                shape,
                n_samples=cfg.sampling.n_traj_viz,
                num_steps=cfg.diffusion.num_sample_steps,
                device=device,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
            )
            # Two rows per sample: the noisy running state x_t, then the Tweedie clean
            # estimate x0hat at the same step. [n_frames, B, 2, ...] -> [n_frames, 2B, ...]
            frames = torch.stack([states, x0hats], dim=2).flatten(1, 2)
            pt = run.ckpt_dir.parent / f"trajectory_epoch{epoch}.png"
            cfg.dataset.visualize_trajectory(frames, pt)
            run.log_image(
                "trajectory",
                pt,
                caption=f"epoch {epoch + 1} reverse-DDPM trajectory (row pairs: x_t, x0hat)",
            )
        val_loss, fam_losses = compute_val_loss(model)
        if fam_losses:
            run.log(epoch=epoch, **{f"val/loss/{f}": v for f, v in fam_losses.items()})
        return val_loss

    on_checkpoint = run.checkpoint_callback(
        artifact_name=f"{cfg.dataset.name}-diffusion",
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        dataset=cfg.dataset.name,
        config=cfg.dump(),
    )

    history, ema_model = train_diffusion_prior(
        denoiser,
        scheduler,
        loader,
        n_epochs=cfg.training.n_epochs,
        lr=cfg.training.lr,
        device=device,
        log=run.log,
        log_step=run.log_step,
        warmup_steps=cfg.training.warmup_steps,
        ema_enabled=cfg.training.ema.enabled,
        ema_decay=cfg.training.ema.decay,
        ema_warmup_steps=cfg.training.ema.warmup_steps,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
        precision=cfg.training.precision,
    )
    # Mean DDPM loss over the final epoch's steps (one minibatch is too noisy).
    last_epoch = history[-1]["epoch"]
    last = [h["total"] for h in history if h["epoch"] == last_epoch]
    final_loss = sum(last) / len(last)
    eval_model = ema_model if ema_model is not None else denoiser

    samples = ddpm_sample(
        eval_model,
        scheduler,
        shape,
        n_samples=cfg.sampling.n_eval_viz,
        num_steps=cfg.diffusion.num_sample_steps,
        device=device,
        generator=torch.Generator(device=device).manual_seed(cfg.seed),  # same fixed noise as the per-epoch grids
    )
    final_png = run.ckpt_dir.parent / "samples.png"
    cfg.dataset.visualize(samples, final_png)
    run.log_image("samples_final", final_png)

    final_val_loss, final_fam_losses = compute_val_loss(eval_model)
    summary: dict[str, float] = {
        "val/loss": final_val_loss,
        "train/final_loss": final_loss,
    }
    summary.update({f"val/loss/{f}": v for f, v in final_fam_losses.items()})
    if val_by_family is not None:
        # One shared generated pool vs each family's held-out maps: does the unconditional
        # prior cover every family's region of velocity-map space?
        pool = ddpm_sample(
            eval_model,
            scheduler,
            shape,
            n_samples=N_ENERGY_SAMPLES,
            num_steps=cfg.diffusion.num_sample_steps,
            device=device,
            generator=torch.Generator(device=device).manual_seed(cfg.seed),
        )
        energies = per_family_energy_distance(pool, val_by_family)
        summary.update({f"val/energy/{f}": v for f, v in energies.items()})
    run.finish(**summary)


if __name__ == "__main__":
    main()
