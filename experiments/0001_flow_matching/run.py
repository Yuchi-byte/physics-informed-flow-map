"""Train flow matching on swappable datasets (2D Gaussians, MNIST) via Hydra.

    uv run python experiments/0001_flow_matching/run.py                          # gaussians
    uv run python experiments/0001_flow_matching/run.py experiment=mnist
    uv run python experiments/0001_flow_matching/run.py experiment=smoke
    uv run python experiments/0001_flow_matching/run.py experiment=mnist training.n_epochs=80 training.eval_every_epochs=5
    uv run python experiments/0001_flow_matching/run.py experiment=openfwi
Logs held-out FM loss as the run summary scalar.
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
    OpenFWIDatasetConfig,
)
from physics_informed_flow_map.flow_matching.family_eval import (
    N_ENERGY_SAMPLES,
    family_reference_grid,
    per_family_energy_distance,
    per_family_val_loss,
)
from physics_informed_flow_map.flow_matching.models import (
    DiTModelConfig,
    MLPModelConfig,
    ModelConfig,
    build_model,
    count_parameters,
)
from physics_informed_flow_map.flow_matching.sample import sample, sample_trajectory
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train

EXPERIMENT = "0001_flow_matching"


class EmaConfig(Config):
    enabled: bool = False
    decay: float = Field(0.999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = 1e-3
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
    sampler_steps: int = Field(100, gt=0)
    n_eval_viz: int = Field(64, gt=0)  # samples drawn for each per-epoch viz
    n_traj_viz: int = Field(4, gt=0)   # samples shown in the trajectory grid
    traj_every_epochs: int = Field(20, gt=0)  # save a trajectory viz every N eval epochs


class FlowMatchingConfig(Config):
    seed: int = 0
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
    # name= the run-dir basename (dataset + timestamp) so the wandb run maps 1:1 to its
    # /workspace/runs folder instead of a random codename.
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name=run_dir.name)

    dataset = cfg.dataset.build()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0,
    )
    if len(loader) == 0:
        raise SystemExit(
            f"empty loader: dataset ({len(dataset)} samples) is smaller than "  # type: ignore[arg-type]  # torch's Dataset base isn't typed Sized; ours are map-style
            f"batch_size={cfg.training.batch_size} with drop_last=True"
        )
    model = build_model(cfg.dataset.shape, cfg.dataset.num_classes, cfg.model).to(
        device
    )
    n_params, n_trainable = count_parameters(model)
    print(
        f"[{EXPERIMENT}] model params: {n_params:,} total ({n_trainable:,} trainable)"
    )
    run.update_config(
        **{"model/num_params": n_params, "model/num_trainable_params": n_trainable}
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
    val_loss_fn = make_loss_fn(cfg.dataset.num_classes)

    # OpenFWI: per-family val losses + a one-time real-map reference grid (the priors are
    # unconditional, so samples can't be stratified by family — the real grid is the visual
    # reference; per-family losses and final energy distances are the coverage metrics).
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

    # Fixed noise for reproducible per-epoch visualizations: same starting points every eval,
    # so the grids track model improvement on identical samples rather than fresh random draws.
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    eval_noise = torch.randn(cfg.sampling.n_eval_viz, *cfg.dataset.shape, device=device, generator=g)
    traj_noise = eval_noise[: cfg.sampling.n_traj_viz]

    @torch.no_grad()
    def _batch_val_loss(m: BaseModel, xb: torch.Tensor, lb: torch.Tensor) -> float:
        opt_losses, _ = val_loss_fn(m, None, xb.to(device), lb.to(device), step=0)
        return float(sum(opt_losses.values()).item())

    @torch.no_grad()
    def compute_val_loss(m: BaseModel) -> tuple[float, dict[str, float]]:
        """Global val loss + per-family val losses (empty dict for non-OpenFWI)."""
        m.eval()
        if val_by_family is not None:
            return per_family_val_loss(
                lambda xb, lb: _batch_val_loss(m, xb, lb),
                val_by_family,
                cfg.training.batch_size,
            )
        total, n = 0.0, 0
        for xb, lb in val_loader:
            total += _batch_val_loss(m, xb, lb)
            n += 1
        return total / max(n, 1), {}

    def on_eval(m: BaseModel, epoch: int) -> float | None:
        s = sample(
            m,
            cfg.sampling.n_eval_viz,
            cfg.dataset.shape,
            sampler_steps=cfg.sampling.sampler_steps,
            device=device,
            x_noise=eval_noise,
        )
        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(s, p)
        run.log_image("samples", p, caption=f"epoch {epoch + 1}")
        if (epoch + 1) % cfg.sampling.traj_every_epochs == 0:
            states, x1hats = sample_trajectory(
                m,
                traj_noise,
                sampler_steps=cfg.sampling.sampler_steps,
                device=device,
            )
            # Two rows per sample: the transported ODE state x_t, then the one-step clean
            # estimate x1hat at the same time. [n_frames, B, 2, ...] -> [n_frames, 2B, ...]
            frames = torch.stack([states, x1hats], dim=2).flatten(1, 2)
            pt = run.ckpt_dir.parent / f"trajectory_epoch{epoch}.png"
            cfg.dataset.visualize_trajectory(frames, pt)
            run.log_image(
                "trajectory",
                pt,
                caption=f"epoch {epoch + 1} ODE trajectory (row pairs: x_t, x1hat)",
            )
        val_loss, fam_losses = compute_val_loss(m)
        if fam_losses:
            run.log(epoch=epoch, **{f"val/loss/{f}": v for f, v in fam_losses.items()})
        return val_loss

    on_checkpoint = run.checkpoint_callback(
        artifact_name=f"{cfg.dataset.name}-model",
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
    # Mean FM loss over the final epoch's steps (one minibatch is too noisy).
    last_epoch = history[-1]["epoch"]
    last = [h["total"] for h in history if h["epoch"] == last_epoch]
    final_loss = sum(last) / len(last)
    eval_model = ema_model if ema_model is not None else model

    final_val_loss, final_fam_losses = compute_val_loss(eval_model)
    summary: dict[str, float] = {
        "val/loss": final_val_loss,
        "train/final_loss": final_loss,
    }
    summary.update({f"val/loss/{f}": v for f, v in final_fam_losses.items()})
    if val_by_family is not None:
        # One shared generated pool vs each family's held-out maps: does the unconditional
        # prior cover every family's region of velocity-map space?
        pool = sample(
            eval_model,
            N_ENERGY_SAMPLES,
            cfg.dataset.shape,
            sampler_steps=cfg.sampling.sampler_steps,
            device=device,
        )
        energies = per_family_energy_distance(pool, val_by_family)
        summary.update({f"val/energy/{f}": v for f, v in energies.items()})
    run.finish(**summary)


if __name__ == "__main__":
    main()
