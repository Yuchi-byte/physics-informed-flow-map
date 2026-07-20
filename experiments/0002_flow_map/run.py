"""Same harness as 0001 (same datasets/model/EMA/eval/checkpoint), but the loss adds mfm's
off-diagonal ``s<u`` consistency term (enabled after ``flow_map_warmup_steps``), training a
flow map that maps between arbitrary interpolant times — not just the noise→data diagonal.
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
    activate_x_cond_conditioning,
    build_model,
    count_parameters,
)
from physics_informed_flow_map.flow_matching.sample import (
    flow_map_consistency,
    sample,
    sample_few_step,
    sample_posterior,
    sample_trajectory,
)
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train

from physics_informed_flow_map.training.validate import assess_overfit

EXPERIMENT = "0002_flow_map"


class EmaConfig(Config):
    enabled: bool = False
    decay: float = Field(0.999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)


class TrainingConfig(Config):
    n_epochs: int = Field(10, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = 1e-3
    warmup_steps: int = Field(0, ge=0)  # LR warmup
    num_workers: int = Field(4, ge=0)
    flow_map_warmup_steps: int = Field(
        1000, ge=0
    )  # off-diagonal term on after this step
    flow_map_anneal_end: int = Field(20000, ge=0)
    distillation_type: str = "mf"
    loss_weighting: str = "adaptive"
    uncond_prob: float = Field(0.1, ge=0.0, le=1.0)  # mfm cfg key: t_cond_0_rate
    t_cond_power: float = Field(
        1.0, gt=0.0
    )  # nonzero t_cond ~ U(0,1)**power; 1=uniform, >1 near 0
    t_cond_warmup_steps: int = Field(
        0, ge=0
    )  # force t_cond=0 (unconditional) for this many steps
    # Optional frozen 0001 prior to distil from (local ckpt path or wandb artifact ref).
    # Set => esd_teacher distillation + warm-start; unset => from-scratch mf.
    teacher_ckpt: str | None = None
    eval_every_epochs: int = Field(0, ge=0)
    ckpt_every_epochs: int = Field(0, ge=0)
    precision: str = "fp32"  # "bf16" runs the loss forward under autocast
    resume_from: str | None = None
    ema: EmaConfig = EmaConfig()


TrainingConfig.model_rebuild()


class SamplingConfig(Config):
    sampler_steps: int = Field(100, gt=0)  # ODE reference sampler
    few_steps: int = Field(4, gt=0)  # flow-map consistency sampler
    # Posterior-recon panel: a few references, each reconstructed across this sweep of
    # conditioning levels (panel columns, after the reference itself).
    posterior_refs: int = Field(6, gt=0)
    posterior_t_grid: list[float] = Field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 0.9]
    )
    n_eval_viz: int = Field(64, gt=0)
    n_traj_viz: int = Field(4, gt=0)  # samples shown in the trajectory grid
    traj_every_epochs: int = Field(
        20, gt=0
    )  # save a trajectory viz every N eval epochs


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
        if self.training.teacher_ckpt and self.dataset.num_classes:
            raise ValueError(
                "teacher (esd_teacher) distillation supports only unconditional datasets "
                "(num_classes=0); mfm's posterior-velocity extraction takes an "
                "ImageNet-specific path when class labels are present."
            )
        if self.training.distillation_type == "esd_teacher" and not (
            self.training.teacher_ckpt
        ):
            raise ValueError(
                "distillation_type=esd_teacher requires training.teacher_ckpt "
                "(the frozen teacher to distil from)."
            )
        return self


FlowMapConfig.model_rebuild()


def _resolve_ckpt(ref: str) -> Path:
    """Resolve a teacher reference to a local checkpoint file: a path as-is, else a wandb
    artifact ref (``[entity/project/]name:alias``) downloaded via the wandb API."""
    p = Path(ref)
    if p.is_file():
        return p
    import wandb

    art = wandb.Api().artifact(ref, type="model")
    files = list(Path(art.download()).glob("*.pt"))
    if not files:
        raise SystemExit(f"no .pt checkpoint inside wandb artifact {ref!r}")
    return files[0]


def load_teacher(
    ref: str,
    shape: tuple[int, ...],
    num_classes: int | None,
    model_cfg: ModelConfig,
    device: torch.device,
) -> BaseModel:
    """Build a flow-map-shaped model, load a 0001 checkpoint's weights into it, and freeze it."""
    state = torch.load(_resolve_ckpt(ref), map_location=device, weights_only=False)
    teacher = build_model(shape, num_classes, model_cfg).to(device)
    teacher.load_state_dict(state["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = FlowMapConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, FlowMapConfig)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    # name= the run-dir basename (dataset + mode + timestamp) so the wandb run maps 1:1 to
    # its /workspace/runs folder instead of a random codename.
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
    # Teacher distillation: load a frozen 0001 prior, warm-start the student from it, and switch
    # the off-diagonal target to esd_teacher. Unset teacher_ckpt => from-scratch mf.
    teacher: BaseModel | None = None
    distillation_type = cfg.training.distillation_type
    if cfg.training.teacher_ckpt:
        teacher = load_teacher(
            cfg.training.teacher_ckpt,
            cfg.dataset.shape,
            cfg.dataset.num_classes,
            cfg.model,
            device,
        )
        model.load_state_dict(teacher.state_dict())  # warm-start the student
        distillation_type = "esd_teacher"
        print(f"[{EXPERIMENT}] distilling from teacher {cfg.training.teacher_ckpt}")

    if activate_x_cond_conditioning(model):
        print(f"[{EXPERIMENT}] activated x_cond conditioning (copied x_embedder)")

    # Resume: load weights (after x_cond activation so the checkpoint's trained embedder
    # wins) and hand the checkpoint's train_state to the loop.
    resume_state: dict | None = None
    if cfg.training.resume_from:
        ckpt = torch.load(
            Path(cfg.training.resume_from), map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model"])
        resume_state = ckpt.get("train_state") or {"epoch": int(ckpt["epoch"])}
        if "optimizer" not in resume_state:
            print(
                f"[{EXPERIMENT}] resume checkpoint has no train_state; "
                "restarting optimizer fresh from its weights"
            )
        print(
            f"[{EXPERIMENT}] resuming from {cfg.training.resume_from} "
            f"at epoch {resume_state['epoch'] + 1}"
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
    val_loss_fn = make_loss_fn(cfg.dataset.num_classes)  # step=0 → pure-FM diagonal

    # Fixed held-out references for the posterior-reconstruction panel.
    ref_batch = next(iter(val_loader))[0][: cfg.sampling.posterior_refs].to(device)

    # Fixed noise for reproducible per-epoch visualizations: same starting points every eval,
    # so the grids track model improvement on identical samples rather than fresh random draws.
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    eval_noise = torch.randn(
        cfg.sampling.n_eval_viz, *cfg.dataset.shape, device=device, generator=g
    )
    traj_noise = eval_noise[: cfg.sampling.n_traj_viz]

    @torch.no_grad()
    def _batch_val_loss(m: BaseModel, xb: torch.Tensor, lb: torch.Tensor) -> float:
        opt_losses, _ = val_loss_fn(m, None, xb.to(device), lb.to(device), step=0)
        return float(sum(opt_losses.values()).item())

    @torch.no_grad()
    def compute_val_loss(m: BaseModel) -> float:
        m.eval()
        total, n = 0.0, 0
        for xb, lb in val_loader:
            total += _batch_val_loss(m, xb, lb)
            n += 1
        return total / max(n, 1)

    # Junction times of the few-step sampler — where the off-diagonal consistency metric
    # compares flow-map jumps against the fine ODE (must land on the ODE's Euler grid).
    junction_ts = [
        k / cfg.sampling.few_steps for k in range(cfg.sampling.few_steps + 1)
    ]

    def on_eval(m: BaseModel, epoch: int) -> float | None:
        generated_sample, ode_states = sample(
            m,
            cfg.sampling.n_eval_viz,
            cfg.dataset.shape,
            sampler_steps=cfg.sampling.sampler_steps,
            device=device,
            x_noise=eval_noise,
            return_states_at=junction_ts,
        )

        p = run.ckpt_dir.parent / f"samples_epoch{epoch}.png"
        cfg.dataset.visualize(generated_sample, p)
        run.log_image("samples", p, caption=f"epoch {epoch + 1} ODE")
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
        few_step_generated_sample, few_hist = sample_few_step(
            m,
            cfg.sampling.n_eval_viz,
            cfg.dataset.shape,
            n_steps=cfg.sampling.few_steps,
            device=device,
            x_noise=eval_noise,  # same noise as the ODE grid → comparable cell-for-cell & across epochs
            return_hist=True,
        )
        pf = run.ckpt_dir.parent / f"samples_fewstep_epoch{epoch}.png"
        cfg.dataset.visualize(few_step_generated_sample, pf)
        run.log_image(
            "samples_fewstep",
            pf,
            caption=f"epoch {epoch + 1} {cfg.sampling.few_steps}-step",
        )
        # Off-diagonal validation: self-consistency of flow-map jumps against the fine ODE,
        # reusing the two trajectories above (same eval_noise → directly comparable).
        consistency = flow_map_consistency(m, eval_noise, ode_states, few_hist)
        run.log(epoch=epoch, **{f"val/{k}": v for k, v in consistency.items()})
        # Posterior reconstruction across a sweep of conditioning levels: each row is one held-out
        # reference, columns are [ref, recon@t...]. A conditioned model sharpens toward the ref as
        # t_cond -> 1; one that ignores x_cond returns the same generic sample in every column.
        t_grid = cfg.sampling.posterior_t_grid
        cols = [ref_batch] + [
            sample_posterior(
                m, ref_batch, t, n_steps=cfg.sampling.few_steps, device=device
            )
            for t in t_grid
        ]
        panel = torch.stack(cols, dim=1).reshape(-1, *cfg.dataset.shape)
        pr = run.ckpt_dir.parent / f"posterior_recon_epoch{epoch}.png"
        cfg.dataset.visualize(panel, pr, ncols=len(cols))
        run.log_image(
            "posterior_recon",
            pr,
            caption=f"epoch {epoch + 1}: rows=refs, cols=[ref, t={t_grid}]",
        )

        # Assess whether the generated images are overfitting by comparing with all training images.
        min_mse_train_sample, min_mse = assess_overfit(generated_sample, loader)
        min_mse_train_sample_few_step, min_mse_few_step = assess_overfit(
            few_step_generated_sample, loader
        )
        # plot the images with the lowest mse side by side.

        assess_overfit_p = run.ckpt_dir.parent / "assess_overfit.png"
        cfg.dataset.visualize(min_mse_train_sample, assess_overfit_p)
        run.log_image(
            "assess_overfit",
            assess_overfit_p,
            caption=f"assess_overfit: min_mse={min_mse.mean().item():.4f}",
        )
        run.log(
            epoch=epoch, **{"overfit/min_mse_training_image": min_mse.mean().item()}
        )

        assess_overfit_pf = run.ckpt_dir.parent / "assess_overfit_few_step.png"
        cfg.dataset.visualize(min_mse_train_sample_few_step, assess_overfit_pf)
        run.log_image(
            "assess_overfit_few_step",
            assess_overfit_pf,
            caption=f"assess_overfit_few_step: min_mse={min_mse_few_step.mean().item():.4f}",
        )
        run.log(
            epoch=epoch,
            **{
                "overfit/min_mse_training_image_few_step": min_mse_few_step.mean().item()
            },
        )

        val_loss = compute_val_loss(m)
        return val_loss

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
        log_step=run.log_step,
        warmup_steps=cfg.training.warmup_steps,
        flow_map_warmup_steps=cfg.training.flow_map_warmup_steps,
        flow_map_anneal_end=cfg.training.flow_map_anneal_end,
        distillation_type=distillation_type,
        loss_weighting=cfg.training.loss_weighting,
        uncond_prob=cfg.training.uncond_prob,
        t_cond_power=cfg.training.t_cond_power,
        t_cond_warmup_steps=cfg.training.t_cond_warmup_steps,
        teacher_model=teacher,
        ema_enabled=cfg.training.ema.enabled,
        ema_decay=cfg.training.ema.decay,
        ema_warmup_steps=cfg.training.ema.warmup_steps,
        eval_every_epochs=cfg.training.eval_every_epochs,
        on_eval=on_eval,
        ckpt_every_epochs=cfg.training.ckpt_every_epochs,
        on_checkpoint=on_checkpoint,
        precision=cfg.training.precision,
        resume=resume_state,
    )
    # Mean total loss over the final epoch's steps (one minibatch is too noisy).
    last_epoch = history[-1]["epoch"]
    last = [h["total"] for h in history if h["epoch"] == last_epoch]
    final_loss = sum(last) / len(last)
    eval_model = ema_model if ema_model is not None else model

    final_val_loss = compute_val_loss(eval_model)
    summary: dict[str, float] = {
        "val/loss": final_val_loss,
        "train/final_loss": final_loss,
    }
    run.finish(**summary)


if __name__ == "__main__":
    main()
