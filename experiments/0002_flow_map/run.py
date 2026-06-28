"""Train a meta flow map (mfm, from scratch) on swappable datasets via Hydra.

    uv run python experiments/0002_flow_map/run.py                       # gaussians
    uv run python experiments/0002_flow_map/run.py experiment=mnist
    uv run python experiments/0002_flow_map/run.py experiment=openfwi
    uv run python experiments/0002_flow_map/run.py experiment=smoke

Same harness as 0001 (same datasets/model/EMA/eval/checkpoint), but the loss adds mfm's
off-diagonal ``s<u`` consistency term (enabled after ``flow_map_warmup_steps``), training a
flow map that maps between arbitrary interpolant times — not just the noise→data diagonal.

Two regimes, selected by ``training.teacher_ckpt``:
  * unset → train from scratch (the self-contained ``mf`` target, no teacher).
  * set to a 0001 checkpoint (local path or ``wandb`` artifact ref) → distil the off-diagonal
    from that frozen prior (``esd_teacher``) and warm-start the student from it. Unconditional
    datasets only (gaussians, openfwi); the conditional teacher path is ImageNet-specific in mfm.

    uv run python experiments/0002_flow_map/run.py experiment=openfwi \\
        training=teacher training.teacher_ckpt=runs/0001_flow_matching/<run>/checkpoints/step_9_ema.pt

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
    activate_x_cond_conditioning,
    build_model,
    count_parameters,
)
from physics_informed_flow_map.flow_matching.sample import (
    sample,
    sample_few_step,
    sample_posterior,
)
from physics_informed_flow_map.flow_matching.train import make_loss_fn, train

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
    flow_map_warmup_steps: int = Field(
        1000, ge=0
    )  # off-diagonal term on after this step
    flow_map_anneal_end: int = Field(20000, gt=0)
    distillation_type: str = "mf"
    loss_weighting: str = "adaptive"
    # Conditioning schedule. t_cond is mfm's "conditioning time" = the noise level at which the
    # map is told it observed a context state x_cond along x_τ=(1-τ)·noise+τ·data: t_cond=0 sees
    # pure noise (unconditional), t_cond>0 sees a partial observation, so the map learns the
    # posterior p(data | x_{t_cond}) that inference steers (mfm's velocity API fixes the name
    # `t_cond`). uncond_prob = P(a sample is unconditional, t_cond=0); 0.1 ⇒ 90% conditional.
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

    # Enable the x_cond posterior-conditioning pathway (mfm's dead-init trap; see the helper).
    # After any warm-start so it copies the warm-started x_embedder.
    if activate_x_cond_conditioning(model):
        print(f"[{EXPERIMENT}] activated x_cond conditioning (copied x_embedder)")

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
        num_workers=0,
    )
    val_loss_fn = make_loss_fn(cfg.dataset.num_classes)  # step=0 → pure-FM diagonal

    # Fixed held-out references for the posterior-reconstruction panel.
    ref_batch = next(iter(val_loader))[0][: cfg.sampling.posterior_refs].to(device)

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
