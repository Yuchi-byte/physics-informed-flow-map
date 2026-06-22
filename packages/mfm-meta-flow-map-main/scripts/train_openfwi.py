"""
Flow matching training on OpenFWI velocity maps (no teacher, no distillation, no CFG).

Pixel-space training — no VAE. The DiT operates directly on 64x64 grayscale maps.

Run:
    python scripts/train_openfwi.py data_dir=/workspace/data/openfwi
"""

import math
import os

import hydra
import hydra.core.hydra_config
import lightning as pl
import torch
import torchvision
import wandb
from diffusers import AutoencoderKL
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from mfm.data.openfwi import OpenFWIVelocityDataset
from mfm.models.model_wrapper import SIModelWrapper
from mfm.SI.samplers import ode_sampler_fn
from mfm.utils import EMAWeightAveraging

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def image_scaler(x):
    return (x - 0.5) * 2

def inverse_image_scaler(x):
    return (x / 2) + 0.5


class CustomImageDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.trainer.batch_size
        self.num_workers = cfg.trainer.num_workers
        self.resolution = cfg.dataset.img_resolution
        self.dataset_name = cfg.dataset.get("name", "imagefolder")
        self.inverse_scaler = inverse_image_scaler

        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _make_openfwi(self, families):
        return OpenFWIVelocityDataset(
            root=self.data_dir,
            families=families,
            resolution=self.resolution,
            normalize=True,
        )

    def setup(self, stage=None):
        if self.dataset_name == "openfwi":
            # OpenFWI has no canonical train/val split in the file layout;
            # we do an 80/20 random split here.
            import torch
            full = self._make_openfwi(families=None)
            n_val = max(1, int(0.2 * len(full)))
            n_train = len(full) - n_val
            self.train_ds, self.val_ds = torch.utils.data.random_split(
                full, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            if stage in ("fit", None):
                self.train_ds = ImageFolder(os.path.join(self.data_dir, "train"), self.transform)
                self.val_ds   = ImageFolder(os.path.join(self.data_dir, "val"),   self.transform)
            if stage in ("test", None):
                self.test_ds = ImageFolder(os.path.join(self.data_dir, "val"), self.transform)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        ds = getattr(self, "test_ds", self.val_ds)
        return DataLoader(ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)


# ---------------------------------------------------------------------------
# Training module
# ---------------------------------------------------------------------------

def broadcast(t, shape):
    return t.view(-1, *((1,) * (len(shape) - 1)))


class FlowMatchingModule(pl.LightningModule):
    def __init__(self, cfg, model, SI):
        super().__init__()
        self.model = model
        self.SI = SI
        self.cfg = cfg
        self.use_vae = cfg.dataset.get("use_vae", True)

        if self.use_vae:
            self._vae_container = []
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            vae.eval()
            for p in vae.parameters():
                p.requires_grad = False
            self._vae_container.append(vae)
            self.register_buffer("latents_scale", torch.tensor([0.18215] * 4).view(1, 4, 1, 1))
            self.register_buffer("latents_bias",  torch.zeros(1, 4, 1, 1))

    @property
    def vae(self):
        return self._vae_container[0] if self.use_vae else None

    def setup(self, stage):
        if self.use_vae:
            self.vae.to(self.device)

    @torch.no_grad()
    def _encode(self, x):
        if not self.use_vae:
            return x
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        with torch.amp.autocast("cuda", enabled=False):
            latents = self.vae.encode(x.to(self.vae.dtype)).latent_dist.sample()
        return ((latents - self.latents_bias) * self.latents_scale).to(x.dtype)

    @torch.no_grad()
    def _decode(self, x):
        if not self.use_vae:
            return x.clamp(-1, 1)
        latents_f = x / self.latents_scale + self.latents_bias
        with torch.amp.autocast("cuda", enabled=False):
            images = self.vae.decode(latents_f.to(self.vae.dtype)).sample
        return inverse_image_scaler(images.to(x.dtype)).clamp(0, 1)

    def _fm_loss(self, x1):
        N = x1.shape[0]
        x0 = torch.randn_like(x1)
        t  = torch.rand(N, device=x1.device)

        alpha_t, beta_t = self.SI.get_coefficients(t)
        alpha_t = broadcast(alpha_t, x1.shape)
        beta_t  = broadcast(beta_t,  x1.shape)
        xt = alpha_t * x0 + beta_t * x1

        t_cond  = torch.zeros(N, device=x1.device)
        x_cond  = torch.zeros_like(x1)
        dummy_labels = torch.zeros(N, dtype=torch.long, device=x1.device)

        v_pred = self.model.v(t, t, xt, t_cond, x_cond,
                              class_labels=dummy_labels,
                              cfg_scale=torch.ones(N, device=x1.device))
        target = x1 - x0
        return ((v_pred - target) ** 2).mean()

    def training_step(self, batch, _):
        x, _ = batch
        x = self._encode(x)
        loss = self._fm_loss(x)
        self.log("train/fm_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, _ = batch
        x = self._encode(x)
        loss = self._fm_loss(x)
        self.log("val/fm_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=self.cfg.lr.val)
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0,
            total_iters=self.cfg.lr.warmup_steps
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------------------
# Sampling callback (unconditional, logs a grid to wandb)
# ---------------------------------------------------------------------------

class SamplingCallback(Callback):
    def __init__(self, cfg, SI, image_shape):
        super().__init__()
        self.cfg = cfg
        self.SI = SI
        self.image_shape = image_shape

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.cfg.sampling.every_n_steps != 0 or step == 0:
            return
        if trainer.global_rank != 0:
            return

        pl_module.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            n = self.cfg.sampling.n_samples
            device = pl_module.device
            x0 = torch.randn(n, *self.image_shape, device=device)
            dummy_labels = torch.zeros(n, dtype=torch.long, device=device)
            t_cond = torch.zeros(n, device=device)
            x_cond = torch.zeros(n, *self.image_shape, device=device)
            cfg_scales = torch.ones(n, device=device)
            samples = ode_sampler_fn(
                pl_module.model,
                xt_cond=x_cond,
                t_cond=t_cond,
                n_steps=self.cfg.sampling.n_steps,
                solver="euler",
                eps_start=x0,
                v_type="model_guidance_class",
                labels=dummy_labels,
                cfg_scales=cfg_scales,
            )
            images = pl_module._decode(samples)

        nrow = math.ceil(math.sqrt(n))
        grid = torchvision.utils.make_grid(images.cpu(), nrow=nrow)

        # Save grid to disk
        samples_dir = os.path.join(self.cfg.work_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(samples_dir, f"step_{step:07d}.png"))

        # Also log to WandB
        pl_module.logger.experiment.log({
            "val/samples": [wandb.Image(grid)],
            "global_step": step,
        })
        pl_module.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf/", config_name="config_train_openfwi.yaml", version_base="1.3")
def main(cfg: DictConfig):
    print("start main()")
    seed_everything(cfg.seed, workers=True)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.get("entity", None),
        config=dict(cfg),
    )

    model = instantiate(cfg.model)
    SI    = instantiate(cfg.SI)
    model = SIModelWrapper(model, SI, use_parametrization=False)
    print("finish initialising model")

    datamodule = CustomImageDataModule(cfg)

    train_module = FlowMatchingModule(cfg, model, SI)

    use_vae = cfg.dataset.get("use_vae", True)
    if use_vae:
        # VAE downsamples 8× and outputs 4 channels
        spatial = cfg.dataset.img_resolution // 8
        image_shape = (4, spatial, spatial)
    else:
        # pixel-space: shape matches the model's input directly
        image_shape = (cfg.model.in_channels, cfg.model.input_size, cfg.model.input_size)
    
    OmegaConf.set_struct(cfg, False)
    cfg.work_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )

    print("before ema_callback")
    ema_callback = EMAWeightAveraging(cfg.trainer.ema.decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.work_dir}/checkpoints",
        filename="step-{step}",
        every_n_train_steps=cfg.trainer.checkpoint_every_n_steps,
        save_top_k=-1,
    )
    sampling_callback = SamplingCallback(cfg, SI, image_shape)

    print("before trainer")

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_steps=cfg.trainer.num_train_steps,
        accelerator="gpu",
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[ema_callback, checkpoint_callback, sampling_callback],
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_sanity_val_steps=0,
    )
    print("trainer set. Starting trainer.fit")
    trainer.fit(train_module, datamodule=datamodule,
                ckpt_path=cfg.get("resume_from_checkpoint", None))
    print("end of main")


if __name__ == "__main__":
    main()
