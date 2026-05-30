"""
MFM Pipeline: Train on MNIST, generate digit images.

This script demonstrates the full MFM pipeline:
  1. Load MNIST from torchvision (auto-downloaded to data/)
  2. Train a tiny DiTMFM on the images
  3. Sample new images using the consistency sampler
  4. Save generated images to outputs/

Usage:
    uv run python experiments/mnist_pipeline.py
    uv run python experiments/mnist_pipeline.py --n_steps 500 --n_samples 16
    uv run python experiments/mnist_pipeline.py --n_steps 50   # quick smoke-test

Pretrained ImageNet model (requires GPU):
    See packages/mfm-meta-flow-map-main/README.md — download mfm-xl2.pt from
    https://huggingface.co/adh1s/mfm then use scripts/sample.py
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ── MFM imports ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages/mfm-meta-flow-map-main/src"))

from mfm.SI import Linear
from mfm.SI.samplers import consistency_sampler_fn
from mfm.losses.losses import get_consistency_loss_fn
from mfm.models import DiTMFM


# ── Config helper (mirrors Hydra OmegaConf DictConfig) ────────────────────────
class Cfg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MFM MNIST pipeline")
    p.add_argument("--n_steps",   type=int,   default=200,  help="Training steps")
    p.add_argument("--n_samples", type=int,   default=16,   help="Images to generate")
    p.add_argument("--batch_size",type=int,   default=32,   help="Training batch size")
    p.add_argument("--lr",        type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    p.add_argument("--data_dir",  type=str,   default=str(ROOT / "data"),
                   help="Where to download MNIST")
    p.add_argument("--output_dir",type=str,   default=str(ROOT / "outputs"),
                   help="Where to save generated images and checkpoints")
    p.add_argument("--save_ckpt", action="store_true",
                   help="Save model checkpoint after training")
    p.add_argument("--sampler_steps", type=int, default=4,
                   help="Consistency sampler steps at inference (1 = direct jump)")
    return p.parse_args()


# ── Data ───────────────────────────────────────────────────────────────────────
def get_mnist_loader(data_dir: str, batch_size: int, image_size: int = 32):
    """
    Download MNIST and return a DataLoader.
    Images are resized to image_size×image_size and normalised to [-1, 1].
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),                        # [0, 1]
        T.Normalize(mean=[0.5], std=[0.5]),  # → [-1, 1]
    ])
    dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )
    print(f"MNIST loaded: {len(dataset):,} images  →  "
          f"{image_size}×{image_size} greyscale, normalised to [-1, 1]")
    return loader


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(image_size: int = 32, device: torch.device = torch.device("cpu")):
    """
    Tiny DiTMFM that fits on CPU.

    input_size=32, patch_size=4  →  (32/4)² = 64 patches
    hidden_size=128, depth=4, num_heads=4
    ~1.6M parameters total.
    """
    cfg = dict(
        input_size=image_size,
        patch_size=4,
        in_channels=1,          # MNIST is greyscale
        hidden_size=128,
        depth=4,
        num_heads=4,
        label_dim=10,           # MNIST has 10 digit classes
        encoder_depth=2,
        attn_func="base",       # standard attention — no FlashAttention needed
        is_zero_data=True,
        learn_sigma=False,
    )
    model = DiTMFM(learn_loss_weighting=False, **cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Model: Tiny DiTMFM  ({n:,} parameters)  on {device}")
    return model


# ── Loss / training config ─────────────────────────────────────────────────────
def build_cfg():
    return Cfg(
        SI=Cfg(t_max=1.0),
        trainer=Cfg(
            t_cond_warmup_steps=0,
            t_cond_0_rate=0.1,
            t_cond_power=1.0,
            num_warmup_steps=50,       # diagonal-only for first 50 steps
            anneal_end_step=200,       # full off-diagonal range after 200 steps
            class_dropout_prob=0.1,
        ),
        model=Cfg(
            label_dim=10,
            learn_loss_weighting=False,
            model_guidance_class_ws=[],
            model_guidance_x_cond_ws=[],
            init="dmf",
        ),
        loss=Cfg(
            data_fm=True,
            distill_fm=False,
            distillation_type="mf",
            model_guidance=False,
            model_guidance_base_prob=0.5,
            fm_loss_type="l2",
            distillation_loss_type="l2",
            distill_fm_loss_type="l2",
            distill_teacher_stop_grad=True,
            fm_adaptive_loss_p=None,
            fm_adaptive_loss_c=None,
            distill_adaptive_loss_p=None,
            distill_adaptive_loss_c=None,
        ),
    )


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, loader, n_steps: int, lr: float, device: torch.device):
    SI       = Linear(t_max=1.0)
    cfg      = build_cfg()
    loss_fn  = get_consistency_loss_fn(cfg, SI)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    data_iter = iter(loader)
    history   = []

    print(f"\nTraining for {n_steps} steps  (batch_size={loader.batch_size}, lr={lr})")
    print(f"{'Step':>6}  {'FM Loss':>10}  {'Distill':>10}  {'Total':>10}")
    print("-" * 44)

    for step in range(n_steps):
        # Cycle through the dataset
        try:
            x1, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1, labels = next(data_iter)

        x1     = x1.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        opt_losses, _ = loss_fn(model, None, x1, labels, step=step)
        total = sum(opt_losses.values())
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        fm  = opt_losses["fm_loss"].item()
        dis = opt_losses["distillation_loss"].item()
        tot = total.item()
        history.append({"step": step, "fm": fm, "distill": dis, "total": tot})

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(f"{step:>6}  {fm:>10.4f}  {dis:>10.4f}  {tot:>10.4f}")

    print(f"\nFinal loss: {history[-1]['total']:.4f}  "
          f"(started at {history[0]['total']:.4f})")
    return history, SI


# ── Sampling ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, SI, n_samples: int, sampler_steps: int,
             device: torch.device, image_size: int = 32):
    """
    Generate n_samples images using the consistency sampler.

    MFM samples by:
      1. Starting from pure noise x₀ ~ N(0, I)
      2. Conditioning on that same noise at t_cond=0 (unconditional generation)
      3. Jumping directly to the data manifold in sampler_steps steps
    """
    model.eval()

    # Unconditional generation: condition on t_cond=0 (no noisy observation)
    t_cond = torch.zeros(n_samples, device=device)
    x_noise = torch.randn(n_samples, 1, image_size, image_size, device=device)

    print(f"\nGenerating {n_samples} images  (sampler_steps={sampler_steps})")
    samples = consistency_sampler_fn(
        model, x_noise, t_cond,
        n_steps=sampler_steps,
        eps_start=x_noise,
    )

    # Denormalise: [-1, 1] → [0, 1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    print(f"Done.  Sample stats: mean={samples.mean():.3f}  std={samples.std():.3f}")
    return samples


# ── Saving ─────────────────────────────────────────────────────────────────────
def save_outputs(samples: torch.Tensor, history: list,
                 output_dir: str, n_steps: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Generated image grid ──────────────────────────────────────────────────
    n = len(samples)
    ncols = min(8, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(
        f"MFM Generated Images (MNIST, {n_steps} training steps)",
        fontsize=11
    )
    plt.tight_layout()
    img_path = out / f"generated_n{n_steps}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved image grid  →  {img_path}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    steps  = [h["step"]   for h in history]
    totals = [h["total"]  for h in history]
    fms    = [h["fm"]     for h in history]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, totals, label="Total loss",   color="black",    linewidth=2)
    ax.plot(steps, fms,    label="FM loss",      color="steelblue",linewidth=1.5, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = out / f"loss_n{n_steps}.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve   →  {loss_path}")

    # ── Raw tensors ───────────────────────────────────────────────────────────
    tensor_path = out / f"samples_n{n_steps}.pt"
    torch.save(samples.cpu(), tensor_path)
    print(f"Saved sample tensors →  {tensor_path}")


def save_checkpoint(model, output_dir: str, n_steps: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / f"model_n{n_steps}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint   →  {ckpt_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")
    print(f"Output : {args.output_dir}")

    # 1. Data
    loader = get_mnist_loader(args.data_dir, args.batch_size)

    # 2. Model
    model = build_model(image_size=32, device=device)

    # 3. Train
    history, SI = train(model, loader, args.n_steps, args.lr, device)

    # 4. Generate
    samples = generate(model, SI, args.n_samples, args.sampler_steps, device)

    # 5. Save
    save_outputs(samples, history, args.output_dir, args.n_steps)
    if args.save_ckpt:
        save_checkpoint(model, args.output_dir, args.n_steps)

    print("\n✅ Pipeline complete!")
    print(f"   Generated images : {args.output_dir}/generated_n{args.n_steps}.png")
    print(f"   Loss curve       : {args.output_dir}/loss_n{args.n_steps}.png")


if __name__ == "__main__":
    main()
