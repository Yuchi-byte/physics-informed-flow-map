"""
Generate images using the pretrained MFM-XL/2 checkpoint (GPU).

Prerequisites — run once:
    # 1. Download the pretrained MFM checkpoint (~2.7 GB)
    uv run hf download adh1s/mfm --include "mfm-xl2.pt" --local-dir ckpts

    # 2. diffusers is already added (VAE decoder for latent → RGB image)

Usage:
    uv run python experiments/pretrained_sample.py
    uv run python experiments/pretrained_sample.py --n_samples 4 --cfg_scale 2.5
    uv run python experiments/pretrained_sample.py --classes 207 360 985
    uv run python experiments/pretrained_sample.py --class_names "golden retriever" "daisy" "volcano"
    uv run python experiments/pretrained_sample.py --sampler_steps 4

ImageNet class examples (by index):
    207  = golden retriever
    360  = otter
    985  = daisy
    980  = volcano
    388  = giant panda

Estimated runtime on A4000 GPU:
    n_samples=4, sampler_steps=1  →  ~4s
    n_samples=4, sampler_steps=4  →  ~15s
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages/mfm-meta-flow-map-main/src"))

from mfm.SI import Linear
from mfm.SI.samplers import kernel_sampler_fn
from mfm.models import DiTMFM
from mfm.models.model_wrapper import SIModelWrapper


# ── ImageNet class names (all 1000) ────────────────────────────────────────────
_CLASSES_JSON = ROOT / "packages/mfm-meta-flow-map-main/src/mfm/losses/imagenet_classes.json"
IMAGENET_NAMES: list[str] = json.loads(_CLASSES_JSON.read_text())  # index = class id
NAME_TO_CLASS: dict[str, int] = {name.lower(): i for i, name in enumerate(IMAGENET_NAMES)}


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MFM pretrained inference")
    p.add_argument(
        "--ckpt",
        type=str,
        default=str(ROOT / "ckpts/mfm-xl2.pt"),
        help="Path to mfm-xl2.pt checkpoint",
    )
    p.add_argument(
        "--n_samples", type=int, default=4, help="Number of images to generate"
    )
    p.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale (1.0 = no guidance)",
    )
    p.add_argument(
        "--sampler_steps",
        type=int,
        default=4,
        help="Consistency sampler steps (1 = single jump/fastest, 4+ = better quality)",
    )
    p.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="ImageNet class indices to generate (e.g. 207 985)",
    )
    p.add_argument(
        "--class_names",
        type=str,
        nargs="*",
        default=None,
        help="ImageNet class names to generate (e.g. 'golden retriever' 'daisy')",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=str(ROOT / "outputs/pretrained_sample"))
    return p.parse_args()


# ── VAE decoder (CPU-compatible rewrite of get_imagenet_vae_fn) ────────────────
def load_vae(device: torch.device):
    """
    Load the Stable Diffusion VAE for decoding latents → RGB images.
    Download is ~335 MB, cached in ~/.cache/huggingface after first run.

    Original get_imagenet_vae_fn() in mfm/utils/steering.py hardcodes CUDA
    in encode_fn. We only need the decoder here, so we rewrite it cleanly.
    """
    print("Loading VAE (stabilityai/sd-vae-ft-mse) …")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae = vae.to(device).eval()

    # Latent scale factor used by the SD VAE
    scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215], device=device).view(
        1, 4, 1, 1
    )

    @torch.no_grad()
    def decode(latents: torch.Tensor) -> torch.Tensor:
        """latents: [B, 4, 32, 32] in MFM-normalised space → [B, 3, 256, 256] in [0,1]"""
        x = latents / scale  # undo MFM normalisation
        images = vae.decode(x).sample  # → [B, 3, 256, 256] in [-1, 1]
        images = (images + 1.0) / 2.0  # → [0, 1]
        return images.clamp(0.0, 1.0)

    print("VAE loaded ✓")
    return decode


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> SIModelWrapper:
    """
    Load the pretrained DiTMFM-XL/2 from checkpoint.

    Architecture is read from conf/model/sit_xl_2.yaml:
        hidden_size=1152, depth=28, num_heads=16, input_size=32, in_channels=4
        attn_func="base"  ← standard PyTorch attention, no FlashAttention needed
    """
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt}\n\n"
            "Download it first:\n"
            "    uv run hf download adh1s/mfm --include 'mfm-xl2.pt' --local-dir ckpts\n"
        )

    print(f"Loading checkpoint: {ckpt}  ({ckpt.stat().st_size / 1e9:.1f} GB)")
    start = time.time()

    # Build model — matches conf/model/sit_xl_2.yaml exactly
    dit = DiTMFM(
        learn_loss_weighting=False,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        label_dim=1000,
        encoder_depth=20,
        attn_func="base",  # standard PyTorch attention (no FlashAttention dependency)
        is_zero_data=True,
        use_joint_attention=False,
        preserve_t_cond_0=False,
        model_guidance_class_ws=[2.5],
        model_guidance_x_cond_ws=[],
    )

    SI = Linear(t_max=1.0)
    model = SIModelWrapper(dit, SI, use_parametrization=False)

    # Load weights on CPU regardless of how they were saved
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Checkpoints are sometimes saved with a "model." prefix — strip it
    state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]} …")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]} …")

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model loaded in {time.time() - start:.1f}s  —  {n_params / 1e6:.0f}M parameters ✓"
    )

    model = model.to(device).eval()
    return model


# ── Sampling ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(
    model,
    decode_fn,
    n_samples: int,
    cfg_scale: float,
    sampler_steps: int,
    class_labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate images via:
      1. Pure Gaussian noise in latent space  [B, 4, 32, 32]
      2. Consistency sampler: noise → denoised latent  (sampler_steps jumps)
      3. VAE decoder: latent → RGB image  [B, 3, 256, 256]

    cfg_scale > 1.0 applies classifier-free guidance:
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
    """
    latent_shape = (4, 32, 32)  # MFM latent space for 256×256 images
    image_shape = (3, 256, 256)  # after VAE decode

    print(f"\nGenerating {n_samples} images")
    print(f"  Sampler steps : {sampler_steps}  (1 = single direct jump)")
    print(f"  CFG scale     : {cfg_scale}")
    named = [(c, IMAGENET_NAMES[c]) for c in class_labels.tolist()]
    print(f"  Classes       : {named}")
    print(f"  Device        : {device}")
    print()

    start = time.time()
    samples = kernel_sampler_fn(
        model=model,
        shape=latent_shape,
        shape_decoded=image_shape,
        SI=Linear(t_max=1.0),
        n_samples=n_samples,
        n_batch_size=n_samples,
        n_steps=sampler_steps,
        inverse_scaler_fn=decode_fn,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
    )
    elapsed = time.time() - start
    print(
        f"Generated {n_samples} images in {elapsed:.1f}s  "
        f"({elapsed / n_samples:.1f}s per image)"
    )
    return samples


# ── Save ───────────────────────────────────────────────────────────────────────
def save(
    samples: torch.Tensor,
    class_labels: torch.Tensor,
    output_dir: str,
    cfg_scale: float,
    sampler_steps: int,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < n:
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            cls = class_labels[i].item()
            ax.set_title(IMAGENET_NAMES[cls], fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"MFM-XL/2 pretrained  |  CFG={cfg_scale}  |  steps={sampler_steps}",
        fontsize=11,
    )
    plt.tight_layout()

    name = f"pretrained_cfg{cfg_scale}_steps{sampler_steps}.png"
    path = out / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved →  {path}")

    # Also save individual PNGs
    for i, img_tensor in enumerate(samples):
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        individual = out / f"{i:03d}_{IMAGENET_NAMES[class_labels[i].item()].replace(' ', '_')}.png"
        PILImage.fromarray(img_np).save(individual)

    print(f"Saved {n} individual PNGs to {out}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print(f"Device : {device}  ({torch.cuda.get_device_name(0)})")

    # Class labels — resolved from names, indices, or random
    if args.class_names:
        indices = []
        for name in args.class_names:
            key = name.lower()
            if key not in NAME_TO_CLASS:
                # Fall back to partial match
                matches = [n for n in NAME_TO_CLASS if key in n]
                if not matches:
                    raise ValueError(f"Unknown class name: '{name}'. No match in ImageNet-1k.")
                key = matches[0]
                print(f"  '{name}' → '{key}' (class {NAME_TO_CLASS[key]})")
            indices.append(NAME_TO_CLASS[key])
        labels = torch.tensor(indices[: args.n_samples], device=device)
        # Pad with random if fewer names than n_samples
        if len(labels) < args.n_samples:
            extra = torch.randint(0, 1000, (args.n_samples - len(labels),), device=device)
            labels = torch.cat([labels, extra])
    elif args.classes:
        labels = torch.tensor(args.classes[: args.n_samples], device=device)
        # Pad with random if fewer classes than n_samples
        if len(labels) < args.n_samples:
            extra = torch.randint(
                0, 1000, (args.n_samples - len(labels),), device=device
            )
            labels = torch.cat([labels, extra])
    else:
        labels = torch.randint(0, 1000, (args.n_samples,), device=device)

    # Load
    model = load_model(args.ckpt, device)
    decode_fn = load_vae(device)

    # Generate
    samples = generate(
        model,
        decode_fn,
        n_samples=args.n_samples,
        cfg_scale=args.cfg_scale,
        sampler_steps=args.sampler_steps,
        class_labels=labels,
        device=device,
    )

    # Save
    save(samples, labels, args.output_dir, args.cfg_scale, args.sampler_steps)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
