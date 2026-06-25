"""Diffusion-prior + DPS baseline for FWI: train a diffusion prior over FlatVel-A velocity
maps, then DPS-invert the same held-out map the flow PoC used (camp-A comparison).

Trains an unconditional ``diffusers`` UNet diffusion prior, saves its weights, simulates
seismic data ``d`` from a held-out FlatVel-A map with the Deepwave forward operator, and runs
canonical DPS toward ``d`` to recover the velocity. Plain script (no Hydra). Example:

    uv run python experiments/0002_fwi_tilting/train_diffusion.py --epochs 100 --steps 200 \
        --guidance 0.5 --n-samples 4

A ``--smoke`` flag runs a tiny end-to-end pass (few epochs, few steps, CPU-friendly) for
plumbing checks. Reports per-sample MAE/RMSE (m/s) vs the true map and the data-misfit
reduction vs an unguided sample, and writes a ``true | best v_hat | error`` figure next to
this script. Quantitative head-to-head with the flow PoC is a follow-up.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from physics_informed_flow_map.baselines import (
    build_denoiser,
    dps_sample,
    train_diffusion_prior,
)
from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.flow_matching.openfwi import VMAX, VMIN
from physics_informed_flow_map.physics.forward import simulate

RESOLUTION = 64


def to_mps70(v_norm: torch.Tensor) -> torch.Tensor:
    """(B,1,64,64) in [-1,1] -> (B,70,70) velocity in m/s."""
    v70 = F.interpolate(v_norm, size=70, mode="bilinear", align_corners=False)
    return ((v70 + 1.0) / 2.0 * (VMAX - VMIN) + VMIN)[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-timesteps", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=200, help="DPS reverse steps")
    ap.add_argument("--guidance", type=float, default=0.5)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "diffusion_prior.pt"))
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny end-to-end pass for plumbing checks (overrides epochs/steps)",
    )
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps, args.train_timesteps, args.n_samples = 1, 10, 50, 2

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Diffusion prior: diffusers UNet + DDPM scheduler, trained on FlatVel-A maps.
    denoiser = build_denoiser("unet", sample_size=RESOLUTION, channels=1).to(dev)
    scheduler = DDPMScheduler(num_train_timesteps=args.train_timesteps)
    dataset = OpenFWIDatasetConfig(resolution=RESOLUTION).build()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    print(f"training diffusion prior: {len(dataset)} maps, {args.epochs} epochs")
    history = train_diffusion_prior(
        denoiser,
        scheduler,
        loader,
        n_epochs=args.epochs,
        lr=args.lr,
        device=dev,
        log=lambda **r: None,
    )
    if history:
        print(f"  final train loss {history[-1]['loss']:.4f}")
    torch.save({"model": denoiser.state_dict()}, args.ckpt)
    print(f"  saved prior -> {args.ckpt}")

    # Held-out FlatVel-A map (native 70x70, m/s) -> observed seismic data.
    f = sorted(glob.glob("data/openfwi/FlatVel_A/model/*.npy"))[-1]
    v_true = (
        torch.from_numpy(np.ascontiguousarray(np.load(f, mmap_mode="r")[-1, 0]))
        .float()
        .to(dev)
    )
    d_obs = simulate(v_true).detach()

    def forward_fn(v_norm: torch.Tensor) -> torch.Tensor:
        v_mps = to_mps70(v_norm)
        return torch.stack([simulate(v_mps[b]) for b in range(v_mps.shape[0])])

    denoiser.eval()
    guided = dps_sample(
        denoiser,
        scheduler,
        (1, RESOLUTION, RESOLUTION),
        forward_fn,
        d_obs,
        n_samples=args.n_samples,
        num_steps=args.steps,
        guidance_strength=args.guidance,
        device=dev,
        normalize_grad=True,
    )
    unguided = dps_sample(
        denoiser,
        scheduler,
        (1, RESOLUTION, RESOLUTION),
        forward_fn,
        d_obs,
        n_samples=args.n_samples,
        num_steps=args.steps,
        guidance_strength=0.0,
        device=dev,
        normalize_grad=True,
    )

    vg = to_mps70(guided)
    mae = (vg - v_true).abs().mean(dim=(1, 2))
    rmse = ((vg - v_true) ** 2).mean(dim=(1, 2)).sqrt()
    dm_g = ((forward_fn(guided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((forward_fn(unguided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    best = int(mae.argmin())

    print(f"guidance={args.guidance:g}  steps={args.steps}  n={args.n_samples}")
    print(
        f"  MAE (m/s):  {[round(x) for x in mae.tolist()]}  best={round(float(mae[best]))}"
    )
    print(f"  RMSE (m/s): {[round(x) for x in rmse.tolist()]}")
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  "
        f"unguided={float(dm_u.mean()):.3e}  ratio={float(dm_g.mean() / dm_u.mean()):.3f}"
    )

    out = Path(__file__).parent / "train_diffusion_result.png"
    vt = v_true.cpu().numpy()
    vh = vg[best].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(vt, cmap="viridis")
    ax[0].set_title("true v")
    ax[0].axis("off")
    ax[1].imshow(vh, cmap="viridis", vmin=vt.min(), vmax=vt.max())
    ax[1].set_title(f"v_hat (MAE {round(float(mae[best]))} m/s)")
    ax[1].axis("off")
    im = ax[2].imshow(vh - vt, cmap="RdBu", vmin=-500, vmax=500)
    ax[2].set_title("error")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
