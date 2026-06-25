"""Proof-of-concept: invert a held-out FlatVel-A velocity map by DPS-style tilting.

Takes the trained flow prior over OpenFWI velocity maps, simulates seismic data ``d`` from
a held-out map with the Deepwave forward operator, then guided-samples the prior toward
``d`` to recover the velocity. Plain script (no Hydra). Example:

    uv run python experiments/0002_fwi_tilting/poc.py --guidance 1e-9 --steps 100 --n-samples 2

Reports per-sample MAE/RMSE (m/s) vs the true map and the data-misfit reduction vs an
unguided sample, and writes a ``true | best v_hat | error`` figure next to this script.
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

from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.flow_matching.openfwi import VMAX, VMIN
from physics_informed_flow_map.physics.forward import simulate
from physics_informed_flow_map.physics.tilt import guided_sample

CKPT = "runs/0001_flow_matching/2026-06-25T14-18-57Z/checkpoints/step_99_ema.pt"


def to_mps70(v_norm: torch.Tensor) -> torch.Tensor:
    """(B,1,64,64) in [-1,1] -> (B,70,70) velocity in m/s."""
    v70 = F.interpolate(v_norm, size=70, mode="bilinear", align_corners=False)
    return ((v70 + 1.0) / 2.0 * (VMAX - VMIN) + VMIN)[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--guidance", type=float, default=1e-9)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prior: DiT 256/6/8 @ 64x64 (matches conf/model/dit.yaml).
    model = build_model(
        (1, 64, 64),
        None,
        DiTModelConfig(hidden=256, depth=6, num_heads=8, patch_size=4),
    ).to(dev)
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Held-out FlatVel-A map (native 70x70, m/s) -> observed seismic data.
    f = sorted(glob.glob("data/openfwi/FlatVel_A/model/*.npy"))[-1]
    v_true = (
        torch.from_numpy(np.ascontiguousarray(np.load(f, mmap_mode="r")[-1, 0]))
        .float()
        .to(dev)
    )
    d_obs = simulate(v_true).detach()

    # Bridge: prior sample (B,1,64,64) in [-1,1] -> physical seismic data.
    def forward_fn(v_norm: torch.Tensor) -> torch.Tensor:
        v_mps = to_mps70(v_norm)
        return torch.stack([simulate(v_mps[b]) for b in range(v_mps.shape[0])])

    b = args.n_samples
    x0 = torch.randn(b, 1, 64, 64, device=dev)
    t_cond = torch.zeros(b, device=dev)

    def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        tb = torch.full((x.shape[0],), t, device=dev)
        return model.v(tb, tb, x, t_cond, x0)

    guided = guided_sample(
        velocity_fn,
        x0,
        forward_fn,
        d_obs,
        sampler_steps=args.steps,
        guidance_strength=args.guidance,
        normalize_grad=True,
    )
    unguided = guided_sample(
        velocity_fn,
        x0,
        forward_fn,
        d_obs,
        sampler_steps=args.steps,
        guidance_strength=0.0,
    )

    vg = to_mps70(guided)
    mae = (vg - v_true).abs().mean(dim=(1, 2))
    rmse = ((vg - v_true) ** 2).mean(dim=(1, 2)).sqrt()
    dm_g = ((forward_fn(guided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((forward_fn(unguided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    best = int(mae.argmin())

    print(f"guidance={args.guidance:g}  steps={args.steps}  n={b}")
    print(
        f"  MAE (m/s):  {[round(x) for x in mae.tolist()]}  best={round(float(mae[best]))}"
    )
    print(f"  RMSE (m/s): {[round(x) for x in rmse.tolist()]}")
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  unguided={float(dm_u.mean()):.3e}  ratio={float(dm_g.mean() / dm_u.mean()):.3f}"
    )

    out = Path(__file__).parent / "poc_result.png"
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
