"""Invert a held-out velocity map with a trained diffusion prior — the camp-A diffusion baselines.

    uv run python experiments/0003_baselines/inversion.py ckpt=<run.py artifact> method=dps
    uv run python experiments/0003_baselines/inversion.py ckpt=<...> method=unguided
    uv run python experiments/0003_baselines/inversion.py --multirun ckpt=<...> method=dps,unguided

Loads a diffusion prior trained by ``run.py`` (do not retrain here), simulates seismic data
``d`` from a held-out (validation-split) OpenFWI map with the Deepwave forward operator, then
runs the chosen inference-time scheme to recover the velocity. The ``method`` config group picks
which published baseline to reproduce — ``dps`` (canonical Diffusion Posterior Sampling) or
``unguided`` (prior sample, no physics; the control that shows the wave equation is doing the work).

This is the diffusion counterpart to ``0002_fwi_tilting/poc.py`` (flow-map prior + tilting): same
held-out map, same forward operator, same metrics, so the two camps compare head-to-head. Reports
per-sample MAE/RMSE (m/s) vs the true map plus the ground-truth-free pick (lowest data misfit), the
data-misfit reduction vs an unguided sample, and writes a ``true | v_hat | error`` figure.

Caveat (shared with the flow PoC): ``d_obs`` comes from the *same* noiseless forward operator used
inside the guidance term (an "inverse crime"), so recovery is optimistic relative to field FWI.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field, model_validator

from physics_informed_flow_map.baselines import build_denoiser, dps_sample
from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    OpenFWIDatasetConfig,
)
from physics_informed_flow_map.flow_matching.openfwi import VMAX, VMIN
from physics_informed_flow_map.physics.forward import simulate

EXPERIMENT = "0003_baselines"


class MethodConfig(Config):
    name: str = "dps"  # dps | unguided (RED-DiffEq etc. slot in here later)
    guidance_strength: float = 0.5
    normalize_grad: bool = True


class ModelConfig(Config):
    kind: str = "unet"


class DiffusionConfig(Config):
    num_train_timesteps: int = Field(1000, gt=0)
    num_sample_steps: int = Field(200, gt=0)  # reverse / guidance steps


class InversionConfig(Config):
    seed: int = 0
    ckpt: str = ""  # trained diffusion prior (run.py artifact); empty = untrained, plumbing only
    target_index: int = Field(0, ge=0)  # index into the held-out validation split
    n_samples: int = Field(4, gt=0)
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = OpenFWIDatasetConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    method: MethodConfig = MethodConfig()

    @model_validator(mode="after")
    def _check_openfwi(self) -> "InversionConfig":
        if not isinstance(self.dataset, OpenFWIDatasetConfig):
            raise ValueError(
                "inversion targets OpenFWI velocity maps (dataset=openfwi)"
            )
        if self.method.name not in {"dps", "unguided"}:
            raise ValueError(f"unknown method '{self.method.name}' (dps | unguided)")
        return self


InversionConfig.model_rebuild()


def to_mps70(v_norm: torch.Tensor) -> torch.Tensor:
    """(B,1,64,64) in [-1,1] -> (B,70,70) velocity in m/s (clamped to the trained range)."""
    v70 = F.interpolate(v_norm, size=70, mode="bilinear", align_corners=False).clamp(
        -1.0, 1.0
    )
    return ((v70 + 1.0) / 2.0 * (VMAX - VMIN) + VMIN)[:, 0]


def held_out_native(
    cfg: OpenFWIDatasetConfig, target_index: int
) -> tuple[int, np.ndarray]:
    """``(global_index, native 70x70 m/s map)`` for the ``target_index``-th validation map.

    Drawn from the same seed-0 validation split the prior held out of training (no leakage).
    """
    full, _, val_idx = cfg._split()
    gidx = val_idx[target_index]
    path, row = full.index[gidx]
    native = np.ascontiguousarray(np.load(path, mmap_mode="r")[row, 0])  # (70,70) m/s
    return gidx, native


@hydra.main(version_base=None, config_path="conf", config_name="inversion")
def main(dcfg: DictConfig) -> None:
    cfg = InversionConfig.from_dictconfig(dcfg)
    assert isinstance(cfg, InversionConfig)
    assert isinstance(cfg.dataset, OpenFWIDatasetConfig)  # narrowed by the validator

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run = start_run(EXPERIMENT, run_dir, cfg.dump(), name=f"invert-{cfg.method.name}")

    # Prior: the diffusers UNet denoiser trained by run.py. Untrained when ckpt is empty.
    channels, size, _ = cfg.dataset.shape
    denoiser = build_denoiser(cfg.model.kind, sample_size=size, channels=channels).to(
        dev
    )
    if cfg.ckpt:
        if not Path(cfg.ckpt).is_file():
            raise SystemExit(
                f"ckpt not found: {cfg.ckpt}\nTrain a prior first (uv run python "
                "experiments/0003_baselines/run.py experiment=openfwi) and point ckpt= at its "
                "best/final checkpoint (runs/0003_baselines/<ts>/checkpoints/step_<N>.pt)."
            )
        state = torch.load(cfg.ckpt, map_location=dev, weights_only=False)
        denoiser.load_state_dict(state["model"])
    else:
        print("[inversion] no ckpt= given: using an UNTRAINED denoiser (plumbing only)")
    denoiser.eval()
    scheduler = DDPMScheduler(num_train_timesteps=cfg.diffusion.num_train_timesteps)  # type: ignore[no-untyped-call]

    # Held-out (val-split) OpenFWI map (native 70x70, m/s) -> observed seismic data.
    gidx, native = held_out_native(cfg.dataset, cfg.target_index)
    v_true = torch.from_numpy(native).float().to(dev)
    d_obs = simulate(v_true).detach()
    print(f"target: val map global index {gidx} (native {tuple(v_true.shape)})")

    # Bridge: prior sample (B,1,64,64) in [-1,1] -> physical seismic data.
    def forward_fn(v_norm: torch.Tensor) -> torch.Tensor:
        v_mps = to_mps70(v_norm)
        return torch.stack([simulate(v_mps[b]) for b in range(v_mps.shape[0])])

    shape = cfg.dataset.shape
    n = cfg.n_samples
    steps = cfg.diffusion.num_sample_steps
    gs = cfg.method.guidance_strength if cfg.method.name == "dps" else 0.0

    def invert(guidance_strength: float) -> torch.Tensor:
        return dps_sample(
            denoiser,
            scheduler,
            shape,
            forward_fn,
            d_obs,
            n_samples=n,
            num_steps=steps,
            guidance_strength=guidance_strength,
            device=dev,
            normalize_grad=cfg.method.normalize_grad,
        )

    guided = invert(gs)
    # Unguided control for the misfit-reduction ratio (reuse when the method is itself unguided).
    unguided = guided if gs == 0.0 else invert(0.0)

    vg = to_mps70(guided)
    mae = (vg - v_true).abs().mean(dim=(1, 2))
    rmse = ((vg - v_true) ** 2).mean(dim=(1, 2)).sqrt()
    dm_g = ((forward_fn(guided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    dm_u = ((forward_fn(unguided) - d_obs) ** 2).sum(dim=(1, 2, 3))
    oracle = int(mae.argmin())  # needs ground truth — not available in a real inversion
    reported = int(dm_g.argmin())  # ground-truth-free pick: lowest data misfit

    ratio = float(dm_g.mean() / dm_u.mean())
    print(f"method={cfg.method.name}  guidance={gs:g}  steps={steps}  n={n}")
    print(f"  MAE (m/s):  {[round(x) for x in mae.tolist()]}")
    print(
        f"  MAE mean={round(float(mae.mean()))}  median={round(float(mae.median()))}  worst={round(float(mae.max()))}"
    )
    print(
        f"  oracle (needs GT) best MAE = {round(float(mae[oracle]))}  [sample {oracle}]"
    )
    print(
        f"  reported (min data misfit) MAE = {round(float(mae[reported]))}  [sample {reported}]"
    )
    print(
        f"  data misfit  guided={float(dm_g.mean()):.3e}  unguided={float(dm_u.mean()):.3e}  ratio={ratio:.3f}"
    )

    out = run.ckpt_dir.parent / "inversion.png"
    vt = v_true.cpu().numpy()
    vh = vg[reported].detach().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(vt, cmap="viridis")
    ax[0].set_title("true v")
    ax[0].axis("off")
    ax[1].imshow(vh, cmap="viridis", vmin=vt.min(), vmax=vt.max())
    ax[1].set_title(f"v_hat, min-misfit (MAE {round(float(mae[reported]))} m/s)")
    ax[1].axis("off")
    im = ax[2].imshow(vh - vt, cmap="RdBu", vmin=-500, vmax=500)
    ax[2].set_title("error")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    run.log_image("inversion", out, caption=f"{cfg.method.name} · val map {gidx}")

    run.finish(
        **{
            "inv/mae_mean": float(mae.mean()),
            "inv/mae_reported": float(mae[reported]),
            "inv/rmse_mean": float(rmse.mean()),
            "inv/misfit_ratio": ratio,
            "inv/target_index": gidx,
        }
    )


if __name__ == "__main__":
    main()
