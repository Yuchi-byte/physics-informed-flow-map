"""Invert a held-out velocity map with a trained diffusion prior — the camp-A diffusion baselines.

    uv run python experiments/0003_baselines/inversion.py ckpt=<run.py artifact> method=dps
    uv run python experiments/0003_baselines/inversion.py ckpt=<...> method=unguided
    uv run python experiments/0003_baselines/inversion.py --multirun ckpt=<...> method=dps,unguided

Loads a diffusion prior trained by ``run.py`` (do not retrain here), simulates seismic data ``d``
from a held-out (validation-split) OpenFWI map with the Deepwave forward operator, then runs the
chosen inference-time scheme to recover the velocity. The ``method`` group picks ``dps`` (canonical
Diffusion Posterior Sampling) or ``unguided`` (prior sample, no physics — the control).

Diffusion counterpart to ``0001_flow_matching/inversion.py`` (flow prior + tilting): same held-out
map, forward operator, metrics, and figure (shared via ``inversion.single_target``), so the two
camps compare head-to-head. This entry only supplies how to invert (diffusion prior + ``dps_sample``).

Caveat (shared with the flow inversion): ``d_obs`` comes from the same noiseless forward operator
used inside the guidance term (an "inverse crime"), so recovery is optimistic.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
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
from physics_informed_flow_map.inversion.bridge import seismic_forward
from physics_informed_flow_map.inversion.single_target import invert_and_report

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

    def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
        return dps_sample(
            denoiser,
            scheduler,
            cfg.dataset.shape,
            seismic_forward,
            d_obs,
            n_samples=cfg.n_samples,
            num_steps=cfg.diffusion.num_sample_steps,
            guidance_strength=guidance_strength,
            device=dev,
            normalize_grad=cfg.method.normalize_grad,
        )

    out = run.ckpt_dir.parent / "inversion.png"
    summary, caption = invert_and_report(
        invert,
        dataset_cfg=cfg.dataset,
        target_index=cfg.target_index,
        method_name=cfg.method.name,
        guidance=cfg.method.guidance_strength,
        steps=cfg.diffusion.num_sample_steps,
        device=dev,
        out_png=out,
    )
    run.log_image("inversion", out, caption=caption)
    run.finish(**summary)


if __name__ == "__main__":
    main()
