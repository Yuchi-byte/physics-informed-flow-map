"""Invert a held-out velocity map with a trained flow prior — camp-A flow tilting.

    uv run python experiments/0001_flow_matching/inversion.py ckpt=<run.py EMA ckpt> method=flow_tilt
    uv run python experiments/0001_flow_matching/inversion.py ckpt=<...> method=unguided
    uv run python experiments/0001_flow_matching/inversion.py --multirun ckpt=<...> method=flow_tilt,unguided

Loads a flow-matching prior trained by ``run.py`` (do not retrain here), simulates seismic data
``d`` from a held-out (validation-split) OpenFWI map with the Deepwave forward operator, then
DPS-style tilts the prior toward ``d`` to recover the velocity. The ``method`` group picks
``flow_tilt`` (wave-equation guidance) or ``unguided`` (prior sample, no physics — the control
that shows the wave equation does the work). Diffusion counterpart: ``0003_baselines/inversion.py``.

Scoring + figure are shared with the other inversions via ``inversion.single_target``; this entry
only supplies how to invert (flow prior + ``guided_sample``).

Caveat: ``d_obs`` comes from the same noiseless forward operator used inside the guidance term
(an "inverse crime"), so recovery is optimistic.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pydantic import Field, model_validator

from physics_informed_flow_map.experiment import Config, start_run
from physics_informed_flow_map.flow_matching.datasets import (
    DatasetConfig,
    OpenFWIDatasetConfig,
)
from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.inversion.bridge import seismic_forward
from physics_informed_flow_map.inversion.single_target import invert_and_report
from physics_informed_flow_map.physics.tilt import guided_sample

EXPERIMENT = "0001_flow_matching"


class MethodConfig(Config):
    name: str = "flow_tilt"  # flow_tilt | unguided
    guidance_strength: float = 1.0
    normalize_grad: bool = True


class ModelConfig(Config):
    hidden: int = 256
    depth: int = 6
    num_heads: int = 8
    patch_size: int = 4


class InversionConfig(Config):
    seed: int = 0
    ckpt: str = (
        ""  # trained flow prior (a run.py checkpoint); empty = untrained, plumbing only
    )
    target_index: int = Field(0, ge=0)  # index into the seed-0 validation split
    n_samples: int = Field(4, gt=0)
    steps: int = Field(100, gt=0)
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = OpenFWIDatasetConfig()
    method: MethodConfig = MethodConfig()

    @model_validator(mode="after")
    def _check(self) -> "InversionConfig":
        if not isinstance(self.dataset, OpenFWIDatasetConfig):
            raise ValueError(
                "inversion targets OpenFWI velocity maps (dataset=openfwi)"
            )
        if self.method.name not in {"flow_tilt", "unguided"}:
            raise ValueError(
                f"unknown method '{self.method.name}' (flow_tilt | unguided)"
            )
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

    channels, size, _ = cfg.dataset.shape
    prior = build_model(
        cfg.dataset.shape,
        None,
        DiTModelConfig(
            hidden=cfg.model.hidden,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            patch_size=cfg.model.patch_size,
        ),
    ).to(dev)
    if cfg.ckpt:
        if not Path(cfg.ckpt).is_file():
            raise SystemExit(
                f"ckpt not found: {cfg.ckpt}\nTrain a prior first (uv run python "
                "experiments/0001_flow_matching/run.py experiment=openfwi) and point ckpt= "
                "at its best-val EMA checkpoint (runs/0001_flow_matching/<ts>/checkpoints/"
                "step_<N>_ema.pt)."
            )
        prior.load_state_dict(
            torch.load(cfg.ckpt, map_location=dev, weights_only=False)["model"]
        )
    else:
        print("[inversion] no ckpt= given: using an UNTRAINED prior (plumbing only)")
    prior.eval()

    # The flow prior tilts from a fixed noise context x0 (reused across guided/unguided passes).
    n = cfg.n_samples
    x0 = torch.randn(n, channels, size, size, device=dev)
    t_cond = torch.zeros(n, device=dev)

    def velocity_fn(x: torch.Tensor, t: float) -> torch.Tensor:
        tb = torch.full((x.shape[0],), t, device=dev)
        return prior.v(tb, tb, x, t_cond, x0)

    def invert(d_obs: torch.Tensor, guidance_strength: float) -> torch.Tensor:
        return guided_sample(
            velocity_fn,
            x0,
            seismic_forward,
            d_obs,
            sampler_steps=cfg.steps,
            guidance_strength=guidance_strength,
            normalize_grad=cfg.method.normalize_grad,
        )

    out = run.ckpt_dir.parent / "inversion.png"
    summary, caption = invert_and_report(
        invert,
        dataset_cfg=cfg.dataset,
        target_index=cfg.target_index,
        method_name=cfg.method.name,
        guidance=cfg.method.guidance_strength,
        steps=cfg.steps,
        device=dev,
        out_png=out,
    )
    run.log_image("inversion", out, caption=caption)
    run.finish(**summary)


if __name__ == "__main__":
    main()
