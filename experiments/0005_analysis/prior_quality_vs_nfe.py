"""Quality-vs-NFE analysis for the trained FlatVel_A priors (post-training, standalone).

For each (model, sampler, NFE) point, draw 512 prior samples and score them with the
energy distance against the standard seed-0 held-out FlatVel_A split. Four curves:

  * ``0002 flow map — few-step``: ``sample_few_step`` with ``n_steps = NFE`` (one model
    eval per jump; the off-diagonal path).
  * ``0002 flow map — ODE``: the same checkpoint through the Euler ODE sampler
    (``sampler_steps = NFE``, diagonal ``v(t,t)`` only). The gap between these two curves
    isolates the value of the trained off-diagonal.
  * ``0001 flow matching — ODE``: the teacher-class flow-matching prior.
  * ``0003 diffusion — DDPM``: the UNet diffusion prior through the reverse DDPM chain
    (``num_steps = NFE``).

Fairness: every point is scored against the same 512 real reference maps (seed-0 draw from
the val split, matching the historical per-family evaluation); each point is repeated over
several noise seeds for error bars; a real-vs-real noise floor (energy distance between two
disjoint 512-map halves of the val split) anchors the y-axis.

    uv run python experiments/0005_analysis/prior_quality_vs_nfe.py

Outputs quality_vs_nfe.png + results.md under runs/0005_analysis/prior_quality_vs_nfe_<ts>/
and logs the curves, table and figure to wandb (project
physics-informed-flow-map-0005_analysis).
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from diffusers import DDPMScheduler
from torch import Tensor
from torch.utils.data import Dataset

from physics_informed_flow_map.baselines import build_denoiser
from physics_informed_flow_map.baselines.diffusion_sample import ddpm_sample
from physics_informed_flow_map.experiment import start_run
from physics_informed_flow_map.flow_matching.datasets import OpenFWIDatasetConfig
from physics_informed_flow_map.flow_matching.family_eval import (
    N_ENERGY_REAL,
    energy_distance,
)
from physics_informed_flow_map.flow_matching.models import DiTModelConfig, build_model
from physics_informed_flow_map.flow_matching.sample import sample, sample_few_step

EXPERIMENT = "0005_analysis"
ANALYSIS = "prior_quality_vs_nfe"
RUNS_ROOT = Path("/workspace/runs")
FAMILY = "FlatVel_A"
SHAPE = (1, 64, 64)

# All three priors were trained on the same seed-0 global split of FlatVel_A at 64x64
# with the same DiT shape (the diffusion baseline swaps in a UNet denoiser).
DIT = DiTModelConfig(hidden=256, depth=6, num_heads=8, patch_size=4)
CKPT_FLOW_MAP = (
    RUNS_ROOT
    / "0002_flow_map/openfwi_mf_2026-07-07T10-49-43Z/checkpoints/step_19_ema.pt"
)
CKPT_FM_PRIOR = (
    RUNS_ROOT
    / "0001_flow_matching/openfwi_2026-06-29T09-46-00Z/checkpoints/step_99_ema.pt"
)
CKPT_DIFFUSION = (
    RUNS_ROOT / "0003_diffusion/openfwi_2026-07-05T11-52-01Z/checkpoints/step_99_ema.pt"
)
NUM_TRAIN_TIMESTEPS = 1000


@dataclass
class Curve:
    key: str
    label: str
    ckpt: Path
    nfes: list[int]
    # sampler(model_or_denoiser, n_fe, seed, batch_size, device) -> [512, *SHAPE] on device
    build: Callable[[torch.device], object]
    generate: Callable[[object, int, int, int, int, torch.device], Tensor]


def _load_flow(ckpt: Path, device: torch.device) -> object:
    net = build_model(SHAPE, None, DIT).to(device)
    net.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    net.eval()
    return net


def _load_diffusion(ckpt: Path, device: torch.device) -> tuple[object, DDPMScheduler]:
    denoiser = build_denoiser("unet", sample_size=SHAPE[1], channels=SHAPE[0]).to(
        device
    )
    denoiser.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["model"]
    )
    denoiser.eval()
    scheduler = DDPMScheduler(  # type: ignore[no-untyped-call]  # diffusers is untyped
        num_train_timesteps=NUM_TRAIN_TIMESTEPS
    )
    return denoiser, scheduler


def _seeded_noise(n: int, seed: int, device: torch.device) -> Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(n, *SHAPE, device=device, generator=g)


def _gen_fewstep(
    model: object, nfe: int, seed: int, n: int, batch_size: int, device: torch.device
) -> Tensor:
    noise = _seeded_noise(n, seed, device)
    outs = [
        sample_few_step(model, b.shape[0], SHAPE, n_steps=nfe, device=device, x_noise=b)
        for b in torch.split(noise, batch_size)
    ]
    return torch.cat(outs)


def _gen_ode(
    model: object, nfe: int, seed: int, n: int, batch_size: int, device: torch.device
) -> Tensor:
    noise = _seeded_noise(n, seed, device)
    outs = [
        sample(model, b.shape[0], SHAPE, sampler_steps=nfe, device=device, x_noise=b)
        for b in torch.split(noise, batch_size)
    ]
    return torch.cat(outs)


def _gen_ddpm(
    bundle: object, nfe: int, seed: int, n: int, batch_size: int, device: torch.device
) -> Tensor:
    denoiser, scheduler = cast("tuple[torch.nn.Module, DDPMScheduler]", bundle)
    g = torch.Generator(device=device).manual_seed(seed)
    outs = []
    done = 0
    while done < n:
        bs = min(batch_size, n - done)
        outs.append(
            ddpm_sample(
                denoiser,
                scheduler,
                SHAPE,
                n_samples=bs,
                num_steps=nfe,
                device=device,
                generator=g,
            )
        )
        done += bs
    return torch.cat(outs)


CURVES = [
    Curve(
        key="flow_map_fewstep",
        label="0002 flow map — few-step",
        ckpt=CKPT_FLOW_MAP,
        nfes=[1, 2, 4, 8, 16, 32],
        build=lambda dev: _load_flow(CKPT_FLOW_MAP, dev),
        generate=_gen_fewstep,
    ),
    Curve(
        key="flow_map_ode",
        label="0002 flow map — ODE",
        ckpt=CKPT_FLOW_MAP,
        nfes=[8, 16, 32, 64, 128, 200],
        build=lambda dev: _load_flow(CKPT_FLOW_MAP, dev),
        generate=_gen_ode,
    ),
    Curve(
        key="fm_prior_ode",
        label="0001 flow matching — ODE",
        ckpt=CKPT_FM_PRIOR,
        nfes=[2, 4, 8, 16, 32, 64, 128, 200],
        build=lambda dev: _load_flow(CKPT_FM_PRIOR, dev),
        generate=_gen_ode,
    ),
    Curve(
        key="diffusion_ddpm",
        label="0003 diffusion — DDPM (UNet)",
        ckpt=CKPT_DIFFUSION,
        # 250/500, not 256/512: DDPMScheduler's default "leading" spacing uses
        # step_ratio = 1000 // num_steps, so non-divisor counts truncate the chain
        # (256 -> t_max 765, 512 -> t_max 511) while we start from pure noise;
        # divisors keep t_max ~ 999 at every point.
        nfes=[2, 4, 8, 16, 32, 64, 128, 250, 500, 1000],
        build=lambda dev: _load_diffusion(CKPT_DIFFUSION, dev),
        generate=_gen_ddpm,
    ),
]


def noise_floor(
    val_ds: Dataset, n: int, seeds: list[int], device: torch.device
) -> list[float]:
    """Energy distance between two disjoint n-map halves of the val split, per seed."""
    n_val = len(val_ds)  # type: ignore[arg-type]  # map-style dataset
    if 2 * n > n_val:
        raise SystemExit(
            f"val split has {n_val} maps, need {2 * n} for the noise floor"
        )
    floors = []
    for seed in seeds:
        # Offset the seed so the halves are not correlated with the seed-0 reference draw.
        perm = torch.randperm(
            n_val, generator=torch.Generator().manual_seed(1000 + seed)
        )[: 2 * n]
        a = torch.stack([val_ds[int(i)][0] for i in perm[:n]]).to(device)
        b = torch.stack([val_ds[int(i)][0] for i in perm[n:]]).to(device)
        floors.append(energy_distance(a, b))
    return floors


def crossover_nfe(target: float, nfes: list[int], means: list[float]) -> float | None:
    """Smallest (log-interpolated) NFE at which a descending curve reaches ``target``."""
    for (n0, e0), (n1, e1) in zip(zip(nfes, means), zip(nfes[1:], means[1:])):
        if e0 > target >= e1:
            if e0 == e1:
                return float(n1)
            frac = (e0 - target) / (e0 - e1)
            return float(math.exp(math.log(n0) + frac * (math.log(n1) - math.log(n0))))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="plumbing check: 2 NFE points per curve, 32 samples, 1 seed",
    )
    args = ap.parse_args()
    if args.smoke:
        args.n_samples, args.n_seeds = 32, 1
        for c in CURVES:
            c.nfes = c.nfes[:2]
    seeds = list(range(args.n_seeds))

    for c in CURVES:
        if not c.ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {c.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = RUNS_ROOT / EXPERIMENT / f"{ANALYSIS}_{ts}"

    # The standard seed-0 held-out split all three priors were trained against.
    dataset = OpenFWIDatasetConfig(
        families=[FAMILY], resolution=SHAPE[1], val_fraction=0.1, split_scheme="global"
    )
    val_ds = dataset.build_val()

    config = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "seeds": seeds,
        "family": FAMILY,
        "dataset_fingerprint": dataset.fingerprint(),
        "curves": {
            c.key: {"ckpt": str(c.ckpt), "nfes": c.nfes, "label": c.label}
            for c in CURVES
        },
    }
    run = start_run(EXPERIMENT, run_dir, config, name=f"{ANALYSIS}_{ts}")

    # The seed-0 reference draw every point is scored against (the historical
    # per-family-eval reference: up to N_ENERGY_REAL maps, seed-0 permutation).
    n_val = len(val_ds)  # type: ignore[arg-type]  # map-style dataset
    ref_idx = torch.randperm(n_val, generator=torch.Generator().manual_seed(0))[
        :N_ENERGY_REAL
    ]
    real_ref = torch.stack([val_ds[int(i)][0] for i in ref_idx]).to(device)

    floors = noise_floor(val_ds, args.n_samples, seeds, device)
    floor_mean = sum(floors) / len(floors)
    print(f"real-vs-real noise floor: {floor_mean:.4f} (per-seed {floors})")

    # rows: (curve_key, nfe, seed, energy, gen_seconds)
    rows: list[tuple[str, int, int, float, float]] = []
    results: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for curve in CURVES:
        model = curve.build(device)
        means, stds = [], []
        for nfe in curve.nfes:
            energies = []
            for seed in seeds:
                t0 = time.perf_counter()
                pool = curve.generate(
                    model, nfe, seed, args.n_samples, args.batch_size, device
                )
                gen_s = time.perf_counter() - t0
                ed = energy_distance(pool, real_ref)
                energies.append(ed)
                rows.append((curve.key, nfe, seed, ed, gen_s))
            m = sum(energies) / len(energies)
            sd = (sum((e - m) ** 2 for e in energies) / len(energies)) ** 0.5
            means.append(m)
            stds.append(sd)
            print(f"{curve.key:18s} NFE={nfe:4d}  energy={m:.4f} ± {sd:.4f}")
        results[curve.key] = (curve.nfes, means, stds)
        del model
        torch.cuda.empty_cache()

    # Headline crossover: what DDPM step count matches the flow map at 4 NFE?
    few_nfes, few_means, _ = results["flow_map_fewstep"]
    fm4 = few_means[few_nfes.index(4)] if 4 in few_nfes else None
    diff_nfes, diff_means, _ = results["diffusion_ddpm"]
    cross = crossover_nfe(fm4, diff_nfes, diff_means) if fm4 is not None else None

    # Figure: energy distance vs NFE, log x, noise-floor band.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve in CURVES:
        nfes, means, stds = results[curve.key]
        ax.errorbar(
            nfes, means, yerr=stds, marker="o", ms=4, capsize=3, label=curve.label
        )
    ax.axhline(floor_mean, color="gray", ls="--", lw=1, label="real-vs-real floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NFE (model evaluations per sample)")
    ax.set_ylabel(f"energy distance vs held-out {FAMILY}")
    title = f"Sample quality vs NFE ({FAMILY}, {args.n_samples} samples, {len(seeds)} seeds)"
    if cross is not None:
        title += f"\nflow map @ 4 NFE ≈ DDPM @ {cross:.0f} NFE"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = run_dir / "quality_vs_nfe.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # results.md
    lines = [
        f"# Quality vs NFE — {FAMILY}, {args.n_samples} samples/point, seeds {seeds}",
        "",
        "Energy distance of the generated pool vs the standard seed-0 held-out "
        f"{FAMILY} split (512 reference maps, shared across every point). Lower is "
        "better; the floor is real-vs-real between two disjoint 512-map val halves.",
        "",
        f"Real-vs-real noise floor: **{floor_mean:.4f}** "
        f"(± {(sum((f - floor_mean) ** 2 for f in floors) / len(floors)) ** 0.5:.4f})",
        "",
        "| model | NFE | energy (mean) | energy (std) | gen time / 512 (s) |",
        "|---|---|---|---|---|",
    ]
    for curve in CURVES:
        nfes, means, stds = results[curve.key]
        for nfe, m, sd in zip(nfes, means, stds):
            gen_ss = [r[4] for r in rows if r[0] == curve.key and r[1] == nfe]
            lines.append(
                f"| {curve.label} | {nfe} | {m:.4f} | {sd:.4f} "
                f"| {sum(gen_ss) / len(gen_ss):.1f} |"
            )
    if cross is not None:
        lines += [
            "",
            f"**Crossover:** flow map @ 4 NFE (energy {fm4:.4f}) ≈ DDPM @ "
            f"~{cross:.0f} NFE (log-interpolated).",
        ]
    elif fm4 is not None:
        lines += [
            "",
            f"**Crossover:** DDPM never reaches flow map @ 4 NFE ({fm4:.4f}).",
        ]
    (run_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {run_dir / 'results.md'}")

    # wandb: raw points table, one line-series panel, the figure, summary scalars.
    table = wandb.Table(
        columns=["model", "nfe", "seed", "energy", "gen_s"],
        data=[list(r) for r in rows],
    )
    run.run.log({"points": table})
    run.run.log(
        {
            "quality_vs_nfe_curves": wandb.plot.line_series(
                xs=[results[c.key][0] for c in CURVES],
                ys=[results[c.key][1] for c in CURVES],
                keys=[c.label for c in CURVES],
                title="energy distance vs NFE (means)",
                xname="NFE",
            )
        }
    )
    run.run.log({"quality_vs_nfe": wandb.Image(str(png))})
    summary: dict[str, float] = {"noise_floor": floor_mean}
    for curve in CURVES:
        nfes, means, _ = results[curve.key]
        for nfe, m in zip(nfes, means):
            summary[f"energy/{curve.key}/nfe_{nfe}"] = m
    if cross is not None:
        summary["crossover_ddpm_nfe_vs_flowmap4"] = cross
    run.finish(**summary)


if __name__ == "__main__":
    main()
