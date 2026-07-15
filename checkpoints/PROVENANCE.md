# Checkpoint provenance

`conf/prior/*.yaml` addresses each definitive prior by a stable alias, so commands survive
retraining and move between machines unchanged. The `.pt` files are gitignored — copy them here
from the source run and keep this table current. `Run.save_checkpoint` stores only
`{"model", "step"}`, so without this table an alias is an anonymous blob.

Fetch on a fresh machine:

    rsync -a pod:/workspace/physics-informed-flow-map/checkpoints/ checkpoints/

| alias | source run | step | backbone |
|---|---|---|---|
| `0001_flow_matching_openfwi.pt` | `runs/0001_flow_matching/openfwi_2026-07-07T11-19-11Z` | `step_89_ema` | 768/12/12 DiT |
| `0002_flow_map_openfwi.pt` | `runs/0002_flow_map/openfwi_mf_2026-07-08T10-23-48Z` | `step_89_ema` | 768/12/12 DiT |
| `0003_diffusion_openfwi.pt` | `runs/0003_diffusion/openfwi_2026-07-07T23-26-16Z` | `step_59_ema` | 768/12/12 DiT (eps-denoiser) |

Each source run dir holds the wandb run id in its manifest; the run dirs live on the RunPod
network volume under the repo's `runs/`.

Retraining a prior: copy the new `step_<N>_ema.pt` over the alias, update the row, and update
`model:` in the matching `conf/prior/*.yaml` if the backbone shape changed.
