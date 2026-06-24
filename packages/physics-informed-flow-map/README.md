# physics-informed-flow-map

The new method under development: physics-informed constraint losses (PIDM-style)
applied to the Meta Flow Map (MFM) flow-map framework.

## Layout

```
src/physics_informed_flow_map/
├── __init__.py
└── experiment/        # shared experiment harness (typed configs + manifest-pinned runs)
    ├── config.py      # Config base: pydantic schema + OmegaConf override merge
    └── run.py         # start_run / Run.log / Run.finish
```

The harness is consumed by the frameworks under the repo-root `experiments/`
directory (see `experiments/README.md`).
