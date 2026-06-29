# physics-informed-flow-map



'uv run wandb login'

## FWI set up 

One-time setup (inside RunPod, attached to your Network Volume):
```
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ashynf/OpenFWI",
    repo_type="dataset",
    local_dir="/workspace/data/openfwi",
    local_dir_use_symlinks=False,
    allow_patterns=["*FlatVel*", "*FlatFault*"], 
)
EOF
```
This one definitly works:
uv run hf download ashynf/OpenFWI --repo-type dataset --include "*CurveVel*" --local-dir /workspace/data/openfwi

uv run huggingface-cli download ashynf/OpenFWI --repo-type dataset --include "*FlatVel*" "*FlatFault*"  --local-dir /workspace/data/openfwi



No train/val split in the raw files: the dataset does an 80/20 random split automatically.

Native resolution is 70×70: the config upsamples to 256 before encoding through the VAE (required for the DiT architecture). You can lower this to save compute.


Training: 
```
cd /workspace/physics-informed-flow-map/packages/mfm-meta-flow-map-main
python scripts/train_custom.py \
    --config-name config_train_openfwi.yaml \
    data_dir=/workspace/data/openfwi \
    trainer.batch_size=16
```
