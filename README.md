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

## connect runpod to other ssh server

Step 1 — On your local machine: connect to the pod with agent forwarding

ssh-add ~/.ssh/id_ed25519
ssh -A root@213.173.108.40 -p 36015

Step 2 — On the pod (still on local machine): run it in tmux. Do the next two commands one by one. 

tmux new -s xfer 

rsync -avP -e "ssh -J tunneluser@xxx.xxx.xxx.xx -p 2xxx" \
  /workspace/runs \
  /workspace/marmousi \
  xxx@localhost:destination/path

## connect to headless vscode so that the training continues even when connection is down locally. 
Step 1 -- connect to the ssh server. 
ssh xxxxx 

Step 2 run in tmux 
tmux new -s name_of_the_session

Step 3 follow the prompt and url to sign into github and grant permission. Then open the url so that the vscode UI opens in a browser like google chrome. 

Step 4 in the remote vscode, do whatever is needed for the training as normal.

Step 5 to quit the tmux session, Press Ctrl + b, then press : (colon) to open the command prompt. Type kill-session and press Enter.