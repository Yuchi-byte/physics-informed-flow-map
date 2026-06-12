# Below are some notes made on how the environment is set from scratch. 

# uv sync. This creates a venv. 

# uv run script.py (if this doesn't work, sometimes it's because we need torchrun to do the DDP). make sure the file path is precise. 
# to run a single python line, do: uv run python -c "print('hello world')" 
# cannot do: python3 script.py because it runs the system python andthe environment is not activated. This can work if you activate the enviroment. But let's stick to uv and dw about python shell. 

# if there are unfound modules, do uv add package_name. This will update uv.lock and pyproject.toml automatically. 

# download mfm-xl2.pt 
# hf download adh1s/mfm --include "mfm-xl2.pt" --local-dir ckpts

# download the models before running the sample_steered.py script. This will ensure that the models are cached and ready to use when the script is run. So the script doesn't take too long to run. 
from imscore.hps.model import HPSv2
from imscore.imreward.model import ImageReward
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL

print('VAE...')
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
print('HPSv2...')
HPSv2.from_pretrained('RE-N-Y/hpsv21')
print('ImageReward...')
ImageReward.from_pretrained('RE-N-Y/ImageReward')
print('CLIP...')
CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
print('PickScore...')
AutoModel.from_pretrained('yuvalkirstain/PickScore_v1')
AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
print('All done.')

# To run the sample_steered.py script, use the following command. Note to use uv run, torchrun, put the file path, '\' at the end of the line, and ++. 
uv run torchrun --nnodes=1 --nproc_per_node=1 packages/mfm-meta-flow-map-main/scripts/sample_steered.py per_proc_batch_size=4 
# per_proc_batch_size=4 instread of 32 to make the effective batch down so we don't encounter GPUT out of memory issues. 



# when running the sample_steered.py RTX A4500 1x with VRAM 20GB, the execution encounters OOM error every time. Next time, try with a bigger VRAM. 