from huggingface_hub import snapshot_download

# save_dir = "models/BAGEL-7B-MoT"
# repo_id = "ByteDance-Seed/BAGEL-7B-MoT"

save_dir = "/root/highspeedstorage/model_distill/SenseFlow/ckpt/stable-diffusion-3.5-medium"
# save_dir = "/root/highspeedstorage/model_distill/SenseFlow/dataset/LAION_Aesthetics_1024"
repo_id = "stabilityai/stable-diffusion-3.5-medium"
# repo_id = "limingcv/LAION_Aesthetics_1024"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  repo_type="model",
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
#   allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

