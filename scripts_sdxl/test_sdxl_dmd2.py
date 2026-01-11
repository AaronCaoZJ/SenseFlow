from diffusers import UNet2DConditionModel, LCMScheduler, StableDiffusionXLPipeline
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import time

class PromptDataset(Dataset):
    """Dataset for loading prompts from a text file."""
    def __init__(self, path, start_idx, num_prompts):
        with open(path, 'r') as f:
            prompts = [line.strip() for line in f.readlines()]
        self.prompts = prompts[start_idx:start_idx + num_prompts]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def seed_everything(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Inference script for SenseFlow SDXL DMD2 model")
    parser.add_argument("--sdxl_ckpt", type=str, required=True,
                        help="Path to Stable Diffusion XL base checkpoint directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained DMD2 checkpoint")
    parser.add_argument("--unet_config", type=str,
                        help="Path to UNet config file (default: <sdxl_ckpt>/unet/config.json)")
    parser.add_argument("--prompts_file", type=str,
                        default="senseflow_test_prompts.txt",
                        help="Path to prompts text file")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index in prompts file")
    parser.add_argument("--num_prompts", type=int, default=23,
                        help="Number of prompts to process")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for generated images")
    args = parser.parse_args()

    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Set unet config path
    if args.unet_config:
        unet_config_path = args.unet_config
    else:
        unet_config_path = os.path.join(args.sdxl_ckpt, "unet", "config.json")
    
    # Load UNet config and create model
    unet = UNet2DConditionModel.from_config(unet_config_path, variant="fp32")
    
    folder_name = "senseflow_sdxl_dmd2"
    save_folder = os.path.join(args.output_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    # DMD2 checkpoint format: directly contains state_dict, not wrapped in 'model' key
    state_dict = checkpoint
    
    m, u = unet.load_state_dict(state_dict, strict=False)
    print('Missing keys:', m)
    print('Unexpected keys:', u)
    unet.eval()
    
    # Create pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl_ckpt,
        unet=unet,
        torch_dtype=torch.float32
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Load prompts
    prompts_path = os.path.join(project_root, args.prompts_file)
    dataset = PromptDataset(prompts_path, args.start_idx, args.num_prompts)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Generate images
    with torch.autocast("cuda", dtype=torch.float32):
        for idx, prompts in enumerate(dataloader):
            images = pipe(
                prompts,
                num_inference_steps=4,
                guidance_scale=0.0
            ).images
            
            for i in range(len(images)):
                image_idx = args.start_idx + idx * args.batch_size + i
                images[i].save(
                    os.path.join(save_folder, f'{image_idx:04d}.png'),
                    'PNG'
                )
