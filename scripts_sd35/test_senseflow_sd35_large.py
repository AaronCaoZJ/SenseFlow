from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import time
from typing import Union, Optional, Tuple

class FlowMatchEulerX0Scheduler(FlowMatchEulerDiscreteScheduler):
    """Custom scheduler for Flow Matching with x0 prediction."""
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Custom step function:
        - Predicts x0 (original clean sample) first.
        - Adds noise to x0 to get the sample at the next step.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)  # Ensure precision

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        # 1. Compute x0 from model output (assuming model predicts noise)
        x0 = sample - sigma * model_output

        # 2. Add noise to x0 to get the sample for the next step
        noise = torch.randn_like(sample)
        prev_sample = (1 - sigma_next) * x0 + sigma_next * noise

        prev_sample = prev_sample.to(model_output.dtype)  # Convert back to original dtype
        self._step_index += 1  # Move to next step

        if not return_dict:
            return (prev_sample,)
        
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


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
    parser = argparse.ArgumentParser(description="Inference script for SenseFlow SD3.5 Large model")
    parser.add_argument("--sd35_ckpt", type=str, required=True,
                        help="Path to Stable Diffusion 3.5 Large checkpoint directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained SenseFlow checkpoint")
    parser.add_argument("--transformer_config", type=str,
                        help="Path to transformer config file (default: <sd35_ckpt>/transformer/config.json)")
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
    
    # Set transformer config path
    if args.transformer_config:
        transformer_config_path = args.transformer_config
    else:
        transformer_config_path = os.path.join(args.sd35_ckpt, "transformer", "config.json")
    
    # Load transformer config and create model
    transformer = SD3Transformer2DModel.from_config(transformer_config_path)
    
    folder_name = "senseflow_sd35_large"
    save_folder = os.path.join(args.output_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema_model' in checkpoint:
        state_dict = checkpoint['ema_model']
    else:
        state_dict = checkpoint
    
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print('Missing keys:', m)
    print('Unexpected keys:', u)
    transformer.eval().to(torch.bfloat16)
    
    # Create pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.sd35_ckpt,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    pipe.scheduler = FlowMatchEulerX0Scheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    # Load prompts
    prompts_path = os.path.join(project_root, args.prompts_file)
    dataset = PromptDataset(prompts_path, args.start_idx, args.num_prompts)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Generate images
    total_times = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, prompts in enumerate(dataloader):
            start_time = time.time()
            images = pipe(
                prompts,
                num_inference_steps=2,
                guidance_scale=0.0,
                sigmas=[1.0, 0.75]
            ).images
            cur_time = time.time() - start_time
            total_times += cur_time
            print(f'Batch {idx+1}: cur_time={cur_time:.2f}s, avg_time={total_times/(idx+1):.2f}s')
            
            for i in range(len(images)):
                image_idx = args.start_idx + idx * args.batch_size + i
                images[i].save(
                    os.path.join(save_folder, f'{image_idx:04d}.png'),
                    'PNG'
                )
