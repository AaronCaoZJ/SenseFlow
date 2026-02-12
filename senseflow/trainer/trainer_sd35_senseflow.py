import os
import os.path as osp
import sys
sys.path.append("/root/highspeedstorage/model_distill/SenseFlow/latent-diffusion")
import yaml
import random
import numpy as np
import functools
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, default_collate
import torch.distributed as dist
try:
    import wandb
except ImportError:
    wandb = None
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from senseflow.utils import instantiate_from_config
from senseflow.trainer.senseflow_utils import extract_into_tensor
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}
from PIL import Image
import torch.nn as nn
from senseflow.data.senseflow_dataset import SDImageDatasetLMDBwoTokenizer, LaionText2ImageDataset
from senseflow.data.senseflow_dataset import cycle
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from senseflow.models.vfmgan import ProjectedDiscriminatorPlus, GANLoss
from senseflow.models.clip import CLIP

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    # prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = text_encoder(text_input_ids.cuda())[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

@torch.no_grad()
def encode_prompt(
    prompt: str,
    text_encoders,
    tokenizers,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds.cuda(), pooled_prompt_embeds.cuda()
        
def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
    ):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, revision=revision, token=True
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "CLIPTextModelWithProjection":
            from transformers import CLIPTextModelWithProjection

            return CLIPTextModelWithProjection
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")
        
def predict_noise(dit, noisy_latents, text_embeddings, uncond_embedding, timesteps, 
    guidance_scale=1.0, pooled_prompt_embeds=None, uncond_pooled_prompt_embeds=None, decoupled_dmd=False
):
    CFG_GUIDANCE = guidance_scale > 1
    DECOUPLED_DMD = decoupled_dmd

    if CFG_GUIDANCE:
        model_input = torch.cat([noisy_latents] * 2) 
        embeddings = torch.cat([uncond_embedding, text_embeddings]) 
        timesteps = torch.cat([timesteps] * 2) 
        pooled_embeds = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds])

        noise_pred = dit(model_input, timestep=timesteps, encoder_hidden_states=embeddings, pooled_projections=pooled_embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if DECOUPLED_DMD:
            return noise_pred_text, noise_pred_uncond
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    else:
        model_input = noisy_latents 
        embeddings = text_embeddings
        timesteps = timesteps
        pooled_embeds = pooled_prompt_embeds
        noise_pred = dit(model_input, timestep=timesteps, encoder_hidden_states=embeddings, pooled_projections=pooled_embeds).sample

    return noise_pred

class Trainer(object):

    def __init__(self, config_path, save_path):
        self.config = self.load_config(config_path)
        self.save_path = osp.abspath(save_path)

    def load_config(self, config_path):
        assert osp.exists(config_path), "{} does not exist !".format(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
            f.close()
        return config

    def setup(self):
        self.init_seed()
        self.init_distributed()
        self.init_logger()
        self.init_saver()
        self.init_resume()
        self.build_model()
        self.build_dmd2_img_dataloader(global_batch_size = self.world_size)
        self.build_optimizer()
        self.build_lr_scheduler()

    def init_resume(self):
        self.resume = False
        self.last_iter = -1
        if not osp.exists(self.save_path) and self.rank == 0:
            os.makedirs(self.save_path)
        last_iter_file = osp.join(self.save_path, "last_iter")
        if not osp.exists(last_iter_file):
            return
        self.resume = True
        with open(last_iter_file) as f:
            self.last_iter = int(f.readlines()[0])
            f.close()
        self.resume_model_ckpt_path = osp.join(
            self.save_path, "ckpt_model_{}.pth".format(self.last_iter)
        )
        self.resume_ema_ckpt_path = osp.join(
            self.save_path, "ckpt_ema_{}.pth".format(self.last_iter)
        )
        self.resume_scheduler_ckpt_path = osp.join(
            self.save_path, "ckpt_scheduler_{}.pth".format(self.last_iter)
        )
        self.last_iter -= 1

    def init_seed(self):
        seed = self.config["train"]["seed"]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def init_logger(self):
        self.log_interval = self.config["train"]["log_interval"]
        # wandb config from yaml
        wandb_config = self.config.get("wandb", {})
        self.use_wandb = wandb_config.get("enabled", False) and wandb is not None
        self.wandb_iters = wandb_config.get("wandb_iters", 20)  # visual logging interval (every 20 iters)
        self.wandb_log_loss_iters = wandb_config.get("log_loss_iters", 10)  # loss logging interval (every 10 iters)
        if self.use_wandb and self.rank == 0:
            wandb.init(
                project=wandb_config.get("project", "senseflow-sd35"),
                entity=wandb_config.get("entity", "aaroncaozj_team"),
                name=wandb_config.get("name", f"sd35-senseflow-{time.strftime('%m%d-%H%M')}"),
                config=self.config,
                dir=self.save_path,
                mode=wandb_config.get("mode", "online"),
            )
            print(f"[wandb] initialized: project={wandb_config.get('project')}, name={wandb.run.name}")
        elif self.rank == 0 and not self.use_wandb:
            print("[wandb] disabled. Set wandb.enabled=true in config yaml to enable.")

    def init_saver(self):
        self.save_interval = self.config["train"]["save_interval"]

    def init_distributed(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        torch.distributed.init_process_group("nccl")

    def build_dmd2_img_dataloader(
            self,
            global_batch_size: int = 8,
            num_workers: int = 32,
            pin_memory: bool = False,
            persistent_workers: bool = True,
            ):
        # TODO: Replace PLACEHOLDER_JSON_DATASET_PATH with your local path to the dataset JSON file
        # The JSON file should contain 'keys', 'image_paths', and 'prompts' fields
        # See README for the required JSON file structure
        self.real_dataset = LaionText2ImageDataset(json_path='/root/highspeedstorage/model_distill/SenseFlow/dataset/LAION_Aesthetics_1024/training_data/dataset.json', repeat=1)
        sampler = DistributedSampler(self.real_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
        denoising_dataloader = DataLoader(
            self.real_dataset,
            batch_size=global_batch_size // self.world_size,
            shuffle=False,  # shuffle is controlled by sampler
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sampler=sampler,
            collate_fn=default_collate
        )
        self.denoising_dataloader = cycle(denoising_dataloader)
        self.batch_size = global_batch_size // self.world_size

    def build_model(self):
        # vae
        # text encoder
        # generator / student
        # guidance model
        self.backward_simulation = True
        self.denoising_timestep = 1000
        self.num_denoising_step = 4
        self.num_train_timesteps = 1000
        self.denoising_step_list = torch.tensor(
            list(range(self.denoising_timestep-1, 0, -(self.denoising_timestep//self.num_denoising_step))),
            dtype=torch.long,
        ).cuda() # []
        # sdxl configs and ckpt paths
        from argparse import Namespace
        args = Namespace()
        # TODO: Replace PLACEHOLDER_SD35_MEDIUM_PATH with your local path to stable-diffusion-3.5-medium
        # Download from HuggingFace: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
        args.pretrained_teacher_model = '/root/highspeedstorage/model_distill/SenseFlow/ckpt/stable-diffusion-3.5-medium'
        args.teacher_revision = None
        args.variant = None
        args.pretrained_vae_model_name_or_path = None
        args.pretrained_unet_lcm_path = None
        args.num_train_timesteps = 1000
        args.min_step_percent = 0.02
        args.max_step_percent = 0.98
        args.real_guidance_scale = 4.0
        args.fake_guidance_scale = 1.0
        args.diffusion_gan = True
        args.diffusion_gan_max_timestep = 1000
        args.use_decoupled_dmd = True
        args.use_isg = True

        self.sdxl_lora = False
        self.disable_sdxl_crossattn = False # True
        self.allin_bf16 = True
        self.laion_crop_size = 1024 # 768 # 512 # 256 # 768 # 1024 # 512
        self.use_decoupled_dmd = args.use_decoupled_dmd
        self.use_isg = args.use_isg
        print('laion crop size', self.laion_crop_size, 'all in bf16', self.allin_bf16)
        print('use Decoupled DMD:', self.use_decoupled_dmd)
        print('use ISG:', self.use_isg)
        print('sdxl lora', self.sdxl_lora, 'crossatt disable', self.disable_sdxl_crossattn, 'lcm path', args.pretrained_unet_lcm_path)
        if args.pretrained_unet_lcm_path is not None:
            print('using lcm pretraining')
        self.fp16vae = torch.float32
        self.fp16unet = torch.float32

        # load SD3 vae
        vae_path = (
            args.pretrained_teacher_model
            if args.pretrained_vae_model_name_or_path is None
            else args.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.teacher_revision,
            torch_dtype=self.fp16vae,
        )
        vae.enable_gradient_checkpointing()
        
        # load SDXL text encoders
        tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_teacher_model, subfolder="tokenizer", revision=args.teacher_revision, use_fast=False
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            args.pretrained_teacher_model, subfolder="tokenizer_2", revision=args.teacher_revision, use_fast=False
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="tokenizer_3",
            revision=args.teacher_revision,
        )

        # 3. Load text encoders from SD-XL checkpoint.
        # import correct text encoder classes
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.teacher_revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.teacher_revision, subfolder="text_encoder_2"
        )
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
            args.pretrained_teacher_model, args.teacher_revision, subfolder="text_encoder_3"
        )
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="text_encoder",
            revision=args.teacher_revision,
            variant=args.variant,
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="text_encoder_2",
            revision=args.teacher_revision,
            variant=args.variant,
        )
        text_encoder_three = text_encoder_cls_three.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="text_encoder_3",
            revision=args.teacher_revision,
            variant=args.variant,
        )

        text_encoder_one.requires_grad_(False).eval().cuda()
        text_encoder_two.requires_grad_(False).eval().cuda()
        text_encoder_three.requires_grad_(False).eval().cuda()
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        self.compute_embeddings_fn = functools.partial(
                                                    encode_prompt,
                                                    text_encoders=text_encoders,
                                                    tokenizers=tokenizers,
                                                )

        # load generator
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="transformer",
            revision=args.teacher_revision,
            variant=args.variant,
        )
        transformer.enable_gradient_checkpointing()

        # guidance model
        fake_transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="transformer",
            revision=args.teacher_revision,
            variant=args.variant,
        )
        fake_transformer.enable_gradient_checkpointing()
        real_transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_teacher_model,
            subfolder="transformer",
            revision=args.teacher_revision,
            variant=args.variant,
        )
        real_transformer.enable_gradient_checkpointing()
        fake_transformer.requires_grad_(True)
        real_transformer.requires_grad_(False)

        self.guidance_model = GuidanceModel(fake_unet=fake_transformer, real_unet=real_transformer, args=args)
        try:
            import xformers
            # unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
            # teacher_unet.enable_xformers_memory_efficient_attention()
            print('** enable xformer')
        except:
            pass

        self.use_hybrid = False
        self.use_full_shard = True

        if self.use_hybrid and self.use_full_shard:
            raise NotImplementedError

        print("use hybrid: {}".format(self.use_hybrid))

        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=0
        )
        print('self.local_rank: ', self.local_rank)
        if self.use_hybrid:
            self.model = FSDP(
                transformer.cuda(),
                device_id=self.local_rank,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.HYBRID_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
            )

            self.guidance_model = FSDP(
                self.guidance_model.cuda(),
                device_id=self.local_rank,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.HYBRID_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
            )

        elif self.use_full_shard:
            self.model = FSDP(
                transformer.cuda(),
                device_id=self.local_rank,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
            )

            self.guidance_model = FSDP(
                self.guidance_model.cuda(),
                device_id=self.local_rank,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=my_auto_wrap_policy,
            )

        else:
            pass

        # noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model, subfolder="scheduler"
        )
        self.sigmas = torch.flip(self.noise_scheduler.sigmas, dims=[0]).to('cuda')
        self.sigmas_timew = self.noise_scheduler.sigmas.to('cuda')

        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.cuda()

        self.scaling_factor = self.vae.config.scaling_factor

        # VFM GAN
        dino_name = 'vit_large_patch14_dinov2.lvd142m' 
        # hooks = [5, 11, 17, 23]
        hooks = [2, 5, 8, 10, 14, 19, 23]
        fix_res_dino = False
        crop_plan = 'none'
        p_crop, use_checkpoint, dis_useatt, ret_cls = 0.5, False, False, False
        dis_conv2d, dsample, diffaug, dino_pretrain = True, 3, True, True
        self.net_d = torch.nn.parallel.DistributedDataParallel(
            ProjectedDiscriminatorPlus(c_dim=768, dino_name=dino_name, hooks=hooks, crop_plan=crop_plan, p_crop=p_crop, use_checkpoint=use_checkpoint, 
                fix_res_dino=fix_res_dino, useatt=dis_useatt, ret_cls=ret_cls, conv2d=dis_conv2d, downsample=dsample, diffaug=diffaug, dino_pretrain=dino_pretrain).cuda(),
            device_ids=[self.local_rank],
        )
        self.net_d.train()
        self.cri_gan = GANLoss(
            gan_type="hinge",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0,
        )
        self.clip = CLIP().cuda().eval().requires_grad_(False)


    def build_optimizer(self):
        assert "optimizer" in self.config, "config does not contain key 'optimizer'"
        optimizer_config = self.config["optimizer"]

        use_8bit_adamw = False
        print('** 8bit adamw', use_8bit_adamw)
        if use_8bit_adamw:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=1.0, 
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"], eps=optimizer_config["eps"],)
            self.optimizer_guidance = bnb.optim.AdamW8bit([param for param in self.guidance_model.parameters() if param.requires_grad], lr=1.0, 
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"], eps=optimizer_config["eps"],)
            self.optimizer_d = bnb.optim.AdamW8bit(
                list(self.net_d.module.heads.parameters()),
                lr=1.0,
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                eps=optimizer_config["eps"],
                weight_decay=optimizer_config["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1.0,
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                eps=optimizer_config["eps"],
                weight_decay=optimizer_config["weight_decay"],
            )
            self.optimizer_guidance = torch.optim.AdamW(
                [param for param in self.guidance_model.parameters() if param.requires_grad],
                lr=1.0,
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                eps=optimizer_config["eps"],
                weight_decay=optimizer_config["weight_decay"],
            )
            self.optimizer_d = torch.optim.AdamW(
                list(self.net_d.module.heads.parameters()),
                lr=1.0,
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                eps=optimizer_config["eps"],
                weight_decay=optimizer_config["weight_decay"],
            )

    def build_lr_scheduler(self):
        assert (
            "lr_scheduler" in self.config
        ), "config does not contain key 'lr_scheduler'"
        scheduler_config = self.config["lr_scheduler"]
        scheduler = instantiate_from_config(scheduler_config)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=scheduler.schedule)
        guidance_lr_scheduler = instantiate_from_config(scheduler_config)
        self.guidance_lr_scheduler = LambdaLR(self.optimizer_guidance, lr_lambda=guidance_lr_scheduler.schedule)
        d_lr_scheduler = instantiate_from_config(scheduler_config)
        self.d_lr_scheduler = LambdaLR(self.optimizer_d, lr_lambda=d_lr_scheduler.schedule)

    @torch.no_grad()
    # 图像空间 -> 潜空间
    def encode_first_stage_model(self, data):
        images = data.cuda()
        batch_size = images.shape[0]
        output = self.vae.encode(images).latent_dist.sample()
        output = (output - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return output

    @torch.no_grad()
    def sample_backward(self, noisy_image, prompt_embeds, pooled_prompt_embeds):
        batch_size = noisy_image.shape[0]
        device = noisy_image.device
        # we choose a random step and share it across all gpu
        selected_step = torch.randint(low=0, high=self.num_denoising_step, size=(1,), device=device, dtype=torch.long)

        generated_image = noisy_image


        for constant in self.denoising_step_list[:selected_step]:
            current_sigmas = self.sigmas[constant]
            # import pdb; pdb.set_trace()
            # print('selected_step, constant, current_sigmas: ', selected_step, constant, current_sigmas)
            # if generated_image != noisy_image:
            #     noisy_image = current_sigmas * torch.randn_like(generated_image) + (1.0 - current_sigmas) * generated_image
            # 采样一个与 generated_image 同尺寸的噪声图，叠加在上一轮生成的图像上
            noisy_image = current_sigmas * torch.randn_like(generated_image) + (1.0 - current_sigmas) * generated_image

            current_timesteps = torch.ones(batch_size, device=device, dtype=torch.long) * current_sigmas * self.noise_scheduler.config.num_train_timesteps

            generated_noise = self.model(
                hidden_states=noisy_image.float(),
                timestep=current_timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                pooled_projections=pooled_prompt_embeds.float()
            ).sample

            # 从 z_t 完全去噪到 x_0
            generated_image = (noisy_image - current_sigmas * generated_noise).to(noisy_image.dtype)

        return_timesteps = self.denoising_step_list[selected_step] * torch.ones(batch_size, device=device, dtype=torch.long)
        return_sigmas = self.sigmas[self.denoising_step_list[selected_step]] 
        return generated_image, return_timesteps, return_sigmas, selected_step

    @torch.no_grad()
    def prepare_denoising_data(self, denoising_dict, real_train_dict, noise):

        indices = torch.randint(
            0, self.num_denoising_step, (noise.shape[0],), device=noise.device, dtype=torch.long
        )
        timesteps = self.denoising_step_list.to(noise.device)[indices]
        cur_sigmas = self.sigmas[timesteps]
        cur_timesteps = cur_sigmas * self.noise_scheduler.config.num_train_timesteps
        denoising_prompt_embeds, denoising_pooled_prompt_embeds = self.compute_embeddings_fn(denoising_dict['caption'])

        if real_train_dict is not None:
            real_prompt_embeds, real_pooled_prompt_embeds = self.compute_embeddings_fn(real_train_dict['caption'])
            real_train_dict['text_embedding'] = real_prompt_embeds

            real_train_dict['pooled_prompt_embeds'] = real_pooled_prompt_embeds

        if self.backward_simulation:
            clean_images, timesteps, cur_sigmas, indices = self.sample_backward(torch.randn_like(noise), denoising_prompt_embeds, denoising_pooled_prompt_embeds)
        else:
            clean_images = denoising_dict['image'].to(noise.device)
        # print('cur_sigmas: ', cur_sigmas)

        # flow matching 的前向加噪过程
        noisy_image = cur_sigmas * noise + (1.0 - cur_sigmas) * clean_images

        return timesteps, cur_sigmas, denoising_prompt_embeds, denoising_pooled_prompt_embeds, real_train_dict, noisy_image, clean_images, indices

    # 潜空间 -> 图像空间
    def differentiable_decode_first_stage(self, z):
        z = 1.0 / self.scaling_factor * z + self.vae.config.shift_factor
        return self.vae.decode(z).sample

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
        return x[(...,) + (None,) * dims_to_append]
    
    def get_isg_guidance(
            self,
            noisy_image,
            timestep_indice,
            prompt_embeds,
            pooled_prompt_embeds,
            uncond_prompt_embeds,
            uncond_pooled_prompt_embeds,
            generator_pred,
        ):
        """
        Intra-Segment Guidance (ISG) loss computation for SD3.5 model.
        Samples an intermediate timestep between timestep and timestep_prev,
        and computes guidance loss by comparing generator's direct path with 
        the reference path through the intermediate timestep.
        """
        # Sample an intermediate timestep_mid between timestep and timestep_prev
        # 对应 backward sim 的最后一个时间步
        timestep = self.denoising_step_list.to(noisy_image.device)[timestep_indice]
        cur_sigmas = self.sigmas[timestep]
        current_timesteps = torch.ones(self.batch_size, device=noisy_image.device, dtype=torch.long) * cur_sigmas * self.noise_scheduler.config.num_train_timesteps
        # 采样一个中间终点
        if timestep_indice == 3:
            timestep_prev = torch.zeros_like(timestep).to(torch.long).to('cuda')
        else:
            timestep_prev = self.denoising_step_list.to(noisy_image.device)[timestep_indice+1]
        # 在 [timestep-50, timestep_prev+50] 之间随机采样一个中间时间步
        timestep_mid = torch.randint(torch.ceil(timestep_prev+50).int(), torch.floor(timestep-50).int()+1, (1,)).to('cuda').to(torch.long)
        sigma_mid = self.sigmas[timestep_mid]
        sigma_prev = self.sigmas[timestep_prev]
        current_timesteps_mid = torch.ones(self.batch_size, device=noisy_image.device, dtype=torch.long) * sigma_mid * self.noise_scheduler.config.num_train_timesteps
        # Use guidance_model.real_unet to denoise from timestep to timestep_mid (no grad)
        with torch.no_grad():
            c = random.uniform(2, 4)
            real_score_pred = predict_noise(
                self.guidance_model.real_unet, noisy_image, prompt_embeds, uncond_prompt_embeds, 
                current_timesteps, guidance_scale=c,
                pooled_prompt_embeds=pooled_prompt_embeds,
                uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds
            )
            x_mid = noisy_image + ((sigma_mid - cur_sigmas) * real_score_pred).to(noisy_image.dtype)
            generated_noise = self.model(
                hidden_states=x_mid.float(),
                timestep=current_timesteps_mid,
                encoder_hidden_states=prompt_embeds.float(),
                pooled_projections=pooled_prompt_embeds.float()
            ).sample
            # 增量式的去噪公式
            target_x_prev = x_mid + ((sigma_prev - sigma_mid) * generated_noise).to(x_mid.dtype)
        generated_x_prev = noisy_image + ((sigma_prev - cur_sigmas) * generator_pred).to(noisy_image.dtype)
        isg_guidance_loss = torch.mean(torch.sqrt((generated_x_prev.float() - target_x_prev.float()) ** 2 + 0.001**2) - 0.001)
        return isg_guidance_loss, target_x_prev, generated_x_prev

    def train(self):
        scaler = ShardedGradScaler()
        guidance_scaler = ShardedGradScaler()
        scaler_d = ShardedGradScaler()
        iter_time_list = []
        torch.cuda.synchronize()
        dist.barrier()
        iter_begin_time = time.time()

        self.num_total_iters = 10000000
        self.GD = 10  # Ratio of generator to discriminator optimization steps
        self.gan_weight = 0.1

        self.latent_channel = 16
        self.latent_resolution = 128
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.ida_w = 0.98

        rank, world_size = dist.get_rank(), dist.get_world_size()
        current_iter = 0
        for index in range(self.num_total_iters):
            current_iter += 1
            COMPUTE_GENERATOR_GRADIENT = current_iter % self.GD == 0
            self.backward_simulation = torch.rand(1).item() > 0.5  # 50% probability

            # 1. 数据集采样一个batch
            denoising_dict = next(self.denoising_dataloader)

            # 图像预处理
            ori_image = denoising_dict['image'].cuda()
            # Resize to 1024x1024 to ensure consistent latent dimensions (only once!)
            ori_image_resized = F.interpolate(ori_image, size=(1024, 1024), mode='bilinear', align_corners=False)
            # vae 编码
            denoising_dict['image'] = self.encode_first_stage_model(ori_image_resized)
            # 采样纯噪声
            noise = torch.randn(self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution).cuda()

            # generator forward
            # 有条件文本编码 + denoising 数据准备
            timesteps, sigmas, denoising_prompt_embeds, denoising_pooled_prompt_embeds, _, noisy_image, clean_image, timestep_indice = self.prepare_denoising_data(
                    denoising_dict, None, noise
                )
            current_timesteps = torch.ones(self.batch_size, device=noise.device, dtype=torch.long) * sigmas * self.noise_scheduler.config.num_train_timesteps
            # prompt_embeds = denoising_encoded_text_cond.pop("prompt_embeds")
            # Build unconditional embeddings
            # uncond = copy.deepcopy(denoising_encoded_text_cond)
            # uncond["text_embeds"] = torch.zeros_like(denoising_encoded_text_cond["text_embeds"]).cuda()
            # uncond_prompt_embeds = torch.zeros_like(prompt_embeds).cuda().float()
            # 无条件文本编码为全零向量
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = self.compute_embeddings_fn([''] * noise.shape[0])
            
            # 2. generator 前向
            if COMPUTE_GENERATOR_GRADIENT:
                with self.network_context_manager:
                    generated_noise = self.model(
                        hidden_states=noisy_image.float(),
                        timestep=current_timesteps,
                        encoder_hidden_states=denoising_prompt_embeds.float(),
                        pooled_projections=denoising_pooled_prompt_embeds.float()
                    ).sample
                    # ISG guidance
                    if self.use_isg:
                        isg_guidance_loss, target_x_prev, generated_x_prev = self.get_isg_guidance(
                            noisy_image=noisy_image,
                            timestep_indice=timestep_indice,
                            prompt_embeds=denoising_prompt_embeds.float(),
                            pooled_prompt_embeds=denoising_pooled_prompt_embeds.float(),
                            uncond_prompt_embeds=uncond_prompt_embeds.float(),
                            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds.float(),
                            generator_pred=generated_noise,
                        )
                        print('isg_guidance_loss: ', isg_guidance_loss)
                    else:
                        isg_guidance_loss = 0.0
                        target_x_prev = None
                        generated_x_prev = None
            else:
                # 冻结模型计算生成图像
                with torch.no_grad():
                    generated_noise = self.model(
                        hidden_states=noisy_image.float(),
                        timestep=current_timesteps,
                        encoder_hidden_states=denoising_prompt_embeds.float(),
                        pooled_projections=denoising_pooled_prompt_embeds.float()
                    ).sample

            generated_image = (noisy_image - sigmas * generated_noise).to(noisy_image.dtype)

            with torch.no_grad():
                clipcond = self.clip.encode_text(denoising_dict["caption"])

            if COMPUTE_GENERATOR_GRADIENT:
                generator_data_dict = {
                    "image": generated_image,
                    "text_embedding": denoising_prompt_embeds,
                    "uncond_embedding": uncond_prompt_embeds,
                    "real_train_dict": None,
                    "pooled_prompt_embeds": denoising_pooled_prompt_embeds,
                    "uncond_pooled_prompt_embeds": uncond_pooled_prompt_embeds,
                    "timesteps": timesteps,  # Pass timesteps for Decoupled DMD
                }

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                # guidance 前向计算 generator loss
                generator_loss_dict, generator_log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )

                # Compute GAN loss for generator
                with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
                    # vae decode 回图像域，用来计算 GAN Loss
                    imgres = self.differentiable_decode_first_stage(generated_image.float())
                    # Resize to fixed 1024x1024 for discriminator (only once!)
                    imgres_resized = F.interpolate(imgres, size=(1024, 1024), mode='bilinear', align_corners=False)
                with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                    fake_g_pred = self.net_d(imgres_resized.float(), clipcond, ori_image_resized.float())
            else:
                generator_loss_dict = {}
                generator_log_dict = {}
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
                        imgres = self.differentiable_decode_first_stage(generated_image.float())
                        # Resize to fixed 1024x1024 for discriminator (only once!)
                        imgres_resized = F.interpolate(imgres, size=(1024, 1024), mode='bilinear', align_corners=False)
                    with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                        fake_g_pred = self.net_d(imgres_resized.float(), clipcond, ori_image_resized.float())

            # GAN Loss 带 timesteps 权重，高噪声权重大
            l_g_gan = self.gan_weight * self.cri_gan(
                        fake_g_pred, True, is_disc=False, keepdim=True
                    )
            timew = (
                    extract_into_tensor(
                        self.sigmas_timew,
                        torch.clamp_max(timesteps, 999),
                        l_g_gan.shape,
                    )
                    ** 2
                )
            # print("timew: ", timew)
            generator_loss_dict['l_g_gan'] = (l_g_gan * timew).mean()


            generator_log_dict["guidance_data_dict"] = {
                "image": generated_image.detach(),
                "text_embedding": denoising_prompt_embeds.detach(),
                "uncond_embedding": uncond_prompt_embeds.detach(),
                "real_train_dict": None,
                "pooled_prompt_embeds": denoising_pooled_prompt_embeds,
                "uncond_pooled_prompt_embeds": uncond_pooled_prompt_embeds
            }
            generator_log_dict['denoising_timestep'] = timesteps
            generator_log_dict['denoising_sigmas'] = sigmas

            # 3. generator model 更新
            generator_loss = 0.0 
            if COMPUTE_GENERATOR_GRADIENT:
                if self.use_decoupled_dmd:
                    generator_loss += generator_loss_dict["loss_decoupled"]
                else:
                    raise NotImplementedError("Decoupled DMD not enabled")
                generator_loss += (timew * l_g_gan).mean()
                if self.use_isg:
                    generator_loss += 0.5 * isg_guidance_loss
                scaler.scale(generator_loss).backward()
                scaler.unscale_(self.optimizer)
                self.model.clip_grad_norm_(1.0)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                self.optimizer_guidance.zero_grad()

                if self.ida_w != 1.0:
                    with torch.no_grad():
                        target_model_named_parm_dict = dict(self.guidance_model.fake_unet.named_parameters())
                        for name_A, param_A in self.model.named_parameters():
                            param_B = target_model_named_parm_dict[name_A]
                            if param_A.requires_grad:
                                param_B.data.mul_(self.ida_w).add_(param_A.data, alpha=1 - self.ida_w)

            self.lr_scheduler.step()

            # 4. guidance model 更新
            self.guidance_model.requires_grad_(True)
            self.guidance_model.real_unet.requires_grad_(False)
            # update guidance model (dfake and classifier)
            guidance_loss_dict, guidance_log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=generator_log_dict['guidance_data_dict']
            )
            guidance_loss = 0 

            guidance_loss += guidance_loss_dict["loss_fake_mean"]
            guidance_scaler.scale(guidance_loss).backward()
            guidance_scaler.unscale_(self.optimizer_guidance)
            self.guidance_model.clip_grad_norm_(1.0)
            guidance_scaler.step(self.optimizer_guidance)
            guidance_scaler.update()
            self.optimizer.zero_grad()
            self.optimizer_guidance.zero_grad()
            self.guidance_lr_scheduler.step()


            # discriminator 更新
            d_loss_dict = {}
            # Reuse already resized ori_image (gt is same as ori_image_resized)
            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                real_d_pred, _ref, _fea = self.net_d(
                    ori_image_resized, clipcond, ori_image_resized, return_ref=True
                )
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True, keepdim=True)
                l_d_real = torch.mean(l_d_real * timew)
                d_loss_dict["l_d_real"] = l_d_real
                d_loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())

                scaler_d.scale(l_d_real).backward()
                # Reuse imgres_resized computed earlier
                fake_d_pred = self.net_d(imgres_resized.detach(), clipcond, ori_image_resized)
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True, keepdim=True)
                l_d_fake = torch.mean(l_d_fake * timew)
                d_loss_dict["l_d_fake"] = l_d_fake
                d_loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())

                scaler_d.scale(l_d_fake).backward()
                scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(
                    self.net_d.module.heads.parameters(), 1.0
                )
                scaler_d.step(self.optimizer_d)
                scaler_d.update()
                self.optimizer_d.zero_grad()
                self.d_lr_scheduler.step()


            torch.cuda.synchronize()
            dist.barrier()
            iter_end_time = time.time()
            iter_time = iter_end_time - iter_begin_time
            iter_begin_time = iter_end_time
            iter_time_list.append(iter_time)
            if len(iter_time_list) > 100:
                _ = iter_time_list.pop(0)

            # ==================== wandb loss logging ====================
            if self.use_wandb and self.rank == 0 and current_iter % self.wandb_log_loss_iters == 0:
                # Compute generated image statistics
                generated_image_mean = generator_log_dict["guidance_data_dict"]["image"].mean().item()
                generated_image_std = generator_log_dict["guidance_data_dict"]["image"].std().item()
                
                wandb_loss_dict = {
                    "loss_fake_mean": guidance_loss_dict['loss_fake_mean'].item(),
                    "l_g_gan": float(generator_loss_dict['l_g_gan'].detach().cpu()),
                    "l_d_real": float(d_loss_dict['l_d_real'].detach().cpu()),
                    "l_d_fake": float(d_loss_dict['l_d_fake'].detach().cpu()),
                    "out_d_real": float(d_loss_dict['out_d_real'].detach().cpu()),
                    "out_d_fake": float(d_loss_dict['out_d_fake'].detach().cpu()),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "guidance_lr": self.optimizer_guidance.param_groups[0]["lr"],
                    "iter_time": iter_time,
                    "generated_image_mean": generated_image_mean,
                    "generated_image_std": generated_image_std,
                    "denoising_timestep": generator_log_dict['denoising_timestep'][0].cpu().item(),
                }
                if COMPUTE_GENERATOR_GRADIENT:
                    wandb_loss_dict["generator_loss"] = float(generator_loss) if isinstance(generator_loss, float) else generator_loss.item()
                    wandb_loss_dict["guidance_loss"] = float(guidance_loss) if isinstance(guidance_loss, float) else guidance_loss.item()
                    if self.use_decoupled_dmd:
                        wandb_loss_dict.update({
                            "loss_decoupled": generator_loss_dict["loss_decoupled"].item(),
                            "loss_ca": generator_loss_dict["loss_ca"].item(),
                            "loss_dm": generator_loss_dict["loss_dm"].item(),
                            "ca_update_norm": generator_log_dict.get('ca_update_norm', 0.0),
                            "dm_update_norm": generator_log_dict.get('dm_update_norm', 0.0),
                            "ca_timesteps_mean": generator_log_dict.get('ca_timesteps_mean', 0.0),
                            "dm_timesteps_mean": generator_log_dict.get('dm_timesteps_mean', 0.0),
                        })
                    if self.use_isg:
                        wandb_loss_dict["isg_guidance_loss"] = float(isg_guidance_loss) if isinstance(isg_guidance_loss, float) else isg_guidance_loss.item()
                wandb.log(wandb_loss_dict, step=current_iter)

            # ==================== image logging (local + wandb) ====================
            visual = (current_iter % self.wandb_iters == 0) if self.use_wandb else False
            self.image_logger_save_dir = self.save_path
            if current_iter % 20 == 0 and self.rank == 0:
                with torch.no_grad():
                    print('denoising_dict caption: ', denoising_dict['caption'])
                    batchid = 0
                    model_pred_img = self.differentiable_decode_first_stage(generated_image.float())
                    clean_img = self.differentiable_decode_first_stage(clean_image.float())
                    x_start_img = self.differentiable_decode_first_stage(denoising_dict['image'].float().cuda())
                    if COMPUTE_GENERATOR_GRADIENT and self.use_decoupled_dmd:
                        pred_dm_real_cond_image = self.differentiable_decode_first_stage(generator_log_dict['pred_dm_real_cond_image'].float().cuda())
                        pred_dm_fake_cond_image = self.differentiable_decode_first_stage(generator_log_dict['pred_dm_fake_cond_image'].float().cuda())
                        pred_ca_real_cond_image = self.differentiable_decode_first_stage(generator_log_dict['pred_ca_real_cond_image'].float().cuda())
                        pred_ca_real_uncond_image = self.differentiable_decode_first_stage(generator_log_dict['pred_ca_real_uncond_image'].float().cuda())
                        if self.use_isg and target_x_prev is not None:
                            target_x_prev_img = self.differentiable_decode_first_stage(target_x_prev.float().cuda()) #target_x_prev, generated_x_prev
                            generated_x_prev_img = self.differentiable_decode_first_stage(generated_x_prev.float().cuda())
                    faketrain_x0_pred =  self.differentiable_decode_first_stage(guidance_log_dict['faketrain_x0_pred'].float())
                    print(f"save image {denoising_dict['caption'][batchid]}")
                    if COMPUTE_GENERATOR_GRADIENT and self.use_decoupled_dmd:
                        if self.use_isg:
                            save_model_pred_img , save_clean_img, save_x_start_img, save_pred_dm_real_cond_image, save_pred_dm_fake_cond_image, save_pred_ca_real_cond_image, save_pred_ca_real_uncond_image, save_faketrain_x0_pred, save_target_x_prev_img, save_generated_x_prev_img = \
                                model_pred_img[[batchid]], clean_img[[batchid]], x_start_img[[batchid]], pred_dm_real_cond_image[[batchid]], pred_dm_fake_cond_image[[batchid]], pred_ca_real_cond_image[[batchid]], pred_ca_real_uncond_image[[batchid]], faketrain_x0_pred[[batchid]], target_x_prev_img[[batchid]], generated_x_prev_img[[batchid]]
                        else:
                            save_model_pred_img , save_clean_img, save_x_start_img, save_pred_dm_real_cond_image, save_pred_dm_fake_cond_image, save_pred_ca_real_cond_image, save_pred_ca_real_uncond_image, save_faketrain_x0_pred = \
                                model_pred_img[[batchid]], clean_img[[batchid]], x_start_img[[batchid]], pred_dm_real_cond_image[[batchid]], pred_dm_fake_cond_image[[batchid]], pred_ca_real_cond_image[[batchid]], pred_ca_real_uncond_image[[batchid]], faketrain_x0_pred[[batchid]]

                    else:
                        save_model_pred_img , save_clean_img, save_x_start_img, save_faketrain_x0_pred = \
                            model_pred_img[[batchid]], clean_img[[batchid]], x_start_img[[batchid]], faketrain_x0_pred[[batchid]]
                    th2np = lambda x: x[batchid].detach().permute(1, 2, 0).cpu().numpy()
                    np2save = lambda x: ((x + 1).clip(0, 2) / 2. * 255).astype(np.uint8)
                    res = []
                    if COMPUTE_GENERATOR_GRADIENT and self.use_decoupled_dmd:
                        if self.use_isg:
                            for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_pred_dm_real_cond_image, save_pred_dm_fake_cond_image, save_pred_ca_real_cond_image, save_pred_ca_real_uncond_image, save_faketrain_x0_pred, save_target_x_prev_img, save_generated_x_prev_img]:
                                res.append(np2save(th2np(item)))
                        else:
                            for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_pred_dm_real_cond_image, save_pred_dm_fake_cond_image, save_pred_ca_real_cond_image, save_pred_ca_real_uncond_image, save_faketrain_x0_pred]:
                                res.append(np2save(th2np(item)))
                    else:
                        for item in [save_model_pred_img, save_clean_img, save_x_start_img, save_faketrain_x0_pred]:
                            res.append(np2save(th2np(item)))
                    os.makedirs(os.path.join(self.image_logger_save_dir, 'image_log'), exist_ok=True)
                    tpath = os.path.join(self.image_logger_save_dir, 'image_log', 'image_log-%d-%d.jpg' % (current_iter, generator_log_dict['denoising_timestep'][0].cpu().item()))
                    print('log img save', tpath)
                    res = np.concatenate(res, axis=1)
                    Image.fromarray(res).save(tpath)

                    # ========== wandb visual: log generated images and intermediate results ==========
                    if visual and self.use_wandb:
                        wandb_data_dict = {}
                        # --- generated image (the main output) ---
                        wandb_data_dict["generated_image"] = wandb.Image(
                            np2save(th2np(save_model_pred_img)),
                            caption=denoising_dict['caption'][batchid][:200]
                        )
                        # --- clean image (backward simulation x0) ---
                        wandb_data_dict["clean_image_backward_sim"] = wandb.Image(
                            np2save(th2np(save_clean_img)),
                            caption="backward simulation x0"
                        )
                        # --- original encoded image ---
                        wandb_data_dict["original_image"] = wandb.Image(
                            np2save(th2np(save_x_start_img)),
                            caption="VAE encoded original"
                        )
                        # --- dfake x0 prediction ---
                        wandb_data_dict["faketrain_x0_pred"] = wandb.Image(
                            np2save(th2np(save_faketrain_x0_pred)),
                            caption="dfake x0 prediction"
                        )
                        # --- Decoupled DMD intermediate images ---
                        if COMPUTE_GENERATOR_GRADIENT and self.use_decoupled_dmd:
                            wandb_data_dict["pred_dm_real_cond_image"] = wandb.Image(
                                np2save(th2np(save_pred_dm_real_cond_image)),
                                caption="DM: real model cond pred"
                            )
                            wandb_data_dict["pred_dm_fake_cond_image"] = wandb.Image(
                                np2save(th2np(save_pred_dm_fake_cond_image)),
                                caption="DM: fake model cond pred"
                            )
                            wandb_data_dict["pred_ca_real_cond_image"] = wandb.Image(
                                np2save(th2np(save_pred_ca_real_cond_image)),
                                caption="CA: real model cond pred"
                            )
                            wandb_data_dict["pred_ca_real_uncond_image"] = wandb.Image(
                                np2save(th2np(save_pred_ca_real_uncond_image)),
                                caption="CA: real model uncond pred"
                            )
                            # # --- difference map: DM real vs fake ---
                            # dm_diff = (pred_dm_real_cond_image[[batchid]] - pred_dm_fake_cond_image[[batchid]])
                            # dm_diff_norm = dm_diff.abs().mean().item()
                            # dm_diff_vis = (dm_diff - dm_diff.min()) / (dm_diff.max() - dm_diff.min() + 1e-8)
                            # dm_diff_vis = (dm_diff_vis - 0.5) / 0.5  # normalize to [-1, 1]
                            # wandb_data_dict["dm_difference_map"] = wandb.Image(
                            #     np2save(th2np(dm_diff_vis)),
                            #     caption=f"DM diff norm: {dm_diff_norm:.4f}"
                            # )
                            # wandb_data_dict["dm_difference_norm"] = dm_diff_norm
                            # --- Decoupled DMD loss details ---
                            wandb_data_dict["loss_decoupled"] = generator_loss_dict["loss_decoupled"].item()
                            wandb_data_dict["loss_ca"] = generator_loss_dict["loss_ca"].item()
                            wandb_data_dict["loss_dm"] = generator_loss_dict["loss_dm"].item()
                            # --- ISG images if available ---
                            if self.use_isg and target_x_prev is not None:
                                wandb_data_dict["isg_target_x_prev"] = wandb.Image(
                                    np2save(th2np(save_target_x_prev_img)),
                                    caption="ISG: target x_prev"
                                )
                                wandb_data_dict["isg_generated_x_prev"] = wandb.Image(
                                    np2save(th2np(save_generated_x_prev_img)),
                                    caption="ISG: generator x_prev"
                                )
                        # --- denoising timestep ---
                        wandb_data_dict["denoising_timestep"] = generator_log_dict['denoising_timestep'][0].cpu().item()
                        # --- the concatenated local image as well ---
                        wandb_data_dict["image_log_concat"] = wandb.Image(
                            Image.fromarray(res),
                            caption=f"iter={current_iter} t={generator_log_dict['denoising_timestep'][0].cpu().item()}"
                        )
                        wandb.log(wandb_data_dict, step=current_iter)

            if current_iter % 20 == 0 and self.rank == 0:
                if COMPUTE_GENERATOR_GRADIENT and self.use_decoupled_dmd:
                    if self.use_isg:
                        print(
                            "iter {}/{}, generator_loss: {}, loss_decoupled: {}, loss_ca: {}, loss_dm: {}, isg_guidance_loss: {}, guidance_loss: {}, loss_fake_mean: {}, l_g_gan: {}, l_d_real: {}, l_d_fake: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                                current_iter,
                                self.num_total_iters,
                                generator_loss,
                                generator_loss_dict["loss_decoupled"].mean(),
                                generator_loss_dict["loss_ca"].mean(),
                                generator_loss_dict["loss_dm"].mean(),
                                isg_guidance_loss,
                                guidance_loss,
                                guidance_loss_dict["loss_fake_mean"],
                                float(generator_loss_dict["l_g_gan"].detach().cpu()),
                                float(d_loss_dict["l_d_real"].detach().cpu()),
                                float(d_loss_dict["l_d_fake"].detach().cpu()),
                                self.optimizer.param_groups[0]["lr"],
                                self.optimizer_guidance.param_groups[0]["lr"],
                                sum(iter_time_list) / len(iter_time_list),
                                iter_time,
                            )
                        )
                    else:
                        print(
                            "iter {}/{}, generator_loss: {}, loss_decoupled: {}, loss_ca: {}, loss_dm: {}, guidance_loss: {}, loss_fake_mean: {}, l_g_gan: {}, l_d_real: {}, l_d_fake: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                                current_iter,
                                self.num_total_iters,
                                generator_loss,
                                generator_loss_dict["loss_decoupled"].mean(),
                                generator_loss_dict["loss_ca"].mean(),
                                generator_loss_dict["loss_dm"].mean(),
                                guidance_loss,
                                guidance_loss_dict["loss_fake_mean"],
                                float(generator_loss_dict["l_g_gan"].detach().cpu()),
                                float(d_loss_dict["l_d_real"].detach().cpu()),
                                float(d_loss_dict["l_d_fake"].detach().cpu()),
                                self.optimizer.param_groups[0]["lr"],
                                self.optimizer_guidance.param_groups[0]["lr"],
                                sum(iter_time_list) / len(iter_time_list),
                                iter_time,
                            )
                        )
                    
                else:
                    print(
                        "iter {}/{}, generator_loss: {}, guidance_loss: {}, loss_fake_mean: {}, l_g_gan: {}, l_d_real: {}, l_d_fake: {}, lr: {}, guidance_lr: {}, iter time avg: {}, iter time: {}".format(
                            current_iter,
                            self.num_total_iters,
                            generator_loss,
                            guidance_loss,
                            guidance_loss_dict["loss_fake_mean"],
                            float(generator_loss_dict["l_g_gan"].detach().cpu()),
                            float(d_loss_dict["l_d_real"].detach().cpu()),
                            float(d_loss_dict["l_d_fake"].detach().cpu()),
                            self.optimizer.param_groups[0]["lr"],
                            self.optimizer_guidance.param_groups[0]["lr"],
                            sum(iter_time_list) / len(iter_time_list),
                            iter_time,
                        )
                    )

            if (current_iter + 1) % self.save_interval == 0:
                self.save(current_iter + 1)

    def get_current_state_dict(self):
        fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            model_state = self.model.state_dict()
        return model_state

    def save(self, save_iter):
        fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )

        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
        ):
            model_state = self.model.state_dict()

        if self.rank == 0:

            model = {"model": model_state}

            model_save_path = osp.join(
                self.save_path, "ckpt_model_{}.pth".format(save_iter)
            )

            print("saving checkpoint to {}".format(model_save_path))
            torch.save(model, model_save_path)

            with open(osp.join(self.save_path, "last_iter"), "w+") as f:
                f.writelines([str(save_iter)])
                f.close()
            print("save done")

        torch.cuda.synchronize()
        dist.barrier()

class GuidanceModel(nn.Module):
    def __init__(self, fake_unet, real_unet, args):
        super().__init__()
        self.fake_unet = fake_unet
        self.real_unet = real_unet
        self.real_unet = self.real_unet.to(torch.bfloat16)
        # noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_teacher_model, subfolder="scheduler"
        )
        self.sigmas = torch.flip(self.noise_scheduler.sigmas, dims=[0]).to('cuda')

        self.num_train_timesteps = args.num_train_timesteps
        self.min_step = int(args.min_step_percent * self.noise_scheduler.config.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.noise_scheduler.config.num_train_timesteps)
        
        self.real_guidance_scale = args.real_guidance_scale 
        self.fake_guidance_scale = args.fake_guidance_scale
        self.use_bf16 = True
        self.use_decoupled_dmd = args.use_decoupled_dmd
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device='cuda', dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.cuda()
        timesteps = timesteps.cuda()
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def compute_decoupled_distribution_matching_loss(
        self, 
        latents,
        text_embedding,
        uncond_embedding,
        pooled_prompt_embeds,
        uncond_pooled_prompt_embeds,
        generator_timesteps=None,
    ):
        """Decoupled DMD: Separate CA (Classifier-free Alignment) and DM (Distribution Matching)"""
        pred_generator_image = latents
        batch_size = latents.shape[0]
        
        with torch.no_grad():
            if generator_timesteps is None:
                raise ValueError("generator_timesteps must be provided for Decoupled DMD")
            
            schedule_timesteps = self.noise_scheduler.timesteps.to(latents.device)
            schedule_sigmas = self.noise_scheduler.sigmas.to(latents.device)
            num_steps = len(schedule_timesteps)
            
            # Sample CA timesteps: earlier/cleaner than generator timesteps
            gen_indices = torch.zeros(batch_size, device=latents.device, dtype=torch.long)
            for i in range(batch_size):
                gen_t = generator_timesteps[i].float()
                gen_indices[i] = (schedule_timesteps.float() - gen_t).abs().argmin()
            
            # 更小的 timesteps 对应更大的 idx
            min_offset = 5

            start_indices = torch.clamp(gen_indices + min_offset, max=self.max_step)
            end_indices = torch.full_like(start_indices, self.max_step)

            ca_indices = torch.zeros(batch_size, device=latents.device, dtype=torch.long)
            for i in range(batch_size):
                if start_indices[i] < end_indices[i]:
                    ca_indices[i] = torch.randint(start_indices[i].item(), end_indices[i].item() + 1, (1,), device=latents.device)
                else:
                    ca_indices[i] = end_indices[i]

            ca_timesteps = schedule_timesteps[ca_indices]
            ca_sigmas = schedule_sigmas[ca_indices]
            
            # Sample DM timesteps: uniformly from full range
            u = compute_density_for_timestep_sampling(
                weighting_scheme='logit_normal',
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            dm_indices = (u * num_steps).long().clamp(self.min_step, self.max_step)
            dm_timesteps = schedule_timesteps[dm_indices]
            dm_sigmas = schedule_sigmas[dm_indices]

            ca_sigmas = ca_sigmas.view(batch_size, 1, 1, 1)
            dm_sigmas = dm_sigmas.view(batch_size, 1, 1, 1)
            
            # Add independent noise for CA and DM
            ca_noise = torch.randn_like(latents)
            dm_noise = torch.randn_like(latents)
            ca_noisy_latents = (1.0 - ca_sigmas) * latents + ca_sigmas * ca_noise
            dm_noisy_latents = (1.0 - dm_sigmas) * latents + dm_sigmas * dm_noise
            
            ca_current_timesteps = ca_timesteps.float()
            dm_current_timesteps = dm_timesteps.float()
            
            # ===== DM: Fake model prediction (without CFG, guidance_scale=1.0) =====
            pred_dm_fake_cond_noise = predict_noise(
                self.fake_unet, dm_noisy_latents, text_embedding, uncond_embedding, 
                dm_current_timesteps, 
                guidance_scale=self.fake_guidance_scale,
                pooled_prompt_embeds=pooled_prompt_embeds,
                uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
                decoupled_dmd=False
            )
            pred_dm_fake_cond_image = (dm_noisy_latents - dm_sigmas * pred_dm_fake_cond_noise).to(dm_noisy_latents.dtype)
            
            # ===== DM: Real model prediction (with guidance) =====
            self.real_guidance_scale = random.uniform(2, 8)
            if self.use_bf16:
                pred_dm_real_cond_noise, pred_dm_real_uncond_noise = predict_noise(
                    self.real_unet, dm_noisy_latents.to(torch.bfloat16), 
                    text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    dm_current_timesteps, 
                    guidance_scale=self.real_guidance_scale,
                    pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
                    uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds.to(torch.bfloat16),
                    decoupled_dmd=True
                )
            else:
                pred_dm_real_cond_noise, pred_dm_real_uncond_noise = predict_noise(
                    self.real_unet, dm_noisy_latents, text_embedding, uncond_embedding, 
                    dm_current_timesteps, 
                    guidance_scale=self.real_guidance_scale,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
                    decoupled_dmd=True
                )
            # pred_dm_real_noise = pred_dm_real_uncond_noise + self.real_guidance_scale * (pred_dm_real_cond_noise - pred_dm_real_uncond_noise)
            pred_dm_real_cond_image = (dm_noisy_latents - dm_sigmas * pred_dm_real_cond_noise).to(dm_noisy_latents.dtype)
            pred_dm_real_uncond_image = (dm_noisy_latents - dm_sigmas * pred_dm_real_uncond_noise).to(dm_noisy_latents.dtype)
            # ===== CA: Real model CFG predictions =====
            if self.use_bf16:
                pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
                    self.real_unet, ca_noisy_latents.to(torch.bfloat16), 
                    text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    ca_current_timesteps, 
                    guidance_scale=self.real_guidance_scale,
                    pooled_prompt_embeds=pooled_prompt_embeds.to(torch.bfloat16),
                    uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds.to(torch.bfloat16),
                    decoupled_dmd=True
                )
            else:
                pred_ca_real_cond_noise, pred_ca_real_uncond_noise = predict_noise(
                    self.real_unet, ca_noisy_latents, text_embedding, uncond_embedding, 
                    ca_current_timesteps, 
                    guidance_scale=self.real_guidance_scale,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds,
                    decoupled_dmd=True
                )
            
            pred_ca_real_cond_image = (ca_noisy_latents - ca_sigmas * pred_ca_real_cond_noise).to(ca_noisy_latents.dtype)
            pred_ca_real_uncond_image = (ca_noisy_latents - ca_sigmas * pred_ca_real_uncond_noise).to(ca_noisy_latents.dtype)
            
            # ===== Compute gradients =====
            # DM gradient
            dm_update_vector = (self.real_guidance_scale - 1) * (pred_dm_real_cond_noise - pred_dm_fake_cond_noise)
            dm_norm_factor = (pred_dm_real_cond_noise - pred_dm_fake_cond_noise).abs().mean(dim=[1, 2, 3], keepdim=True)
            dm_update_vector = dm_update_vector / (dm_norm_factor + 1e-8)
            
            # CA gradient (classifier-free alignment)
            ca_update_vector = (pred_ca_real_cond_noise - pred_ca_real_uncond_noise)
            ca_norm_factor = (pred_ca_real_cond_noise - pred_ca_real_uncond_noise).abs().mean(dim=[1, 2, 3], keepdim=True)
            ca_update_vector = ca_update_vector / (ca_norm_factor + 1e-8)
        
        # Compute losses
        loss_dm = 0.5 * F.mse_loss(pred_generator_image.float(), (pred_generator_image - dm_update_vector).detach().float(), reduction="mean")
        loss_ca = 0.5 * F.mse_loss(pred_generator_image.float(), (pred_generator_image - ca_update_vector).detach().float(), reduction="mean")
        loss_decoupled = loss_dm + loss_ca
        
        loss_dict = {
            "loss_decoupled": loss_decoupled,
            "loss_dm": loss_dm,
            "loss_ca": loss_ca,
        }
        
        decoupled_log_dict = {
            "ca_noisy_latents": ca_noisy_latents.detach().float(),
            "dm_noisy_latents": dm_noisy_latents.detach().float(),
            "pred_ca_real_cond_image": pred_ca_real_cond_image.detach().float(),
            "pred_ca_real_uncond_image": pred_ca_real_uncond_image.detach().float(),
            "pred_dm_real_cond_image": pred_dm_real_cond_image.detach().float(),
            "pred_dm_fake_cond_image": pred_dm_fake_cond_image.detach().float(),
            "ca_update_vector": ca_update_vector.detach().float(),
            "dm_update_vector": dm_update_vector.detach().float(),
            "ca_norm_factor": ca_norm_factor.detach().float().mean(),  # 记录平均值
            "dm_norm_factor": dm_norm_factor.detach().float().mean(),  # 记录平均值
            "ca_update_norm": torch.norm(ca_update_vector).item(),  # 添加更新向量的范数
            "dm_update_norm": torch.norm(dm_update_vector).item(),  # 添加更新向量的范数
            "ca_timesteps_mean": ca_timesteps.float().mean().item(),  # CA 时间步平均值
            "dm_timesteps_mean": dm_timesteps.float().mean().item(),  # DM 时间步平均值
        }
        
        return loss_dict, decoupled_log_dict
    
    def compute_loss_fake(
        self,
        latents,
        text_embedding,
        uncond_embedding,
        pooled_prompt_embeds,
        uncond_pooled_prompt_embeds
    ):
        latents = latents.detach()
        batch_size = latents.shape[0]
        u = compute_density_for_timestep_sampling(
                    weighting_scheme='logit_normal',
                    batch_size=batch_size,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,  # Only used when weighting_scheme is 'mode'
                )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noise = torch.randn_like(latents)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        weighting = compute_loss_weighting_for_sd3(weighting_scheme='logit_normal', sigmas=sigmas)

        with self.network_context_manager:
            fake_noise_pred = predict_noise(
                self.fake_unet, noisy_latents, text_embedding, uncond_embedding, 
                timesteps, guidance_scale=1,
                pooled_prompt_embeds=pooled_prompt_embeds,
                uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds
            )

        fake_noise_pred = fake_noise_pred.float()
        target = noise - latents
        loss_fake = torch.mean(
            (weighting.float() * (fake_noise_pred - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss_fake = loss_fake.mean()

        fake_x0_pred = (noisy_latents - sigmas * fake_noise_pred).to(noisy_latents.dtype)

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_latents.detach().float(),
            "faketrain_x0_pred": fake_x0_pred.detach().float()
        }
        return loss_dict, fake_log_dict


    def generator_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        pooled_prompt_embeds=None,
        uncond_pooled_prompt_embeds=None,
        timesteps=None,
    ):
        loss_dict = {}
        log_dict = {}

        # image.requires_grad_(True)
        # if not self.gan_alone:
        if self.use_decoupled_dmd:
            dm_dict, dm_log_dict = self.compute_decoupled_distribution_matching_loss(
                image, text_embedding, uncond_embedding, 
                pooled_prompt_embeds, uncond_pooled_prompt_embeds,
                generator_timesteps=timesteps
            )
        else:
            raise NotImplementedError("Decoupled DMD not enabled")

        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        # if self.cls_on_clean_image:
        #     clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
        #         image, text_embedding, pooled_prompt_embeds
        #     )
        #     loss_dict.update(clean_cls_loss_dict)

        return loss_dict, log_dict 
    
    def guidance_forward(
        self,
        image,
        text_embedding,
        uncond_embedding,
        real_train_dict=None,
        pooled_prompt_embeds=None,
        uncond_pooled_prompt_embeds=None
    ):
        fake_dict, fake_log_dict = self.compute_loss_fake(
            image, text_embedding, uncond_embedding,
            pooled_prompt_embeds=pooled_prompt_embeds,
            uncond_pooled_prompt_embeds=uncond_pooled_prompt_embeds
        )

        loss_dict = fake_dict 
        log_dict = fake_log_dict

        return loss_dict, log_dict 

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None,
        guidance_data_dict=None
    ):
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                image=generator_data_dict["image"],
                text_embedding=generator_data_dict["text_embedding"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                pooled_prompt_embeds=generator_data_dict["pooled_prompt_embeds"],
                uncond_pooled_prompt_embeds=generator_data_dict["uncond_pooled_prompt_embeds"],
                timesteps=generator_data_dict.get("timesteps", None)
            )
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                image=guidance_data_dict["image"],
                text_embedding=guidance_data_dict["text_embedding"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                pooled_prompt_embeds=guidance_data_dict["pooled_prompt_embeds"],
                uncond_pooled_prompt_embeds=guidance_data_dict["uncond_pooled_prompt_embeds"]
            )
        else:
            raise NotImplementedError

        return loss_dict, log_dict 
        
