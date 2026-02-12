## 📄 Paper Info

### SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation

[![arXiv](https://img.shields.io/badge/Arxiv-2506.00523-b31b1b)](https://arxiv.org/abs/2506.00523)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/domiso/SenseFlow)

[Xingtong Ge](https://xingtongge.github.io/)<sup>1,2</sup>, Xin Zhang<sup>2</sup>, [Tongda Xu](https://tongdaxu.github.io/)<sup>3</sup>, [Yi Zhang](https://zhangyi-3.github.io/)<sup>4</sup>, [Xinjie Zhang](https://xinjie-q.github.io/)<sup>1</sup>, [Yan Wang](https://yanwang202199.github.io/)<sup>3</sup>, [Jun Zhang](https://eejzhang.people.ust.hk/)<sup>1</sup>

<sup>1</sup>HKUST, <sup>2</sup>SenseTime Research, <sup>3</sup>Tsinghua University, <sup>4</sup>CUHK MMLab

## 🚀 Model Weights and Quick Inference

Can be found in [README-og.md/Model Weight & Quick Start with SenseFlow-FLUX](README-og.md)

## 💻 Installation

Two methods to set up the environment: using conda with `environment.yaml` or using pip with `requirements.txt`.

1. Create a new conda environment from the provided `environment.yaml`:
   ```bash
   cd ./SenseFlow
   conda env create -f environment.yaml
   conda activate senseflow
   pip install -e .
   ```

2. Create a new virtual environment by yourself (Python 3.10 is required):
   ```bash
   cd ./SenseFlow
   conda create -n senseflow python=3.10 -y
   conda activate senseflow
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   pip install -e .
   ```

If you are experiment with SD model, you are likely to have to install the following dependency:
1. dnnlib
   ```bash
   pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl
   ```
2. ldm
   ```bash
   git clone https://github.com/CompVis/latent-diffusion.git
   ```
   And make sure you write the following command in trainer code:
   ```python
   import sys
   sys.path.append("path/to/latent-diffusion/repo")
   ```

## ⚙️ Setup

**Note**: You can just using the simple ALL in one command, and config your path in trainer after that:
```bash
bash ./exp_sd35/download_sd35.sh
```

For `SDXL` and `FLUX` please refer to [README-og.md/Setup](README-og.md), for `SD3.5 Medium`:

### Checkpoint Preparation
Before training, you need to download the pretrained teacher models and configure the paths in the trainer files.

1. Download Stable Diffusion 3.5 Medium from HuggingFace:
   ```python
   from huggingface_hub import snapshot_download

   save_dir = "/root/highspeedstorage/model_distill/SenseFlow/ckpt/stable-diffusion-3.5-medium"
   repo_id = "stabilityai/stable-diffusion-3.5-medium"
   cache_dir = save_dir + "/cache"

   snapshot_download(cache_dir=cache_dir,
   repo_type="model",
   local_dir=save_dir,
   repo_id=repo_id,
   local_dir_use_symlinks=False,
   resume_download=True,
   )
   ```

2. Update the path in trainer file:
   - Open `senseflow/trainer/trainer_sd35_senseflow.py`
   - Replace `PLACEHOLDER_SD35_MEDIUM_PATH` with your local path to `stable-diffusion-3.5-medium`

### Dataset Preparation

For SD3.5 and FLUX training, you need to generate a text-image datasets with a JSON file format.

1. Download dataset, here I choose [LAION_Aesthetics_1024](https://huggingface.co/datasets/limingcv/LAION_Aesthetics_1024)
   ```python
   from huggingface_hub import snapshot_download
   save_dir = "/root/highspeedstorage/model_distill/SenseFlow/dataset/LAION_Aesthetics_1024"
   repo_id = "limingcv/LAION_Aesthetics_1024"
   cache_dir = save_dir + "/cache"

   snapshot_download(cache_dir=cache_dir,
   repo_type="dataset",
   local_dir=save_dir,
   repo_id=repo_id,
   local_dir_use_symlinks=False,
   resume_download=True,
   )
   ```
2. Prepare your dataset JSON file with the following structure:
   ```json
   {
       "keys": ["00000000", "00000001", "00000002", ...],
       "image_paths": [
           "/path/to/images/00000000.png",
           "/path/to/images/00000001.png",
           "/path/to/images/00000002.png",
           ...
       ],
       "prompts": [
           "A beautiful sunset over the ocean",
           "A cat sitting on a windowsill",
           "A modern city skyline at night",
           ...
       ]
   }
   ```
   
   **Important**: The three lists (`keys`, `image_paths`, `prompts`) must have the same length, and each index corresponds to one sample.

3. Update the dataset path in trainer files:
   - For SD3.5 Medium: Open `senseflow/trainer/trainer_sd35_senseflow.py`
   - For SD3.5 Large: Open `senseflow/trainer/trainer_sd35_large_senseflow.py`
   - For FLUX: Open `senseflow/trainer/trainer_flux_senseflow.py`
   - Replace `PLACEHOLDER_JSON_DATASET_PATH` with your local path to the JSON file

4. Ensure image paths in the JSON file are absolute paths or paths relative to where you run the training script.

## 🏋️ Training

For `SDXL`, `FLUX` and `SD3.5 Large` please refer to [README-og.md/Training](README-og.md), for `SD3.5 Medium`:

```bash
# This is a Decoupled DMD approach, replaced og trainer_sd35_senseflow
work_path=$(dirname $0)
filename=$(basename $work_path)
T=$(date +%m%d%H%M)
OMP_NUM_THREADS=1 \
PYTHONFAULTHANDLER=True \
torchrun \
--nproc_per_node 4 \
--nnodes 1 \
main_trainer_sd35_senseflow.py \
   /root/highspeedstorage/model_distill/SenseFlow/configs/SD35/sd35_senseflow.yaml \
   /root/highspeedstorage/model_distill/SenseFlow/exp_sd35/output/$T
```

You may change the training config by editing the `args` in trainer or `.yaml` file in .config folder.

## 🎨 Inference

Can be found in [README-og.md/Inference](README-og.md)

## 📈 OG-Results

### Table 1: Quantitative Results on COCO-5K Dataset

**Bold** = best, <ins>Underline</ins> = second best. All results on 4-step generation.

#### Stable Diffusion XL Comparison

| Method           | NFE | FID-T | Patch FID-T | CLIP | HPSv2 | Pick | ImageReward |
|------------------|--------|----------|----------------|--------|---------|---------|----------------|
| SDXL             | 80     | --       | --             | 0.3293 | 0.2930  | 22.67   | 0.8719         |
| LCM-SDXL         | 4      | 18.47    | 30.63          | 0.3230 | 0.2824  | 22.22   | 0.5693         |
| PCM-SDXL         | 4      | 14.38    | 17.77          | 0.3242 | 0.2920  | 22.54   | 0.6926         |
| Flash-SDXL       | 4      | 17.97    | 23.24          | 0.3216 | 0.2830  | 22.17   | 0.4295         |
| SDXL-Lightning   | 4      | **13.67**| **16.57**      | 0.3214 | 0.2931  | 22.80   | 0.7799         |
| Hyper-SDXL       | 4      | <ins>13.71</ins>  | <ins>17.49</ins>        | 0.3254 | <ins>0.3000</ins> | <ins>22.98</ins> | <ins>0.9777</ins> |
| DMD2-SDXL        | 4      | 15.04    | 18.72          | **0.3277** | 0.2963 | <ins>22.98</ins> | 0.9324         |
| Ours-SDXL        | 4      | 17.76    | 21.01          | 0.3248 | **0.3010** | **23.17** | **0.9951** |

#### Stable Diffusion 3.5 Comparison

| Method               | NFE | FID-T | Patch FID-T | CLIP | HPSv2 | Pick | ImageReward |
|----------------------|--------|----------|----------------|--------|---------|---------|----------------|
| SD 3.5 Large         | 100    | --       | --             | 0.3310 | 0.2993  | 22.98   | 1.1629         |
| SD 3.5 Large Turbo   | 4      | <ins>13.58</ins>  | 22.88          | 0.3262 | 0.2909  | 22.89   | 1.0116         |
| Ours-SD 3.5          | 4      | **13.38**| **17.48**      | <ins>0.3286</ins> | **0.3016** | **23.01** | <ins>1.1713</ins> |
| Ours-SD 3.5 (Euler)  | 4      | 15.24    | <ins>20.26</ins>        | **0.3287** | <ins>0.3008</ins> | <ins>22.90</ins> | **1.2062** |

#### FLUX Comparison

| Method            | NFE | FID-T | Patch FID-T | CLIP | HPSv2 | Pick | ImageReward |
|-------------------|--------|----------|----------------|--------|---------|---------|----------------|
| FLUX.1 dev        | 50     | --       | --             | 0.3202 | 0.3000  | 23.18   | 1.1170         |
| FLUX.1 dev        | 25     | --       | --             | 0.3207 | 0.2986  | 23.14   | 1.1063         |
| FLUX.1-schnell    | 4      | --       | --             | **0.3264** | 0.2962 | 22.77   | 1.0755         |
| Hyper-FLUX        | 4      | <ins>11.24</ins>  | 23.47          | <ins>0.3238</ins> | 0.2963  | 23.09   | <ins>1.0983</ins> |
| FLUX-Turbo-Alpha  | 4      | **11.22**| 24.52          | 0.3218 | 0.2907  | 22.89   | 1.0106         |
| Ours-FLUX         | 4      | 15.64    | **19.60**      | 0.3167 | <ins>0.2997</ins> | <ins>23.13</ins> | 1.0921         |
| Ours-FLUX (Euler) | 4      | 16.50    | <ins>20.29</ins>        | 0.3171 | **0.3008** | **23.26** | **1.1424**     |

