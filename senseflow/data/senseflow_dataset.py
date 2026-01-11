"""
SenseFlow dataset classes for training.
"""

import json
import io
import pickle
from typing import Optional

import numpy as np
import torch
import lmdb
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, shape, row_index):
    """Retrieve a specific row from a specific array in the LMDB."""
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    array = np.frombuffer(row_bytes, dtype=dtype)
    
    if len(shape) > 0:
        array = array.reshape(shape)
    return array


def get_array_shape_from_lmdb(lmdb_env, array_name):
    """Get the shape of an array stored in LMDB."""
    with lmdb_env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))

    return image_shape


class SDImageDatasetLMDB(Dataset):
    """Dataset for loading SD latents from LMDB with tokenization."""
    
    def __init__(self, dataset_path, tokenizer_one, is_sdxl=False, tokenizer_two=None):
        self.KEY_TO_TYPE = {
            'latents': np.float16
        }
        self.is_sdxl = is_sdxl  # SDXL uses two tokenizers
        self.dataset_path = dataset_path
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.latent_shape = get_array_shape_from_lmdb(self.env, "latents")

        self.length = self.latent_shape[0]

        print(f"Dataset length: {self.length}")
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = retrieve_row_from_lmdb(
            self.env, 
            "latents", self.KEY_TO_TYPE['latents'], self.latent_shape[1:], idx
        )
        image = torch.tensor(image, dtype=torch.float32)

        with self.env.begin() as txn:
            prompt = txn.get(f'prompts_{idx}_data'.encode()).decode()

        text_input_ids_one = self.tokenizer_one(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        output_dict = { 
            'images': image,
            'text_input_ids_one': text_input_ids_one,
        }

        if self.is_sdxl:
            text_input_ids_two = self.tokenizer_two(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            output_dict['text_input_ids_two'] = text_input_ids_two

        return output_dict


class SDImageDatasetLMDBwoTokenizer(Dataset):
    """Dataset for loading SD latents from LMDB without tokenization."""
    
    def __init__(self, dataset_path='/mnt/basemodel_afsv1/gexingtong/DMD2/data/sdxl_vae_latents_laion_500k_lmdb/'):
        self.KEY_TO_TYPE = {
            'latents': np.float16
        }
        self.dataset_path = dataset_path

        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.latent_shape = get_array_shape_from_lmdb(self.env, "latents")

        self.length = self.latent_shape[0]

        print(f"Dataset length: {self.length}")
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = retrieve_row_from_lmdb(
            self.env, 
            "latents", self.KEY_TO_TYPE['latents'], self.latent_shape[1:], idx
        )
        image = torch.tensor(image, dtype=torch.float32)
        
        with self.env.begin() as txn:
            prompt = txn.get(f'prompts_{idx}_data'.encode()).decode()
        
        orig_size = (1024, 1024)
        crop_coords = (0, 0)
        return {
            'images': image,
            'caption': prompt,
            'orig_size': orig_size,
            'crop_coords': crop_coords
        }


class LaionText2ImageDataset(Dataset):
    """Dataset for loading LAION images from JSON file."""
    
    def __init__(
        self,
        json_path: str,
        resolution: int = 1024,
        repeat: int = 1,
    ):
        """
        Initialize dataset.
        
        Args:
            json_path: Path to JSON file containing samples.
            resolution: Image resolution.
            repeat: Number of times to repeat each sample.
        """
        # Load sample data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.keys = data.get('keys', [])
        self.image_paths = data.get('image_paths', [])
        self.prompts = data.get('prompts', [])
        self.resolution = resolution
        self.repeat = repeat

        # Check data consistency
        print('dataset len: ', len(self.keys), len(self.image_paths), len(self.prompts))
        assert len(self.keys) == len(self.image_paths) == len(self.prompts), "Inconsistent lengths in dataset!"

        # Repeat samples to expand dataset size
        self.keys = self.keys * self.repeat
        self.image_paths = self.image_paths * self.repeat
        self.prompts = self.prompts * self.repeat

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Get data sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Sample data including image and caption
        """
        # Get image path and corresponding text
        img_path = self.image_paths[index]
        prompt = self.prompts[index]

        image = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = TF.normalize(image, [0.5], [0.5])

        return {
            'image': image,
            'caption': prompt
        }


def cycle(dl):
    """Cycle through a DataLoader indefinitely."""
    while True:
        for data in dl:
            yield data

