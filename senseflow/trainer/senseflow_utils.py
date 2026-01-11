"""
SenseFlow training utilities.
"""

import torch
import torch.nn as nn


def predict_noise(unet, noisy_latents, text_embeddings, uncond_embedding, timesteps, 
    guidance_scale=1.0, unet_added_conditions=None, uncond_unet_added_conditions=None
):
    CFG_GUIDANCE = guidance_scale > 1

    if CFG_GUIDANCE:
        model_input = torch.cat([noisy_latents] * 2) 
        embeddings = torch.cat([uncond_embedding, text_embeddings]) 
        timesteps = torch.cat([timesteps] * 2) 

        if unet_added_conditions is not None:
            assert uncond_unet_added_conditions is not None 
            condition_input = {}
            for key in unet_added_conditions.keys():
                condition_input[key] = torch.cat(
                    [uncond_unet_added_conditions[key], unet_added_conditions[key]]
                )
        else:
            condition_input = None 

        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=condition_input).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 
    else:
        model_input = noisy_latents 
        embeddings = text_embeddings
        timesteps = timesteps    
        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=unet_added_conditions).sample

    return noise_pred


def extract_into_tensor(a, t, x_shape):
    """Extract values from tensor `a` at indices `t` and reshape to match `x_shape`."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample


class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)

