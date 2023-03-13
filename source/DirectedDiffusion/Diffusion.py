import os
import torch
import random
import numpy as np
import datetime

from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher

import DirectedDiffusion


@torch.no_grad()
def stablediffusion(
    model_bundle,
    attn_editor_bundle={},
    device="cuda",
    prompt="",
    steps=50,
    seed=None,
    width=512,
    height=512,
    t_start=0,
    guidance_scale=7.5,
    init_latents=None,
    is_save_attn=False,
    is_save_recons=False,
    folder = "./",
):

    # neural networks
    unet = model_bundle["unet"]
    vae = model_bundle["vae"]
    clip_tokenizer = model_bundle["clip_tokenizer"]
    clip = model_bundle["clip_text_model"]
    # attn editor bundle, our stuff
    num_affected_steps = int(attn_editor_bundle.get("num_affected_steps") or 0)
    if not num_affected_steps:
        print("Not using attn editor")
    else:
        print("Using attn editor")
    DirectedDiffusion.AttnCore.init_attention_edit(
        unet,
        tokens=attn_editor_bundle.get("edit_index") or [],
        rios=attn_editor_bundle.get("roi") or [],
        noise_scale=attn_editor_bundle.get("noise_scale") or [],
        length_prompt=len(prompt.split(" ")),
        num_trailing_attn=attn_editor_bundle.get("num_trailing_attn") or [],
    )

    # Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64
    # If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None:
        seed = random.randrange(2 ** 32 - 1)
    generator = torch.cuda.manual_seed(seed)
    # Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    scheduler.set_timesteps(steps)
    scheduler.timesteps = scheduler.timesteps.half().cuda()

    noise_weight = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=10,
    )
    noise_weight.set_timesteps(num_affected_steps)
    # if num_affected_steps:
    #     noise_weight.set_timesteps(num_affected_steps)
    #     noise_weight.timesteps /= torch.max(noise_weight.timesteps)

    init_latent = torch.zeros(
        (1, unet.in_channels, height // 8, width // 8), device=device
    )
    t_start = t_start
    # Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    # latent = noise * scheduler.init_noise_sigma
    latent = scheduler.add_noise(
        init_latent,
        noise,
        torch.tensor(
            [scheduler.timesteps[t_start]], device=device, dtype=torch.float16
        ),
    ).to(device)



    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%y%m%d-%H%M%S")
    folder = os.path.join(folder, current_time+"_internal")
    if not os.path.exists(folder) and (is_save_attn or is_save_recons):
        os.makedirs(folder)
    # Process clip
    with autocast(device):
        embeds_uncond = DirectedDiffusion.AttnEditorUtils.get_embeds(
            "", clip, clip_tokenizer
        )
        embeds_cond = DirectedDiffusion.AttnEditorUtils.get_embeds(
            prompt, clip, clip_tokenizer
        )
        timesteps = scheduler.timesteps[t_start:]
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, t
            ).half()
            noise_pred_uncond = unet(
                latent_model_input, t, encoder_hidden_states=embeds_uncond
            ).sample

            if i < num_affected_steps:
                DirectedDiffusion.AttnEditorUtils.use_add_noise(
                    unet, noise_weight.timesteps[i]
                )
                DirectedDiffusion.AttnEditorUtils.use_edited_attention(unet)
                noise_pred_cond = unet(
                    latent_model_input, t, encoder_hidden_states=embeds_cond
                ).sample

            else:
                noise_pred_cond = unet(
                    latent_model_input, t, encoder_hidden_states=embeds_cond
                ).sample

            delta = noise_pred_cond - noise_pred_uncond
            # Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * delta
            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

            if is_save_attn:
                filepath = os.path.join(folder, "ca.{:04d}.jpg".format(i))
                DirectedDiffusion.Plotter.plot_activation(filepath, unet, prompt, clip_tokenizer)
            if is_save_recons:
                filepath = os.path.join(folder, "recons.{:04d}.jpg".format(i))
                recons = DirectedDiffusion.AttnEditorUtils.get_image_from_latent(vae, latent)
                recons.save(filepath)
    return DirectedDiffusion.AttnEditorUtils.get_image_from_latent(vae, latent)
