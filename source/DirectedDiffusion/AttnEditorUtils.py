import torch
import os
import numpy as np
import torchvision
from PIL import Image
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel


def get_embeds(prompt, clip, clip_tokenizer, device="cuda"):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    embeds = clip(tokens.input_ids.to(device)).last_hidden_state
    return embeds


@torch.no_grad()
def get_image_from_latent(vae, latent):
    latent = latent / 0.18215
    image = vae.decode(latent.to(vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


@torch.no_grad()
def get_latent_from_image(vae, image, device="cuda"):
    generator = torch.cuda.manual_seed(798122)
    # Resize and transpose for numpy b h w c -> torch b c h w
    # image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0 * 2.0 - 1.0
    image = torch.from_numpy(image[np.newaxis, ...].transpose(0, 3, 1, 2))
    # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
    if image.shape[1] > 3:
        image = image[:, :3] * image[:, 3:] + (1 - image[:, 3:])
    # Move image to GPU
    image = image.to(device)
    # Encode image
    init_latent = vae.encode(image).latent_dist.sample(generator=generator) * 0.18215
    return init_latent


def load_all_models(model_path_diffusion):

    clip_tokenizer = CLIPTokenizer.from_pretrained(
        model_path_diffusion, subfolder="tokenizer"
    )
    clip_text_model = CLIPTextModel.from_pretrained(
        model_path_diffusion, subfolder="text_encoder", torch_dtype=torch.float16
    )

    # Init diffusion model
    auth_token = True  # Replace this with huggingface auth token as a string if model is not already downloaded
    # model_path_diffusion = "assets/models/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(
        model_path_diffusion,
        subfolder="unet",
        use_auth_token=auth_token,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        model_path_diffusion,
        subfolder="vae",
        use_auth_token=auth_token,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    # Move to GPU
    device = "cuda"
    unet.to(device)
    vae.to(device)
    clip_text_model.to(device)
    model_bundle = {}
    model_bundle["unet"] = unet
    model_bundle["vae"] = vae
    model_bundle["clip_tokenizer"] = clip_tokenizer
    model_bundle["clip_text_model"] = clip_text_model
    return model_bundle


@torch.no_grad()
def check_clip_score(clip_model, clip_processor, prompts=[], images=[]):
    if len(prompts) == 1:
        dim = 0
    if len(images) == 1:
        dim = 1
    inputs = clip_processor(
        text=prompts, images=images, return_tensors="pt", padding=True
    )
    inputs["pixel_values"] = torch.tensor(
        inputs["pixel_values"], dtype=clip_model.dtype, device=clip_model.device
    )
    inputs["input_ids"] = torch.tensor(inputs["input_ids"], device=clip_model.device)
    inputs["attention_mask"] = torch.tensor(
        inputs["attention_mask"], device=clip_model.device
    )
    outputs = clip_model(**inputs)
    a = clip_model.get_image_features(inputs["pixel_values"])
    b = clip_model.get_text_features(inputs["input_ids"])
    prob = torch.matmul(a, b.t()).softmax(dim=dim)
    return prob


def get_attn(unet, use=True):
    attn = []
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            if module.attn.size() == torch.Size([8, 1024, 77]):
                attn.append(module.attn)
    attn = torch.cat(attn, dim=0)
    attn = torch.sum(attn, dim=0)
    resized = torch.zeros([64, 64, 77])
    f = torchvision.transforms.Resize(size=(64, 64))
    for i in range(77):
        dim = int(np.sqrt(attn.shape[0]))
        attn_slice = attn[..., i].view(1, dim, dim)
        resized[..., i] = f(attn_slice)[0]
    return resized.cpu().numpy()


def save_attn(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            folder = "/tmp"
            filepath = os.path.join(folder, name + ".pt")
            torch.save(module.attn, filepath)
            print(filepath)


def use_add_noise(unet, level, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.use_add_noise = use
            module.noise_level = level


def use_edited_attention(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.use_edited_attn = use


def prompt_token(prompt, index):
    tokens = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    ).input_ids[0]
    return clip_tokenizer.decode(tokens[index : index + 1])
