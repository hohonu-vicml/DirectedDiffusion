"""
ORIGINAL SOURCE CODE
https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
"""

import torch
import math
import copy
import numpy as np

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ Gaussian weight
    Args:
       x(float): sample x
       x(float): sample x
    """
    return (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx ** 2) + (y - my) ** 2 / (2 * sy ** 2)))
    )

def init_attention_edit(
    unet, tokens=[], rios=[], num_trailing_attn=[], length_prompt=0, noise_scale=[],
):
    """ To initialize overwritting the attention module.
    """
    rios_ = copy.deepcopy(rios)
    tokens_ = copy.deepcopy(tokens)
    num_regions = len(rios)
    num_trailing_attn_ = copy.deepcopy(num_trailing_attn)
    noise_scale_ = copy.deepcopy(noise_scale)

    for i in range(num_regions):
        total_unrelated = length_prompt + 1 + num_trailing_attn[i]
        tokens_trailing = []
        for j in range(length_prompt + 1, total_unrelated):  # trailing
            if j > 76:
                break
            tokens_[i].append(j)
            # tokens_trailing.append(j)
        tokens_[i] = list(set(tokens_[i]))

    def new_attention(self, query, key, value):

        attn_slice = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_slice = attn_slice.softmax(dim=-1)

        if self.use_edited_attn and attn_slice.size()[2] == 77:

            in_tokens = tokens_[:]
            rios = rios_[:]
            num_trailing_attn = num_trailing_attn_[:]
            dim = int(np.sqrt(attn_slice.size()[1]))
            attn_slice = attn_slice.view(8, dim, dim, 77)

            # the mask for all interested words in prompt
            # global_mask = torch.ones_like(attn_slice, dtype=torch.bool)
            global_mask = torch.zeros_like(attn_slice, dtype=torch.bool)
            for i in range(num_regions):
                # region
                left = int(dim * rios[i][0])
                right = int(dim * rios[i][1])
                top = int(dim * rios[i][2])
                bottom = int(dim * rios[i][3])

                # add noise to avoid nullity
                if self.use_add_noise:
                    tmp = attn_slice[
                        :, top:bottom, left:right, in_tokens[i]
                    ].clone() * (noise_scale_[i] * self.noise_level)
                    # TODO: the uniform distribution on the region
                    w = tmp.shape[2]
                    h = tmp.shape[1]
                    # noise_u = torch.abs(torch.randn(h, w))
                    x = torch.linspace(0, h, h)
                    y = torch.linspace(0, w, w)
                    x, y = torch.meshgrid(x, y, indexing="ij")
                    noise_g = gaussian_2d(
                        x,
                        y,
                        mx=int(h / 2),
                        my=int(w / 2),
                        sx=float(h) / 2.0,
                        sy=float(w) / 2.0,
                    )
                    noise = noise_g
                    noise = (
                        noise.unsqueeze(0)
                        .unsqueeze(-1)
                        .repeat(tmp.shape[0], 1, 1, tmp.shape[-1])
                        .to(attn_slice.device)
                    )

                    attn_slice[:, top:bottom, left:right, in_tokens[i]] = tmp + noise

                scale = 0
                mask = torch.ones_like(attn_slice, dtype=torch.bool)
                mask[:, :, right:, in_tokens[i]] = False
                mask[:, :, :left, in_tokens[i]] = False
                mask[:, :top, :, in_tokens[i]] = False
                mask[:, bottom:, :, in_tokens[i]] = False
                global_mask[..., in_tokens[i]] |= mask[..., in_tokens[i]]

                mask = torch.zeros_like(attn_slice, dtype=torch.bool)
                mask[:, top:bottom, :, in_tokens[i]] = True
                mask[:, :, left:right, in_tokens[i]] = True
                global_mask[..., in_tokens[i]] &= mask[..., in_tokens[i]]

            zeros_indices = torch.where(global_mask == False)
            # global_mask = torch.tensor(
            #     global_mask.clone().detach(), dtype=torch.float16
            # )
            global_mask = global_mask.clone().detach().half()
            global_mask[zeros_indices] = 0.01
            attn_slice *= global_mask
            attn_slice = attn_slice.view(8, dim * dim, 77)
            self.use_add_noise = False
            self.use_edited_attn = False

        self.attn = attn_slice.clone()
        hidden_states = torch.matmul(attn_slice, value)
        # reshape hidden_states, e.g., torch.Size([1, 4096, 320])
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.use_edited_attn = False
            module.save_last_attn_slice = False
            module.use_add_noise = False
            module.noise_level = 0.0
            module.attn = None
            module._attention = new_attention.__get__(module, type(module))
