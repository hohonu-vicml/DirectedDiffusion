"""
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision

import DirectedDiffusion

plt.rcParams["figure.figsize"] = [float(v)*1.5 for v in plt.rcParams["figure.figsize"]]

def plot_activation(filepath, unet, prompt, clip_tokenizer):
    a = DirectedDiffusion.AttnEditorUtils.get_attn(unet)
    splitted_prompt = prompt.split(" ")
    n = len(splitted_prompt)
    start = 0
    arrs = []
    for j in range(1):
        arr = []
        for i in range(start,start+n):
            b = a[..., i+1] / (a[..., i+1].max() + 0.001)
            arr.append(b.T)
        start += n
        arr = np.hstack(arr)
        arrs.append(arr)
    arrs = np.vstack(arrs).T
    plt.imshow(arrs, cmap='jet', vmin=0, vmax=.8)
    plt.title(prompt)
    plt.savefig(filepath)
