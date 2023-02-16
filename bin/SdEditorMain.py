"""
The raw code for different purposes different from the command line file
e.g., you could do "python -i" and keep the model alive in a session
"""
import sys
sys.path.append("source")
import torch
import DirectedDiffusion
import importlib
importlib.reload(DirectedDiffusion)

from PIL import Image, ImageFont, ImageDraw

model_bundle = DirectedDiffusion.AttnEditorUtils.load_all_models(
    model_path_clip = 'assets/models/clip-vit-large-patch14',
    model_path_diffusion = 'assets/models/stable-diffusion-v1-4'
)

prompt = "A cat sitting on a car"
attn_editor_bundle = {
    'edit_index':[[1,2]],
    'roi':[(0.0, .5, 0., 1.)],
    'num_trailing_attn':[5],
    'num_affected_steps':0,
    'noise_scale':[.5],
}
img = DirectedDiffusion.Diffusion.stablediffusion(
        model_bundle,
        attn_editor_bundle=attn_editor_bundle,
        guidance_scale=7.5,
        prompt=prompt,
        steps=50,
        seed=248396402730,
        is_save_attn=True,
        is_save_recons=True,
    )
img.show()



#text = ('\n'.join((str(item)) for item in attn_editor_bundle.items()))
# font = ImageFont.truetype('/usr/local/share/fonts/ttf/JetBrainsMono-Bold.ttf', size=12)
# img = Image.new('RGB', (640, 480), color=(128,128,128,255))
# image_editable = ImageDraw.Draw(img)
# image_editable.multiline_text((15,15), text, (255, 255, 255, 0), font=font, stroke_width=2, stroke_fill=(0,0,0,255))
# img.show()
