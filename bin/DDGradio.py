# This demo needs to be run from the repo folder.
# python demo/fake_gan/run.py
import os
import random
import gradio as gr
import itertools
from PIL import Image, ImageFont, ImageDraw
import sys

sys.path.append("source")

import DirectedDiffusion



# prompt
# boundingbox
# prompt indices for region
# number of trailing attention
# number of DD steps
# gaussian coefficient
# seed
EXAMPLES = [
    [
        "A painting of a tiger, on the wall in the living room",
        "0.2,0.6,0.0,0.5",
        "1,5",
        5,
        15,
        1.0,
        2094889,
    ],
    [
        "a dog diving into a pool in sunny day",
        "0.0,0.5,0.0,0.5",
        "1,2",
        10,
        20,
        5.0,
        2483964026826,
    ],
    [
        "A red cube above a blue sphere",
        "0.4,0.7,0.0,0.5 0.4,0.7,0.5,1.0",
        "2,3 6,7",
        10,
        20,
        1.0,
        1213698,
    ],
]


model_bundle = DirectedDiffusion.AttnEditorUtils.load_all_models(
    model_path_diffusion="assets/models/stable-diffusion-v1-4"
)


def directed_diffusion(
    in_prompt,
    in_bb,
    in_token_ids,
    in_slider_trailings,
    in_slider_ddsteps,
    in_slider_gcoef,
    in_seed,
    is_draw_bbox,
):
    str_arg_to_val = lambda arg, f: [
        [f(b) for b in a.split(",")] for a in arg.split(" ")
    ]
    roi = str_arg_to_val(in_bb, float)
    attn_editor_bundle = {
        "edit_index": str_arg_to_val(in_token_ids, int),
        "roi": roi,
        "num_trailing_attn": [in_slider_trailings] * len(roi),
        "num_affected_steps": in_slider_ddsteps,
        "noise_scale": [in_slider_gcoef] * len(roi),
    }
    img = DirectedDiffusion.Diffusion.stablediffusion(
        model_bundle,
        attn_editor_bundle=attn_editor_bundle,
        guidance_scale=7.5,
        prompt=in_prompt,
        steps=50,
        seed=in_seed,
        is_save_attn=False,
        is_save_recons=False,
    )
    if is_draw_bbox and in_slider_ddsteps > 0:
        for r in roi:
            x0, y0, x1, y1 = [int(r_ * 512) for r_ in r]
            image_editable = ImageDraw.Draw(img)
            image_editable.rectangle(
                xy=[x0, x1, y0, y1], outline=(255, 0, 0, 255), width=5
            )

    return img


def run_it(
    in_prompt,
    in_bb,
    in_token_ids,
    in_slider_trailings,
    in_slider_ddsteps,
    in_slider_gcoef,
    in_seed,
    is_draw_bbox,
    is_grid_search,
    progress=gr.Progress(),
):

    num_affected_steps = [in_slider_ddsteps]
    noise_scale = [in_slider_gcoef]
    num_trailing_attn = [in_slider_trailings]
    if is_grid_search:
        num_affected_steps = [5, 10]
        noise_scale = [1.0, 1.5, 2.5]
        num_trailing_attn = [10, 20, 30, 40]

    param_list = [num_affected_steps, noise_scale, num_trailing_attn]
    param_list = list(itertools.product(*param_list))

    results = []
    progress(0, desc="Starting...")
    for i, element in enumerate(progress.tqdm(param_list)):
        print("=========== Arguments ============")
        print("Prompt:", in_prompt)
        print("BoundingBox:", in_bb)
        print("Token indices:", in_token_ids)
        print("Num Trialings:", element[2])
        print("Num DD steps:", element[0])
        print("Gaussian coef:", element[1])
        print("Seed:", in_seed)
        print("===================================")
        img = directed_diffusion(
            in_prompt=in_prompt,
            in_bb=in_bb,
            in_token_ids=in_token_ids,
            in_slider_trailings=element[2],
            in_slider_ddsteps=element[0],
            in_slider_gcoef=element[1],
            in_seed=in_seed,
            is_draw_bbox=is_draw_bbox,
        )
        results.append(
            (
                img,
                "#Trailing:{},#DDSteps:{},GaussianCoef:{}".format(
                    element[2], element[0], element[1]
                ),
            )
        )
    return results


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Directed Diffusion
    Let's pin the object in the prompt as you wish!
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column(variant="compact"):
            in_prompt = gr.Textbox(
                label="Enter your prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
            ).style(
                container=False,
            )
            with gr.Row(variant="compact"):
                in_bb = gr.Textbox(
                    label="Bounding box",
                    show_label=True,
                    max_lines=1,
                    placeholder="e.g., 0.1,0.5,0.3,0.6",
                )
                in_token_ids = gr.Textbox(
                    label="Token indices",
                    show_label=True,
                    max_lines=1,
                    placeholder="e.g., 1,2,3",
                )
                in_seed = gr.Number(
                    value=2483964026821236, label="Random seed", interactive=True
                )
            with gr.Row(variant="compact"):
                is_grid_search = gr.Checkbox(
                    value=False,
                    label="Grid search? (If checked then sliders are ignored)",
                )
                is_draw_bbox = gr.Checkbox(
                    value=True,
                    label="To draw the bounding box?",
                )
            with gr.Row(variant="compact"):
                in_slider_trailings = gr.Slider(
                    minimum=0, maximum=30, value=10, step=1, label="#trailings"
                )
                in_slider_ddsteps = gr.Slider(
                    minimum=0, maximum=30, value=10, step=1, label="#DDSteps"
                )
                in_slider_gcoef = gr.Slider(
                    minimum=0, maximum=10, value=1.0, step=0.1, label="GaussianCoef"
                )
            btn = gr.Button("Generate image").style(full_width=False)

            gr.Markdown(
                """ Note:
                1) Please click one of the examples below for quick setup.
                2) if #DDsteps==0, it means the SD process runs without DD.
                """
            )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        args = [
            in_prompt,
            in_bb,
            in_token_ids,
            in_slider_trailings,
            in_slider_ddsteps,
            in_slider_gcoef,
            in_seed,
            is_draw_bbox,
            is_grid_search,
        ]
        btn.click(run_it, inputs=args, outputs=gallery)
    examples = gr.Examples(
        examples=EXAMPLES,
        inputs=args,
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
