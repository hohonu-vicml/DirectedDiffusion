#!/usr/bin/env python
"""Command line based runner.
"""


import sys, os

sys.path.append("/home/kma/.conda/envs/py36/lib/python3.8/site-packages")
sys.path.append("source")

from PIL import Image, ImageFont, ImageDraw
import argparse
import DirectedDiffusion
import datetime
import itertools
import textwrap
import pprint
import collections
import numpy as np
import git


def get_current_commit_hash(length=None) -> str:
    """Method to get the current commit hash of the gtm repository

    Returns:
        str
    """
    # Get the path of the root directory
    repo = git.Repo(os.getcwd())
    if length is None:
        return repo.head.commit.hexsha
    else:
        return repo.head.commit.hexsha[:length]


def get_current_branch_name():
    # TODO
    repo = git.Repo(os.getcwd())
    return repo.active_branch.name


def is_repo_dirty():
    # TODO
    repo = git.Repo(os.getcwd())
    return repo.is_dirty()


def get_args():
    """args parsing
    Args:
    Returns:
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DirectedDiffusion.ProgramInfo.get_parser_description(),
    )

    parser.add_argument(
        "-ds",
        "--diffusion-steps",
        metavar="N",
        type=int,
        nargs="+",
        default=[50],
        help="number of diffusion steps",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="./assets/experiments",
        help="output folder for generated images",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=2483964026821236,
        help="random seed. If seed value S is less than 1000 (S<1000), then we generate S seeds for you to run the DD",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="",
        help="stable diffusion prompt",
    )
    # the attributes for stable diffusion editor
    parser.add_argument(
        "-ns",
        "--num-affected-steps",
        metavar="N",
        type=int,
        nargs="+",
        default=[0],
        help=textwrap.dedent(
            """\
            A list of integer(s) of steps affected by attention editing
            from scratch (Can grid search)"""
        ),
    )
    parser.add_argument(
        "-nt",
        "--num-trailing-attn",
        metavar="N",
        type=str,
        nargs="+",
        default=[],
        help=textwrap.dedent(
            """\
            a list string(s) containing a collection of integer(s)
            associated with trailing attention maps for editing (Can grid
            search)."""
        ),
    )
    parser.add_argument(
        "-s",
        "--noise-scale",
        metavar="N",
        type=str,
        nargs="+",
        default=[],
        help=textwrap.dedent(
            """\
            a list string(s) containing a collection of
            float(s) assocaited with the noise strength in the attention maps step
            (Can grid search) """
        ),
    )
    parser.add_argument(
        "-ei",
        "--edit-index",
        metavar="N",
        type=str,
        nargs="+",
        default=[],
        help="a list of indices associated with words of prompt",
    )
    parser.add_argument(
        "-roi",
        "--region-of-interest",
        metavar="N",
        type=str,
        nargs="+",
        default=[],
        help='a list of string describing the region of interests "left,right,top,bottom",...',
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=textwrap.dedent(
            """\
        This program will only show the given
        arguments and skip to run stable diffusion to check the user input. Good
        for checking grid-search. """
        ),
    )
    parser.add_argument(
        "-m",
        "--memo",
        action="store_true",
        help=textwrap.dedent(
            """\
            It draws the metadata on the saved image so you'll remember all the attributes."""
        ),
    )
    parser.add_argument(
        "-l1",
        "--lazy-1",
        action="store_true",
        help=textwrap.dedent(
            """\
            Using default parameters and turning on all
            the memos for initial investigation (Long version)"""
        ),
    )
    parser.add_argument(
        "-l2",
        "--lazy-2",
        action="store_true",
        help=textwrap.dedent(
            """\
            Using default parameters and turning on all
            the memos for initial investigation (Long version)"""
        ),
    )
    parser.add_argument(
        "-is",
        "--is-show",
        action="store_true",
        help=textwrap.dedent(
            """\
            Whether you want to show the result automatically or not"""
        ),
    )
    parser.add_argument(
        "-n",
        "--note",
        type=str,
        default="your note",
        help="customized user note",
    )
    parser.add_argument(
        "-cp",
        "--clip-model-path",
        type=str,
        default="assets/models/clip-vit-large-patch14",
        help="clip model path",
    )
    parser.add_argument(
        "-dp",
        "--diffusion-model-path",
        type=str,
        default="assets/models/stable-diffusion-v1-4",
        help="diffusion model path",
    )

    return parser.parse_args()


def main():
    """The entry point to execute this program
    Args:
    Returns:
    """
    args = get_args()

    # # # lazy default parameter searching
    if args.lazy_1:
        args.num_affected_steps = [5, 10, 15, 20]
        args.noise_scale = [1.0, 2.0, 3.0]
        args.noise_scale = [
            ",".join(len(args.edit_index) * [str(v)]) for v in args.noise_scale
        ][:-1]
        args.num_trailing_attn = [5, 10, 15, 20, 25, 30, 35]
        args.num_trailing_attn = [
            ",".join(len(args.edit_index) * [str(v)]) for v in args.num_trailing_attn
        ][:-1]
    if args.lazy_2:
        args.num_affected_steps = [5, 10]
        args.noise_scale = [0.5, 1.0, 2.5]
        args.noise_scale = [
            ",".join(len(args.edit_index) * [str(v)]) for v in args.noise_scale
        ][:-1]
        args.num_trailing_attn = [10, 20, 30, 40]
        args.num_trailing_attn = [
            ",".join(len(args.edit_index) * [str(v)]) for v in args.num_trailing_attn
        ][:-1]

    str_arg_to_val = lambda arg, f: [[f(b) for b in a.split(",")] for a in arg]
    region_of_interest = str_arg_to_val(args.region_of_interest, float)
    edit_index = str_arg_to_val(args.edit_index, int)
    num_trailing_attn = str_arg_to_val(args.num_trailing_attn, int)
    noise_scale = str_arg_to_val(args.noise_scale, float)

    # seed
    if args.seed <= 1000:
        seeds = (np.random.sample(args.seed) * 10000000).astype(np.int32)
    else:
        seeds = [args.seed]

    param_list = [
        num_trailing_attn,
        noise_scale,
        args.num_affected_steps,
        args.diffusion_steps,
        seeds,
    ]
    param_list = list(itertools.product(*param_list))

    # TODO: pretending we have sample here to run the conventional SD
    if len(list(param_list)) == 0:
        edit_index = [[]]
        region_of_interest = [[]]
        param_list = [
            [[0]],
            [[0]],
            [0],
            args.diffusion_steps,
            seeds,
        ]
        param_list = list(itertools.product(*param_list))
    print("There are", len(param_list), "experiments will be executed.")

    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%y%m%d-%H%M%S")
    folder = args.folder
    if len(list(param_list)) > 1:
        code = "_nt{:02d}_n{:02d}_ns{:02d}_s{:02d}_sd{:02d}_".format(
            len(args.num_trailing_attn),
            len(args.noise_scale),
            len(args.num_affected_steps),
            len(args.diffusion_steps),
            len(seeds),
        )
        print(code)
        folder = os.path.join(
            folder, current_time + code + args.prompt.replace(" ", "-")
        )
        os.makedirs(folder)
        print("Folder created", folder)

    # Model loading
    model_bundle = None
    if not args.debug:
        print("Loading model....")
        model_bundle = DirectedDiffusion.AttnEditorUtils.load_all_models(
            args.diffusion_model_path
        )
        print("Loaded all models")

    for i, element in enumerate(param_list):

        n_trailing_attn = element[0]
        noise_scale = element[1]
        n_affected_steps = element[2]
        steps = element[3]
        seed = element[4]
        attn_editor_metadata = collections.OrderedDict()
        attn_editor_metadata["Experiment id"] = i
        attn_editor_metadata["Prompt"] = args.prompt
        attn_editor_metadata["Diffusion steps (G)"] = steps
        attn_editor_metadata["Num affected steps (G)"] = n_affected_steps
        attn_editor_metadata["Num trailing attn (G)"] = n_trailing_attn
        attn_editor_metadata["Noise scale (G)"] = noise_scale
        attn_editor_metadata["Edit idices"] = edit_index
        attn_editor_metadata["Region of interests"] = region_of_interest
        attn_editor_metadata["Seed"] = seed
        attn_editor_metadata["Model"] = args.diffusion_model_path
        attn_editor_metadata["Git Hexsha"] = get_current_commit_hash()
        attn_editor_metadata["Git branch"] = get_current_branch_name()
        attn_editor_metadata["Git branch dirty"] = is_repo_dirty()

        # # # Checking
        print("\n==============================================")
        pprint.pprint(attn_editor_metadata)
        print(
            "Note: (G) means such the attribute is part of grid search method. Check --help"
        )
        print("===============================================")
        assert (
            len(n_trailing_attn)
            == len(noise_scale)
            == len(region_of_interest)
            == len(edit_index)
        ), textwrap.dedent(
            """Incorrect length of given attributes of running regions. We have
                   len(edit_index)={},
                   len(n_trailing_attn)={},
                   len(noise_scale)={},
                   len(region_of_interest)={},
               and they are not matching each other.
            """.format(
                len(edit_index),
                len(n_trailing_attn),
                len(noise_scale),
                len(region_of_interest),
            )
        )

        # # # Running
        attn_editor_bundle = {
            "edit_index": edit_index,
            "roi": region_of_interest,
            "num_trailing_attn": n_trailing_attn,
            "num_affected_steps": n_affected_steps,
            "noise_scale": noise_scale,
        }
        cmd = "python ./bin/SdEditorCmd.py -roi {} -ei {} -nt {} -s {} -ns {} -ds {} -dp {} -p {} --seed {}".format(
            " ".join(args.region_of_interest),
            " ".join(args.edit_index),
            ",".join([str(s) for s in n_trailing_attn]),
            ",".join([str(s) for s in noise_scale]),
            n_affected_steps,
            steps,
            args.diffusion_model_path,
            '"' + args.prompt + '"',
            seed,
        )

        if not model_bundle:
            continue

        img = DirectedDiffusion.Diffusion.stablediffusion(
            model_bundle,
            attn_editor_bundle=attn_editor_bundle,
            guidance_scale=7.5,
            prompt=args.prompt,
            steps=steps,
            seed=seed,
        )

        # # # IO
        current_time = datetime.datetime.now()
        current_time = current_time.strftime("%y%m%d-%H%M%S")
        filename = (
            current_time
            + "_"
            + "id{:03d}".format(i)
            + "_"
            + args.prompt.replace(" ", "-")
        )
        attn_editor_metadata["Filename"] = filename
        attn_editor_metadata["Note"] = args.note
        text = "\n".join((str(item)) for item in attn_editor_metadata.items())
        if args.memo | args.lazy_1 | args.lazy_2:
            img = img.resize((1024, 1024))
            font = ImageFont.truetype("./assets/JetBrainsMono-Bold.ttf", size=13)

            image_editable = ImageDraw.Draw(img)
            for region in region_of_interest:
                x0 = region[0] * 1024
                y0 = region[2] * 1024
                x1 = region[1] * 1024
                y1 = region[3] * 1024
                image_editable.rectangle(
                    xy=[x0, y0, x1, y1], outline=(255, 0, 0, 255), width=5
                )
            image_editable.multiline_text(
                (15, 15),
                text,
                (255, 255, 255, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
                spacing=0,
            )
        if args.is_show:
            img.show()
        filepath = os.path.join(folder, filename + ".jpg")
        img.save(filepath)
        filepath = os.path.join(folder, filename + ".txt")
        f = open(filepath, "w")
        f.write(text)
        f.write("\n\n")
        f.write(cmd)
        f.close()
        print("Saved", filepath)


if __name__ == "__main__":
    main()
