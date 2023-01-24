#!/usr/bin/env python
import argparse

import torch

from aligner.encoder.clip_video_text_encoder import load_clip_model
from aligner.wise import wise_state_dict
from util.argparse_with_defaults import ArgumentParserWithDefaults


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults("Applies weight-space ensembles for fine-tuning (WiSE-FT) on 2 CLIP "
                                        "checkpoints.",
                                        description="See https://arxiv.org/abs/2109.01903 for more info.")
    parser.add_argument("input_path_or_name1", metavar="INPUT_FILE_OR_NAME_1")
    parser.add_argument("input_path_or_name2", metavar="INPUT_FILE_OR_NAME_2")
    parser.add_argument("output_path", metavar="OUTPUT_FILE")
    parser.add_argument("--weight-for-2", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model1 = load_clip_model(args.input_path_or_name1)
    model2 = load_clip_model(args.input_path_or_name2)

    # We don't use the logic scale from CLIP but ours, so we had deleted it. Here we need to re-create the variable,
    # so it doesn't fail when using the checkpoints.
    model1.logit_scale = getattr(model1, "logit_scale", torch.tensor(float("nan")))
    model2.logit_scale = getattr(model2, "logit_scale", torch.tensor(float("nan")))

    state_dict = wise_state_dict(model1, model2, weight_for_2=args.weight_for_2)

    torch.save(state_dict, args.output_path)


if __name__ == "__main__":
    main()
