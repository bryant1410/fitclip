#!/usr/bin/env python
import argparse

import torch

from util.checkpoint_utils import state_dict_from_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", metavar="INPUT_FILE")
    parser.add_argument("output_path", metavar="OUTPUT_FILE")
    parser.add_argument("--prefix", default="encoder.model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state_dict = state_dict_from_checkpoint_path(args.input_path, prefix=args.prefix)

    # We don't use the logic scale from CLIP but ours, so we had deleted it. Here we need to re-create the variable,
    # so it doesn't fail when loading this `state_dict`.
    state_dict["logit_scale"] = torch.tensor(float("nan"))

    torch.save(state_dict, args.output_path)


if __name__ == "__main__":
    main()
