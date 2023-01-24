#!/usr/bin/env python
import argparse
import sys

import torch

from util.checkpoint_utils import state_dict_from_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", metavar="INPUT_FILE")
    parser.add_argument("--prefix", default="encoder.model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = state_dict_from_checkpoint_path(args.input_path, prefix=args.prefix)
    torch.save(state_dict, sys.stdout.buffer)


if __name__ == "__main__":
    main()
