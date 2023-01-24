#!/usr/bin/env python
import argparse

import torch
from cached_path import cached_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", metavar="INPUT_FILE", type=cached_path)
    parser.add_argument("output_path", metavar="OUTPUT_FILE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input_path)
    state_dict = checkpoint["state_dict"]

    first_key = next(iter(state_dict))
    prefix = next(prefix for prefix in ["model", "module"] if first_key.startswith(prefix + "."))

    torch.save({k[len(prefix + "."):]: v for k, v in state_dict.items()}, args.output_path)


if __name__ == "__main__":
    main()
