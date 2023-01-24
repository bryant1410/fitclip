#!/usr/bin/env python
import argparse

import torch
from cached_path import cached_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", metavar="INPUT_FILE", type=cached_path)
    parser.add_argument("output_path", metavar="OUTPUT_FILE")
    parser.add_argument("--prefix", default="encoder.model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input_path)

    prefix = args.prefix + ("" if args.prefix.endswith(".") else ".")
    checkpoint["state_dict"] = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}

    torch.save(checkpoint, args.output_path)


if __name__ == "__main__":
    main()
