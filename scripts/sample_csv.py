#!/usr/bin/env python
import argparse
import sys

import pandas as pd

from util.argparse_with_defaults import ArgumentParserWithDefaults


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("path", metavar="FILE", nargs="?", default="-")
    parser.add_argument("-s", "--size", type=int, default=10)
    args = parser.parse_args()

    args.path = sys.stdin if args.path == "-" else args.path

    return args


def main() -> None:
    args = parse_args()
    print(pd.read_csv(args.path).sample(args.size).to_csv(index=False))


if __name__ == "__main__":
    main()
