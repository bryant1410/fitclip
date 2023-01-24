#!/usr/bin/env python
import argparse

import pandas as pd

from util.argparse_with_defaults import ArgumentParserWithDefaults


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("path1", metavar="FILE1")
    parser.add_argument("path2", metavar="FILE2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df1 = pd.read_csv(args.path1)
    df2 = pd.read_csv(args.path2)

    # From https://stackoverflow.com/a/48647840/1165181
    print(pd.concat([df1, df2]).drop_duplicates(keep=False).to_csv(index=False))


if __name__ == "__main__":
    main()
