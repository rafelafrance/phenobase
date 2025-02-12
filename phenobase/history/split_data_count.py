#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import pandas as pd

from phenobase.pylib import log


def main(args):
    log.started()

    df = pd.read_csv(args.split_csv)
    print(f"{'Total records':<33} {df.shape[0]:4d}")
    print()

    # Flowers before filtering
    df0 = df.loc[df["old_flowers"].isin(["0", "1"])]
    print(f"{'flower records before filtering':<33} {df0.shape[0]:4d}")

    df1 = df0.loc[df["old_flowers"] == "0"]
    print(f"{'flower negative before filtering':<33} {df1.shape[0]:4d}")

    df2 = df0.loc[df["old_flowers"] == "1"]
    print(f"{'flower positive before filtering':<33} {df2.shape[0]:4d}")

    print(f"{'percent negative':<33} {(df2.shape[0] / df0.shape[0]):0.2f}")
    print(f"{'percent positive':<33} {(df1.shape[0] / df0.shape[0]):0.2f}")
    print()

    # Flowers after filtering
    bad = pd.read_csv(args.bad_flower_families)
    old = df0.shape[0]
    df0 = df.loc[df["old_flowers"].isin(["0", "1"])]
    df0 = df0.loc[~df["family"].isin(bad["family"])]
    print(f"{'flower records after filtering':<33} {df0.shape[0]:4d}")

    df1 = df0.loc[df["old_flowers"] == "0"]
    print(f"{'flower negative after filtering':<33} {df1.shape[0]:4d}")

    df2 = df0.loc[df["old_flowers"] == "1"]
    print(f"{'flower positive after filtering':<33} {df2.shape[0]:4d}")

    print(f"{'percent kept':<33} {(df0.shape[0] / old):0.2f}")
    print(f"{'percent negative':<33} {(df1.shape[0] / df0.shape[0]):0.2f}")
    print(f"{'percent positive':<33} {(df2.shape[0] / df0.shape[0]):0.2f}")
    print()

    # Fruits before filtering
    df0 = df.loc[df["old_fruits"].isin(["0", "1"])]
    print(f"{'fruit records before filtering':<33} {df0.shape[0]:4d}")

    df1 = df0.loc[df["old_fruits"] == "0"]
    print(f"{'fruit negative before filtering':<33} {df1.shape[0]:4d}")

    df2 = df0.loc[df["old_fruits"] == "1"]
    print(f"{'fruit positive before filtering':<33} {df2.shape[0]:4d}")

    print(f"{'percent negative':<33} {(df1.shape[0] / df0.shape[0]):0.2f}")
    print(f"{'percent positive':<33} {(df2.shape[0] / df0.shape[0]):0.2f}")
    print()

    # Flowers after filtering
    bad = pd.read_csv(args.bad_fruit_families)
    old = df0.shape[0]
    df0 = df.loc[df["old_fruits"].isin(["0", "1"])]
    df0 = df0.loc[~df["family"].isin(bad["family"])]
    print(f"{'fruit records after filtering':<33} {df0.shape[0]:4d}")

    df1 = df0.loc[df["old_flowers"] == "0"]
    print(f"{'fruit negative after filtering':<33} {df1.shape[0]:4d}")

    df2 = df0.loc[df["old_flowers"] == "1"]
    print(f"{'fruit positive after filtering':<33} {df2.shape[0]:4d}")

    print(f"{'percent kept':<33} {(df0.shape[0] / old):0.2f}")
    print(f"{'percent negative':<33} {(df1.shape[0] / df0.shape[0]):0.2f}")
    print(f"{'percent positive':<33} {(df2.shape[0] / df0.shape[0]):0.2f}")
    print()

    log.finished()


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Count split data."""),
    )

    arg_parser.add_argument(
        "--split-csv",
        type=Path,
        metavar="PATH",
        help="""make sure flowers in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--bad-flower-families",
        type=Path,
        metavar="PATH",
        help="""make sure flowers in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--bad-fruit-families",
        type=Path,
        metavar="PATH",
        help="""Make sure fruits in these families are skipped.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
