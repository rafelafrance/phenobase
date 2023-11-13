#!/usr/bin/env python3
import argparse
import random
import textwrap
from pathlib import Path

import pandas as pd
from pylib import log


def main():
    log.started()
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.trait_csv).set_index("file")
    df = df.drop(["time", "whole_plant", "reproductive_structure"], axis="columns")

    triples = {  # flowers, fruits, leaves
        "none": ["0", "0", "0"],
        "flowers": ["1", "0", "0"],
        "fruits": ["0", "1", "0"],
        "leaves": ["0", "0", "1"],
        "flowers_fruits": ["1", "1", "0"],
        "flowers_leaves": ["1", "0", "1"],
        "fruits_leaves": ["0", "1", "1"],
        "all": ["1", "1", "1"],
    }

    splits = {}
    for label, v in triples.items():
        splits[label] = df[
            (df["flowers"] == v[0]) & (df["fruits"] == v[1]) & (df["leaves"] == v[2])
        ]

    splits2 = {}
    for label in ("flowers", "fruits", "leaves"):
        splits2[label] = df[df[label] == "1"]
        splits2["no_" + label] = df[df[label] == "0"]

    print(df.head())
    print(df.shape)

    for key, df1 in splits.items():
        print(key, df1.shape)

    print()

    for key, df1 in splits2.items():
        print(key, df1.shape)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent(
            """Split the data into training, testing, & validation datasets."""
        ),
    )

    arg_parser.add_argument(
        "--trait-csv",
        "-t",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read traits from this CSV file.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        "-i",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Images for the classifications are stored in this directory.""",
    )

    arg_parser.add_argument(
        "--train-split",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the validation. I.e. evaluating
            training progress at the end of each epoch. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for the testing. I.e. the holdout
            data used to evaluate the model after training. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=234987,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
