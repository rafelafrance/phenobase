#!/usr/bin/env python3
import argparse
import textwrap
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from phenobase.pylib import log


def main():
    log.started()
    args = parse_args()

    df = pd.concat([pd.read_csv(f).set_index("file") for f in args.trait_csv])
    df = df.drop(columns=["time", "whole_plant", "reproductive_structure"])

    train, df2 = train_test_split(
        df, train_size=args.train_split, random_state=args.seed
    )
    test, val = train_test_split(df2, test_size=0.5, random_state=args.seed)
    print(len(df))
    print(f"train {len(train)} val {len(val)} test {len(test)}")
    print(f"sum {len(train) + len(val) + len(test)}")
    print()

    splits = {}

    for df2, split in ((train, "train"), (val, "val"), (test, "test")):
        for label in ("flowers", "fruits", "leaves"):
            splits[(split, label)] = len(df2[df2[label] == "1"])
            splits[(split, f"no_{label}")] = len(df2[df2[label] == "0"])

    print(f"{'split':<5} {'label':<10} {'count':<5}")
    for label in """ flowers no_flowers fruits no_fruits leaves no_leaves """.split():
        for split in ("train", "val", "test"):
            n = splits[(split, label)]
            print(f"{split:<5} {label:<10} {n:5d}")
        print()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    df = pd.concat([train, val, test])
    df.to_csv(args.split_csv)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent(
            """
            Split the data into training, testing, & validation datasets.
            """
        ),
    )

    arg_parser.add_argument(
        "--trait-csv",
        "-t",
        metavar="PATH",
        type=Path,
        required=True,
        action="append",
        help="""Read traits from this CSV file.""",
    )

    arg_parser.add_argument(
        "--split-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the splits to this CSV file.""",
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
