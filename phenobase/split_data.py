#!/usr/bin/env python3
import argparse
import random
import textwrap
from pathlib import Path

import pandas as pd
from pylib import log
from tqdm import tqdm


def main():
    log.started()

    args = parse_args()

    random.seed(args.seed)

    df = pd.read_parquet(args.observations)
    df["split"] = None

    # print(df["reproductiveCondition"].unique())
    df = df[df["reproductiveCondition"].notna()]
    df = df[df["reproductiveCondition"] != "flowering|no evidence of flowering"]

    def flowering(rc):
        flags = [" ".join(f.split()) for f in rc.split("|")]
        return 1 if "flowering" in flags else 0

    def fruiting(rc):
        flags = [" ".join(f.split()) for f in rc.split("|")]
        return 1 if "fruiting" in flags else 0

    df["flowering"] = df["reproductiveCondition"].map(flowering).astype(int)
    df["fruiting"] = df["reproductiveCondition"].map(fruiting).astype(int)

    grouped = df.groupby(["order", "flowering", "fruiting"])
    for repro, group in tqdm(grouped):
        index = group.index.values
        random.shuffle(index)

        count = len(group)
        test = max(1, round(count * args.test_split))
        valid = test + max(1, round(count * args.val_split)) if count > 1 else 0

        for i in range(count):
            if i < test:
                split = "test"
            elif i < valid:
                split = "val"
            else:
                split = "train"
            df.at[index[i], "split"] = split

    df.to_parquet(args.observations)

    print(df.head(10))
    print(df.shape)

    log.finished()


def parse_args():
    description = """
        Split the data into training, testing, & validation data sets.
        """

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--observations",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read from and write to this parquet file containing observations.""",
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
