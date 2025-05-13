#!/usr/bin/env python3

import argparse
import shutil
import textwrap
from pathlib import Path
from random import seed

import pandas as pd

from phenobase.pylib import log


def main(args):
    log.started(args=args)
    seed(args.seed)

    df = pd.read_csv(args.winners_csv, low_memory=False)
    df = df.sample(frac=1.0).reset_index(drop=True)
    print(df.shape)

    if args.exclude_csv:
        exclude = pd.read_csv(args.exclude_csv, usecols=["path"])
        df = df.loc[~df["path"].isin(exclude["path"])]
        print(df.shape)

    make_script = args.image_bash and args.image_dir

    if make_script:
        src = Path() / "args" / "slurm" / "template.bash"
        shutil.copy(src, args.image_bash)
        with args.image_bash.open("a") as f:
            f.write(f"mkdir -p {args.image_dir}\n\n")

    if args.sample_positive:
        pos_df = df.loc[df["winner"] == 1]
        pos_df = pos_df.iloc[: args.sample_positive]
        path = args.output_sample.with_stem(f"{args.output_sample.stem}_pos")
        pos_df.to_csv(path, index=False)
        if make_script:
            pos_df["cp"] = "cp " + pos_df["path"] + " " + str(args.image_dir)
            pos_df["cp"].to_csv(args.image_bash, index=False, mode="a", header=False)

    if args.sample_negative:
        neg_df = df.loc[df["winner"] == 0]
        neg_df = neg_df.iloc[: args.sample_negative]
        path = args.output_sample.with_stem(f"{args.output_sample.stem}_neg")
        neg_df.to_csv(path, index=False)
        if make_script:
            neg_df["cp"] = "cp " + neg_df["path"] + " " + str(args.image_dir)
            neg_df["cp"].to_csv(args.image_bash, index=False, mode="a", header=False)

    if make_script:
        with args.image_bash.open("a") as f:
            f.write(f"\ntar czf {args.image_dir}.tgz {args.image_dir}\n\n")

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Sample inferred ensemble output so that an expert can score it.
            """
        ),
    )

    arg_parser.add_argument(
        "--winners-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""The CSV file that contains all of the ensemble winners.""",
    )

    arg_parser.add_argument(
        "--exclude-csv",
        type=Path,
        metavar="PATH",
        help="""Don't include any records that are in this file.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        help="""Output images to this directory. This directory is on HiPerGator""",
    )

    arg_parser.add_argument(
        "--image-bash",
        type=Path,
        metavar="PATH",
        help="""Create a bash script to move the images into the --image-dir.""",
    )

    arg_parser.add_argument(
        "--output-sample",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output sampled records to this CSV file.""",
    )

    arg_parser.add_argument(
        "--sample-positive",
        type=int,
        default=0,
        metavar="INT",
        help="""How many positive records to sample. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--sample-negative",
        type=int,
        default=0,
        metavar="INT",
        help="""How many negative records to sample. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        default=4049927,
        metavar="INT",
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
