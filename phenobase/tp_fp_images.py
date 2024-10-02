#!/usr/bin/env python

import argparse
import shutil
import textwrap
from pathlib import Path

import pandas as pd
from pylib import log, util
from pylib.binary_metrics import Metrics


def main():
    log.started()
    args = parse_args()

    df = pd.read_csv(args.score_csv)
    df1 = df.loc[(df["checkpoint"] == args.checkpoint) & (df["trait"] == args.trait)]
    metrics = Metrics(y_true=df1["y_true"], y_pred=df1["y_pred"], ids=df1["name"])

    # Write true positive images
    tp = metrics.get_tp(thresh_hi=args.threshold)
    args.tp_dir.mkdir(parents=True, exist_ok=True)
    for name in tp[2]:
        shutil.copy(args.image_dir / name, args.tp_dir / name)

    # Write false positive images
    fp = metrics.get_fp(thresh_hi=args.threshold)
    args.fp_dir.mkdir(parents=True, exist_ok=True)
    for name in fp[2]:
        shutil.copy(args.image_dir / name, args.fp_dir / name)

    log.finished()


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Output true positive & false positive sheet images for a trait model."""
        ),
    )

    arg_parser.add_argument(
        "--score-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""This CSV contains all of the test herbarium sheets scored using all of
            the model checkpoints.""",
    )

    arg_parser.add_argument("--checkpoint", required=True, help="""Use this model.""")

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        required=True,
        help="""The trait to examine.""",
    )

    arg_parser.add_argument(
        "--threshold", type=float, required=True, help="""The threshold."""
    )

    arg_parser.add_argument(
        "--image-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Copy images from this directory.""",
    )

    arg_parser.add_argument(
        "--tp-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Copy true positive images to this directory.""",
    )

    arg_parser.add_argument(
        "--fp-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Copy false positive images to this directory.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
