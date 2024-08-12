#!/usr/bin/env python3

import argparse
import logging
import random
import textwrap
from pathlib import Path

import pandas as pd

from phenobase.pylib import log, util


def main():
    log.started()
    args = parse_args()

    records = get_expert_data(args.ant_csv)
    split_data(records, args.seed, args.train_split, args.eval_split)
    write_csv(args.split_csv, records)

    log.finished()


def get_expert_data(ant_csvs: list[Path]) -> list[dict]:
    dfs = [pd.read_csv(a) for a in ant_csvs]
    df = pd.concat(dfs)

    del df["time"]

    records = df.to_dict(orient="records")

    msg = f"All records {len(records)}"
    logging.info(msg)

    return records


def split_data(
    records: list[dict],
    seed: int,
    train_split: float,
    val_split: float,
):
    random.seed(seed)

    random.shuffle(records)

    total = len(records)
    split1 = round(total * train_split)
    split2 = split1 + round(total * val_split)

    for i, rec in enumerate(records):
        if i < split1:
            rec["split"] = "train"
        elif i < split2:
            rec["split"] = "eval"
        else:
            rec["split"] = "test"

    msg = f"Split records {len(records)}"
    logging.info(msg)

    train = sum(1 for r in records if r["split"] == "train")
    eval_ = sum(1 for r in records if r["split"] == "eval")
    test = sum(1 for r in records if r["split"] == "test")

    msg = f"  Train {train}    Eval {eval_}    Test {test}"
    logging.info(msg)

    msg = "Per trait counts: pos / neg"
    logging.info(msg)

    for trait in util.TRAITS:
        pos_train = sum(1 for r in records if r["split"] == "train" and r[trait] == "1")
        neg_train = sum(1 for r in records if r["split"] == "train" and r[trait] == "0")
        pos_eval = sum(1 for r in records if r["split"] == "eval" and r[trait] == "1")
        neg_eval = sum(1 for r in records if r["split"] == "eval" and r[trait] == "0")
        pos_test = sum(1 for r in records if r["split"] == "test" and r[trait] == "1")
        neg_test = sum(1 for r in records if r["split"] == "test" and r[trait] == "0")
        total_pos = pos_train + pos_eval + pos_test
        total_neg = neg_train + neg_eval + neg_test
        total = total_pos + total_neg
        msg = (
            f"    {trait:<24} train {pos_train:4d} / {neg_train:4d}"
            f"    eval {pos_eval:4d} / {neg_eval:4d}"
            f"    test {pos_test:4d} / {neg_test:4d}"
            f"    total {total:4d}   {total_pos:4d} / {total_neg:4d}"
        )
        logging.info(msg)


def write_csv(split_csv: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(split_csv, index=False)


def validate_splits(args: argparse.Namespace) -> None:
    if sum((args.train_split, args.eval_split, args.test_split)) != 1.0:
        msg = "train, eval, and test splits must sum to 1.0"
        raise ValueError(msg)

    if any(
        s < 0.0 or s > 1.0 for s in (args.train_split, args.eval_split, args.test_split)
    ):
        msg = "All splits must be in the interval [0.0, 1.0]"
        raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Split the data into training, testing, & evaluation datasets."""
        ),
    )

    arg_parser.add_argument(
        "--ant-csv",
        metavar="PATH",
        type=Path,
        required=True,
        action="append",
        help="""These input CSV files hold expert classifications of the traits.
            Use this argument once for every ant CSV file.""",
    )

    arg_parser.add_argument(
        "--split-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the splits to this CSV file.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        action="append",
        required=True,
        help="""Train to classify this trait. Repeat this argument to train
            multiple trait labels. These are the headers in the --ant-csv file
            you want to extract.""",
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
        "--eval-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for evaluating the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for testing the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=2114987,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    validate_splits(args)

    return args


if __name__ == "__main__":
    main()
