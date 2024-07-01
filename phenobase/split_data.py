#!/usr/bin/env python3

import argparse
import logging
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import pandas as pd

from phenobase.pylib import log, util


def main():
    log.started()
    args = parse_args()

    headers = args.trait

    records = get_expert_data(args.ant_csv)
    records = filter_data(records, headers)
    groups = group_data(records, headers)
    records = split_data(groups, args.seed, args.train_split, args.eval_split)
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


def filter_data(records: list[dict], headers: list[str]) -> list[dict]:
    records = [r for r in records if all(r[h] in "01" for h in headers)]

    msg = f"Filtered records {len(records)}"
    logging.info(msg)

    return records


def group_data(records: list[dict], headers: list[str]) -> dict[list[dict]]:
    groups = defaultdict(list)

    for rec in records:
        key = " ".join(rec[h] for h in headers)
        groups[key].append(rec)

    for key, recs in groups.items():
        msg = f"Group: {key} has {len(recs)} records"
        logging.info(msg)

    return groups


def split_data(
    groups: dict[list[dict]],
    seed: int,
    train_split: float,
    val_split: float,
) -> list[dict]:
    random.seed(seed)

    records = []

    for record_list in groups.values():
        random.shuffle(record_list)

        total = len(record_list)
        split1 = round(total * train_split)
        split2 = split1 + round(total * val_split)

        for i, rec in enumerate(record_list):
            if i < split1:
                rec["split"] = "train"
            elif i < split2:
                rec["split"] = "eval"
            else:
                rec["split"] = "test"
            records.append(rec)

    msg = f"Split   records {len(records)}"
    logging.info(msg)

    msg = f"  Train records {len([r for r in records if r['split'] == 'train'])}"
    logging.info(msg)

    msg = f"  Eval  records {len([r for r in records if r['split'] == 'eval'])}"
    logging.info(msg)

    msg = f"  Test  records {len([r for r in records if r['split'] == 'test'])}"
    logging.info(msg)

    return records


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
