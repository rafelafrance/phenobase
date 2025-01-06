#!/usr/bin/env python3

import argparse
import csv
import logging
import random
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

import pandas as pd

from phenobase.pylib import const, log


def main():
    log.started()
    args = parse_args()

    random.seed(args.seed)

    records = get_expert_data(args.ant_csv)
    append_metadata(args.metadata_db, records)

    flowers = filter_trait("flowers", records, args.bad_flower_families)
    fruits = filter_trait("fruits", records, args.bad_fruit_families)
    leaves = filter_trait("leaves", records)
    buds = filter_trait("buds", records)

    split_data("flowers", flowers, args.train_split, args.eval_split)
    split_data("fruits", fruits, args.train_split, args.eval_split)
    split_data("leaves", leaves, args.train_split, args.eval_split)
    split_data("buds", buds, args.train_split, args.eval_split)

    write_csv(args.split_dir / "all_traits.csv", list(records.values()))
    write_csv(args.split_dir / "flowers.csv", flowers)
    write_csv(args.split_dir / "fruits.csv", fruits)
    write_csv(args.split_dir / "leaves.csv", leaves)
    write_csv(args.split_dir / "buds.csv", buds)

    log.finished()


def get_expert_data(ant_csvs: list[Path]) -> dict[str, dict]:
    records = defaultdict(dict)
    for csv_file in ant_csvs:
        with csv_file.open() as inf:
            reader = csv.DictReader(inf)
            for row in reader:
                for trait in const.TRAITS:
                    if value := row.get(trait):
                        records[row["file"]][trait] = value

    logging.info(f"Expert record count {len(records)}")
    return records


def append_metadata(metadata_db: Path, records: dict[str, dict]) -> None:
    sql = """select * from angiosperms where coreid = ?"""
    with sqlite3.connect(metadata_db) as cxn:
        cxn.row_factory = sqlite3.Row
        for file_name, rec in records.items():
            coreid = file_name.split(".")[0]
            data = cxn.execute(sql, (coreid,)).fetchone()
            rec |= dict(data)
            del rec["filter_set"]


def filter_trait(trait: str, records: dict[str, dict], bad_families=None) -> list[dict]:
    filtered = []

    bad = set()
    if bad_families:
        with bad_families.open() as inf:
            reader = csv.DictReader(inf)
            for row in reader:
                bad.add(row["family"])

    for rec in records.values():
        value = rec.get(trait)
        if value and value in "01" and rec["family"] not in bad:
            filtered.append({"value": value, "split": ""})

    logging.info(f"Positive {trait} = {sum(1 for v in filtered if v['value'] == '1')}")
    logging.info(f"Negative {trait} = {sum(1 for v in filtered if v['value'] == '0')}")

    return filtered


def split_data(trait: str, records: list[dict], train_split: float, val_split: float):
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

    logging.info(f"Split {trait} records {len(records)}")

    train = sum(1 for r in records if r["split"] == "train")
    eval_ = sum(1 for r in records if r["split"] == "eval")
    test = sum(1 for r in records if r["split"] == "test")

    logging.info(f"  Train {train:<6}  Eval {eval_}  Test {test}")

    pos_train = sum(1 for r in records if r["split"] == "train" and r["value"] == "1")
    neg_train = sum(1 for r in records if r["split"] == "train" and r["value"] == "0")
    pos_eval = sum(1 for r in records if r["split"] == "eval" and r["value"] == "1")
    neg_eval = sum(1 for r in records if r["split"] == "eval" and r["value"] == "0")
    pos_test = sum(1 for r in records if r["split"] == "test" and r["value"] == "1")
    neg_test = sum(1 for r in records if r["split"] == "test" and r["value"] == "0")
    total_pos = pos_train + pos_eval + pos_test
    total_neg = neg_train + neg_eval + neg_test
    total = total_pos + total_neg
    msg = (
        f"  train {pos_train:4d} / {neg_train:4d}"
        f"    eval {pos_eval:4d} / {neg_eval:4d}"
        f"    test {pos_test:4d} / {neg_test:4d}"
        f"    total {total:4d} {total_pos:4d} / {total_neg:4d}"
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
        "--bad-flower-families",
        metavar="PATH",
        type=Path,
        help="""Make sure flowers in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--bad-fruit-families",
        metavar="PATH",
        type=Path,
        help="""Make sure fruits in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--metadata-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get metadata for the classifications from this DB.""",
    )

    arg_parser.add_argument(
        "--split-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the split CSV files into this directory.""",
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
