#!/usr/bin/env python3

import argparse
import csv
import logging
import random
import sqlite3
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from pylib import log, util
from tqdm import tqdm

TOO_DAMN_SMALL = 10_000

ERRORS = (
    AttributeError,
    BufferError,
    ConnectionError,
    EOFError,
    FileNotFoundError,
    IOError,
    Image.DecompressionBombError,
    Image.UnidentifiedImageError,
    IndexError,
    OSError,
    RuntimeError,
    SyntaxError,
    TimeoutError,
    TypeError,
    ValueError,
)


def main(args):
    log.started(args.log_file, args=args)

    random.seed(args.seed)

    records = get_expert_data(args.ant_csv)
    append_metadata(args.metadata_db, records)

    for file_name, row in records.items():
        row["name"] = file_name

    records = list(records.values())

    filter_trait("flowers", records, args.bad_flower_families)
    filter_trait("fruits", records, args.bad_fruit_families)

    split_data(records, args.seed, args.train_split, args.val_split)
    write_csv(args.split_dir / "all_traits.csv", records)
    process_images(records, args.image_dir, args.split_dir, args.max_width)

    log.finished()


def get_expert_data(ant_csvs: list[Path]) -> dict[str, dict]:
    records = defaultdict(dict)
    for csv_file in ant_csvs:
        with csv_file.open() as inf:
            reader = csv.DictReader(inf)
            for row in reader:
                for trait in util.TRAITS:
                    if label := row.get(trait):
                        records[row["file"]][trait] = label
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


def filter_trait(trait: str, records: list[dict], bad_families: Path) -> None:
    with bad_families.open() as f:
        bad = {row["family"].lower() for row in csv.DictReader(f)}

    for rec in records:
        label = rec.get(trait, "")

        rec[f"old_{trait}"] = label

        if not label or label not in "01":
            continue
        if rec["family"].lower() in bad:
            rec[trait] = "F"


def split_data(
    records: list[dict],
    seed: int,
    train_split: float,
    val_split: float,
) -> None:
    random.seed(seed)

    random.shuffle(records)

    total = len(records)
    split1 = round(total * train_split)
    split2 = split1 + round(total * val_split)

    for i, rec in enumerate(records):
        if i < split1:
            rec["split"] = "train"
        elif i < split2:
            rec["split"] = "val"
        else:
            rec["split"] = "test"


def write_csv(split_csv: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(split_csv, index=False)


def process_images(
    records: list[dict], image_dir: Path, split_dir: Path, max_width: int
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings

        try:
            for rec in tqdm(records):
                src = image_dir / rec["name"]
                dst = split_dir / "images" / rec["name"]

                if not dst.exists():
                    with Image.open(src) as image:
                        width, height = image.size

                        if width * height < TOO_DAMN_SMALL:
                            raise ValueError

                        if max_width < width < height:
                            height = int(height * (max_width / width))
                            width = max_width

                        elif max_width < height:
                            width = int(width * (max_width / height))
                            height = max_width

                        image = image.resize((width, height))

                        image.save(dst)

        except ERRORS:
            logging.exception("Could not process image")


def validate_splits(args: argparse.Namespace) -> None:
    splits = (args.train_split, args.val_split, args.test_split)

    if sum(splits) != 1.0:
        msg = "train, val, and test splits must sum to 1.0"
        raise ValueError(msg)

    if any(s < 0.0 or s > 1.0 for s in splits):
        msg = "All splits must be in the interval [0.0, 1.0]"
        raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Split the data into training, testing, & validation datasets."""
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
        "--image-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get herbarium sheet images from this directory.""",
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
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for validating the model between epochs.
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

    arg_parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        metavar="PIXELS",
        help="""Resize the image to this width, short edge. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--log-file",
        metavar="PATH",
        type=Path,
        help="""Log messages to this file.""",
    )

    args = arg_parser.parse_args()

    validate_splits(args)

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
