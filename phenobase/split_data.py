#!/usr/bin/env python3

import argparse
import csv
import logging
import random
import shutil
import sqlite3
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from phenobase.pylib import const, log

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


def main():
    args = parse_args()
    log.started(args.log_file)

    random.seed(args.seed)

    records = get_expert_data(args.ant_csv)
    append_metadata(args.metadata_db, records)

    flowers = filter_trait("flowers", records, args.bad_flower_families)
    fruits = filter_trait("fruits", records, args.bad_fruit_families)
    leaves = filter_trait("leaves", records)
    buds = filter_trait("buds", records)

    split_data("flowers", flowers, args.train_split, args.valid_split)
    split_data("fruits", fruits, args.train_split, args.valid_split)
    split_data("leaves", leaves, args.train_split, args.valid_split)
    split_data("buds", buds, args.train_split, args.valid_split)

    write_csv(args.split_dir / "all_traits.csv", list(records.values()))
    write_csv(args.split_dir / "flowers.csv", flowers)
    write_csv(args.split_dir / "fruits.csv", fruits)
    write_csv(args.split_dir / "leaves.csv", leaves)
    write_csv(args.split_dir / "buds.csv", buds)

    process_images(flowers, "flowers", args.image_dir, args.split_dir, args.max_width)
    process_images(fruits, "fruits", args.image_dir, args.split_dir, args.max_width)
    process_images(leaves, "leaves", args.image_dir, args.split_dir, args.max_width)
    process_images(buds, "buds", args.image_dir, args.split_dir, args.max_width)

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
                bad.add(row["family"].lower())

    bad_value = 0
    bad_family = 0
    skipped = 0
    for file_name, rec in records.items():
        value = rec.get(trait)
        if not value:
            skipped += 1
        elif value not in "01":
            bad_value += 1
        elif rec["family"].lower() in bad:
            bad_family += 1
        else:
            filtered.append({"name": file_name, "value": value, "split": ""})

    pos = sum(1 for v in filtered if v["value"] == "1")
    neg = sum(1 for v in filtered if v["value"] == "0")

    logging.info(f"Filtered {trait}")
    logging.info(f"  Before filter    = {(len(records) - skipped):5d}")
    logging.info(f"  Kept positive    = {pos:5d}")
    logging.info(f"  Kept negative    = {neg:5d}")
    logging.info(f"  Kept total       = {(pos + neg):5d}")
    logging.info(f"  Removed values   = {bad_value:5d}")
    logging.info(f"  Removed families = {bad_family:5d}")
    logging.info(f"  Removed total    = {(bad_value + bad_family):5d}")

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
            rec["split"] = "valid"
        else:
            rec["split"] = "test"

    logging.info(f"Split {trait:<7}      = {total:5d}")

    for split in ("train", "valid", "test"):
        pos = sum(1 for r in records if r["split"] == split and r["value"] == "1")
        neg = sum(1 for r in records if r["split"] == split and r["value"] == "0")
        logging.info(f"  {split:<5} positive   = {pos:5d}")
        logging.info(f"  {split:<5} negitive   = {neg:5d}")
        logging.info(f"  {split:<5} total      = {(pos + neg):5d}")


def write_csv(split_csv: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(split_csv, index=False)


def process_images(
    records: list[dict], trait, image_dir: Path, split_dir: Path, max_width: int
) -> None:
    dirs = {}
    for split in const.SPLITS:
        dirs[(split, "0")] = split_dir / "data" / trait / split / "0"
        dirs[(split, "1")] = split_dir / "data" / trait / split / "1"

        if dirs[(split, "0")].exists():
            shutil.rmtree(dirs[(split, "0")])

        if dirs[(split, "1")].exists():
            shutil.rmtree(dirs[(split, "1")])

        dirs[(split, "0")].mkdir(parents=True, exist_ok=True)
        dirs[(split, "1")].mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings

        try:
            for rec in tqdm(records, desc=f"{trait:<8} images"):
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

                link = dirs[(rec["split"], rec["value"])] / rec["name"]
                link.absolute().symlink_to(dst.absolute())

        except ERRORS:
            logging.exception("Could not process image")


def validate_splits(args: argparse.Namespace) -> None:
    splits = (args.train_split, args.valid_split, args.test_split)

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
        "--valid-split",
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
    main()
