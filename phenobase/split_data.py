#!/usr/bin/env python3

import argparse
import csv
import logging
import random
import sqlite3
import sys
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

OUT_IMAGE_DIR = Path("datasets") / "images"


def main(args):
    log.started(args.log_file)

    random.seed(args.seed)

    records = get_expert_data(args.ant_csv)
    append_metadata(args.metadata_db, records, gbif_db=args.gbif_db)

    for file_name, row in records.items():
        row["name"] = file_name

    records = list(records.values())

    filter_trait("flowers", records, args.bad_flower_families)
    filter_trait("fruits", records, args.bad_fruit_families)

    split_data(records, args.split_fraction, args.split1, args.split2)
    write_csv(args.split_csv, records)
    # process_images(records, args.image_dir, args.max_width)

    log.finished()


def get_expert_data(ant_csvs: list[Path]) -> dict[str, dict]:
    records = defaultdict(dict)
    for csv_file in ant_csvs:
        with csv_file.open() as inf:
            reader = csv.DictReader(inf)
            for row in reader:
                for trait in const.TRAITS:
                    if label := row.get(trait):
                        records[row["file"]][trait] = label
    return records


def append_metadata(
    metadata_db: Path, records: dict[str, dict], *, gbif_db: bool = False
) -> None:
    if gbif_db:
        from_gbif_db(metadata_db, records)
        return
    from_angiosperm_db(metadata_db, records)


def from_gbif_db(metadata_db: Path, records: dict[str, dict]):
    sql = """select * from multimedia join occurrence using (gbifid)
              where gbifid = ? and tiebreaker = ?"""
    with sqlite3.connect(metadata_db) as cxn:
        cxn.row_factory = sqlite3.Row
        for file_name, rec in records.items():
            stem = file_name.split(".")[0]
            gbifid, tiebreaker, *_ = stem.split("_")
            data = cxn.execute(sql, (gbifid, tiebreaker)).fetchone()
            rec |= dict(data)


def from_angiosperm_db(metadata_db: Path, records: dict[str, dict]):
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

        if rec["family"].lower() in bad:
            rec[trait] = "F"


def split_data(
    records: list[dict], split_fraction: float, split1: const.SPLIT, split2: const.SPLIT
) -> None:
    random.shuffle(records)

    fract = round(len(records) * split_fraction)

    for i, rec in enumerate(records):
        rec["split"] = split1 if i < fract else split2


def write_csv(split_csv: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(split_csv, index=False)


def process_images(records: list[dict], in_image_dir: Path, max_width: int) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings

        try:
            for rec in tqdm(records):
                src = in_image_dir / rec["name"]
                dst = OUT_IMAGE_DIR / rec["name"]

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


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Split the data into training, testing, & validation datasets."""
        ),
    )

    arg_parser.add_argument(
        "--ant-csv",
        type=Path,
        required=True,
        action="append",
        metavar="PATH",
        help="""These input CSV files hold expert classifications of the traits.
            Use this argument once for every ant CSV file.""",
    )

    arg_parser.add_argument(
        "--metadata-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Get metadata for the classifications from this DB.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Get herbarium sheet images from this directory.""",
    )

    arg_parser.add_argument(
        "--split-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output the split CSV to this file.""",
    )

    arg_parser.add_argument(
        "--split1",
        required=True,
        choices=const.SPLITS,
        metavar="SPLIT",
        help="""Output the split CSV to this file.""",
    )

    arg_parser.add_argument(
        "--split2",
        choices=const.SPLITS,
        metavar="SPLIT",
        help="""Output the split CSV to this file.""",
    )

    arg_parser.add_argument(
        "--split-fraction",
        type=float,
        metavar="FRACTION",
        default=0.75,
        help="""What fraction of records to use for training the model [0.0 - 1.0].
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--bad-flower-families",
        type=Path,
        metavar="PATH",
        help="""Make sure flowers in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--bad-fruit-families",
        type=Path,
        metavar="PATH",
        help="""Make sure fruits in these families are skipped.""",
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

    arg_parser.add_argument(
        "--gbif-db",
        action="store_true",
        help="""Does the metadata come from a GBIB database?""",
    )

    args = arg_parser.parse_args()

    args.split_fraction = 1.0 if not args.split2 else args.split_fraction

    if args.split_fraction < 0.0 or args.split_fraction > 1.0:
        sys.exit(f"--split-fraction {args.split_fraction} must be between [0.0 - 1.0]")

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
