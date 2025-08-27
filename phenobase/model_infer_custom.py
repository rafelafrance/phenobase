#!/usr/bin/env python3

import argparse
import csv
import logging
import textwrap
from pathlib import Path

from pylib import gbif, inference, log


def infer_from_custom_data(args):
    log.started(args=args)
    records = get_records(args.custom_data)

    inference.infer_records(
        records,
        args.checkpoint,
        args.problem_type,
        args.image_dir,
        args.image_size,
        args.trait,
        args.output_csv,
    )

    log.finished()


def get_records(custom_data):
    with custom_data.open() as f:
        reader = csv.DictReader(f)
        records = [gbif.GbifRec(r) for r in reader]
    good = gbif.filter_bad_images(records)
    logging.info(f"Total records {len(records)}, good images {len(good)}")
    return good


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Infer phenology traits from a custom dataset."""
        ),
    )

    arg_parser.add_argument(
        "--custom-data",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A CSV file with GBIF records about the custom herbarium sheets.""",
    )

    return inference.common_args(arg_parser)


if __name__ == "__main__":
    ARGS = parse_args()
    infer_from_custom_data(ARGS)
