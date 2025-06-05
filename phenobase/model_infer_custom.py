#!/usr/bin/env python3

import argparse
import csv
import logging
import textwrap
from pathlib import Path

from pylib import inference_util, log


def main(args):
    log.started(args=args)
    records = get_records(args.custom_dataset)

    inference_util.infer_records(
        records,
        args.checkpoint,
        args.problem_type,
        args.image_dir,
        args.image_size,
        args.trait,
        args.threshold_low,
        args.threshold_high,
        args.output_csv,
    )

    log.finished()


def get_records(custom_dataset):
    with custom_dataset.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    total = len(records)
    records = inference_util.filter_bad_images(records)
    good = len(records)
    logging.info(f"Total records {total}, good images {good}")
    return records


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Infer phenology traits from a custom dataset."""
        ),
    )

    arg_parser.add_argument(
        "--custom-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A CSV file with GBIF records about the custom herbarium sheets.""",
    )

    return inference_util.common_args(arg_parser)


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
