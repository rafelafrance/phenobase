#!/usr/bin/env python3

import argparse
import logging
import textwrap
from pathlib import Path

from pylib import inference_util, log


def main(args):
    log.started(args=args)
    records = get_records(args.db, args.limit, args.offset, args.bad_taxa)

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


def get_records(db, limit, offset, bad_taxa):
    records = inference_util.get_inference_records(db, limit, offset)
    total = len(records)
    records = inference_util.filter_bad_images(records)
    good = len(records)
    records = inference_util.filter_bad_taxa(records, bad_taxa)
    families = len(records)
    logging.info(f"Total records {total}, good images {good}, good families {families}")
    return records


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Infer phenology traits."""),
    )

    arg_parser.add_argument(
        "--db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A database with information about herbarium sheets.""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=1_000_000,
        metavar="INT",
        help="""Limit to this many images. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="INT",
        help="""Read records after this offset. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--bad-taxa",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A path to the CSV file with the list of bad families and genera.""",
    )

    return inference_util.common_args(arg_parser)


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
