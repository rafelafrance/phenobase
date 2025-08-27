#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
from pathlib import Path

from pylib import gbif, inference, log


def main(args):
    log.started(args=args)
    bad_taxa = gbif.get_bad_taxa(args.bad_taxa)

    total, processed = 0, 0

    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row

        cursor = cxn.cursor()
        cursor.arraysize = 100_000

        select = """
            select gbifID, tiebreaker, state, family, genus, scientificName
            from multimedia join occurrence using (gbifID)
            """

        select_args = None
        if args.limit or args.offset:
            select += " limit ? offset ?"
            select_args = (args.limit, args.offset)

        cursor.execute(select, select_args)

        while True:
            rows = cursor.fetchmany()
            if not rows:
                break

            total += len(rows)

            rows = [gbif.GbifRec(r) for r in rows]
            rows = gbif.filter_bad_images(rows)
            rows = gbif.filter_bad_taxa(rows, bad_taxa)

            processed += len(rows)

            inference.infer_records(
                rows,
                args.checkpoint,
                args.problem_type,
                args.image_dir,
                args.image_size,
                args.trait,
                args.output_csv,
            )

    logging.info(f"Total records     {total}")
    logging.info(f"Records processed {processed}")

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Infer phenology traits."""),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""This SQLite DB with information about herbarium sheets.""",
    )

    arg_parser.add_argument(
        "--bad-taxa",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A path to the CSV file with the list of bad families and genera.""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=0,
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

    return inference.common_args(arg_parser)


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
