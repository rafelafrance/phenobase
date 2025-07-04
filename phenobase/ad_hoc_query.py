#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
from pathlib import Path

import pandas as pd

from phenobase.pylib import gbif, log


def main(args):
    log.started(args=args)

    if args.redbud:
        redbud(args)
    elif args.counts:
        counts(args)

    log.finished()


def counts(args):
    logging.info("Started counts")
    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        sql = """
            select gbifID, tiebreaker, state, family, genus, scientificName
            from multimedia join occurrence using (gbifID)
            """
        rows = [gbif.GbifRec(r) for r in cxn.execute(sql)]
        logging.info(f"Total records = {len(rows)}")

        rows = [r for r in rows if not r.no_url]
        logging.info(f"With URL = {len(rows)}")

        rows = [r for r in rows if not r.bad_image]
        logging.info(f"Good images = {len(rows)}")

        rows = [r for r in rows if not r.too_small]
        logging.info(f"Large images = {len(rows)}")

        rows = gbif.filter_bad_taxa(rows, args.bad_taxa)
        logging.info(f"Good taxa = {len(rows)}")


def redbud(args):
    logging.info("Started redbud")
    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        sql = """
            select * from multimedia join occurrence using (gbifid)
            where scientificname like 'Cercis canadensis%'
            or scientificname like 'Cercis occidentalis%';
            """
        rows = [dict(r) for r in cxn.execute(sql)]

        msg = f"Raw records = {len(rows)}"
        logging.info(msg)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Ad hoc queries on the GBIF database.
            """
        ),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Read GBIF data from this SQLite DB.""",
    )

    arg_parser.add_argument(
        "--redbud",
        action="store_true",
        help="""Get all Redbud data.""",
    )

    arg_parser.add_argument(
        "--counts",
        action="store_true",
        help="""Count valid records in the database.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="""Output the results to this CSV file.""",
    )

    arg_parser.add_argument(
        "--bad-taxa",
        type=Path,
        metavar="PATH",
        help="""A path to the CSV file with the list of bad families and genera.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
