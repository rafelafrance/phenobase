#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
from pathlib import Path

import pandas as pd

from phenobase.pylib import log


def main(args):
    log.started(args=args)

    if args.redbud:
        redbud(args)

    log.finished()


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

    # rows = [
    #     r for r in rows
    #     if not (r["state"].endswith("error") or r["state"].endswith("small"))
    # ]

    # msg = f"Filtered records = {len(rows)}"
    # logging.info(msg)

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
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="""Output the results to this CSV file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
