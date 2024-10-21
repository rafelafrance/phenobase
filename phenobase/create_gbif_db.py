#!/usr/bin/env python3
import argparse
import logging
import sqlite3
import subprocess
import textwrap
from pathlib import Path

from phenobase.pylib import log


def main():
    log.started()

    args = parse_args()

    create_multimedia(args.gbif_db, args.multimedia_tsv)
    create_occurrence(args.gbif_db, args.occurrence_tsv)

    # The downloaded data only has plants so there is no need to filter on that
    # plants_only()

    log.finished()


def create_multimedia(gbif_db, multimedia_tsv):
    logging.info("Creating multimedia table")
    cmd = [
        "sqlite3",
        str(gbif_db),
        "-cmd",
        ".mode ascii",
        '.separator "\t" "\n"',
        f".import {multimedia_tsv} multimedia",
    ]
    subprocess.run(cmd, check=False)  # noqa: S603

    with sqlite3.connect(gbif_db) as cxn:
        logging.info("Deleting non-image multimedia records")
        cxn.execute("delete from multimedia where type != 'StillImage';")

        logging.info("Vacuuming")
        cxn.executescript("vacuum;")

        logging.info("Creating multimedia index")
        cxn.executescript("create index multimedia_gbifid on multimedia (gbifid);")


def create_occurrence(gbif_db, occurrence_tsv):
    logging.info("Creating occurrence table")
    cmd = [
        "sqlite3",
        str(gbif_db),
        "-cmd",
        ".mode ascii",
        '.separator "\t" "\n"',
        f".import {occurrence_tsv} occurrence",
    ]
    subprocess.run(cmd, check=False)  # noqa: S603

    with sqlite3.connect(gbif_db) as cxn:
        logging.info("Creating occurrence index")
        cxn.executescript("create index occurrence_gbifid on occurrence (gbifid);")

        logging.info("Deleting occurrence records without multimedia")
        cxn.execute(
            """
            delete from occurrence where gbifid not in (select gbifid from multimedia);
        """
        )

        logging.info("Vacuuming")
        cxn.executescript("vacuum;")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Create a GBIF database from downloaded TSV files."""
        ),
    )

    arg_parser.add_argument(
        "--multimedia-tsv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read multimedia data from this TSV file.""",
    )

    arg_parser.add_argument(
        "--occurrence-tsv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read occurrence data from this TSV file.""",
    )

    arg_parser.add_argument(
        "--gbif-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the merged GBIF data to this SQLite DB.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
