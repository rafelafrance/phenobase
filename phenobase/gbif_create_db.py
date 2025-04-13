#!/usr/bin/env python3
import argparse
import hashlib
import logging
import sqlite3
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path

from phenobase.pylib import log

CACHE = "https://api.gbif.org/v1/image/cache/occurrence/{}/media/{}"


def main(args):
    log.started(args=args)

    create_multimedia(args.gbif_db, args.multimedia_tsv)
    create_occurrence(args.gbif_db, args.occurrence_tsv)

    add_multimedia_tiebreaker(args.gbif_db)
    add_multimedia_state(args.gbif_db)

    vacuum(args.gbif_db)

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
    subprocess.run(cmd, check=False)

    with sqlite3.connect(gbif_db) as cxn:
        logging.info("Deleting non-image multimedia records")
        cxn.execute("delete from multimedia where type != 'StillImage';")

        logging.info("Creating multimedia index")
        cxn.executescript("create index multimedia_gbifid on multimedia (gbifid);")


def create_occurrence(gbif_db: Path, occurrence_tsv: Path):
    logging.info("Creating occurrence table")
    if gbif_db.exists() and occurrence_tsv.exists():
        cmd = [
            "sqlite3",
            str(gbif_db),
            "-cmd",
            ".mode ascii",
            '.separator "\t" "\n"',
            f".import {occurrence_tsv} occurrence",
        ]
        subprocess.run(cmd, check=False)

    with sqlite3.connect(gbif_db) as cxn:
        logging.info("Creating occurrence index")
        cxn.executescript("create index occurrence_gbifid on occurrence (gbifid);")

        logging.info("Deleting occurrence records without multimedia")
        cxn.execute(
            """
            delete from occurrence where gbifid not in (select gbifid from multimedia);
            """
        )


def add_multimedia_tiebreaker(gbif_db: Path):
    """Add column as a tiebreaker for multiple multimedia records per gbifID."""
    logging.info("Add multimedia tiebreakers")

    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        cxn.execute("alter table multimedia add column 'tiebreaker' 'int';")

        tiebreakers = defaultdict(int)
        select = "select rowid, gbifid from multimedia;"

        params = []
        for row in cxn.execute(select):
            tiebreakers[row["gbifid"]] += 1
            params.append((tiebreakers[row["gbifid"]], row["rowid"]))

        update = "update multimedia set tiebreaker = ? where rowid = ?;"
        cxn.executemany(update, params)


def add_multimedia_state(gbif_db: Path):
    """Add column to hold the image download state."""
    logging.info("Add multimedia state")
    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row

        cxn.execute("alter table multimedia add column 'state' 'text';")

        select = """select rowid, identifier from multimedia;"""

        params = [
            ("no_url", r["rowid"])
            for r in cxn.execute(select)
            if r["identifier"] == "" or r["identifier"].endswith("/None")
        ]

        update = "update multimedia set state = ? where rowid = ?;"
        cxn.executemany(update, params)


def add_multimedia_cache_link(gbif_db: Path):
    logging.info("Add multimedia cache links")

    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        cxn.execute("alter table multimedia add column 'cache' 'text';")

        select = "select rowid, gbifid, identifier from multimedia;"

        params = []
        for row in cxn.execute(select):
            url = row["identifier"]

            cache = ""
            if url:
                md5 = hashlib.md5(url.encode("utf-8")).hexdigest()  # noqa: S324
                cache = CACHE.format(row["gbifid"], md5)

            params.append((cache, row["rowid"]))

        update = "update multimedia set cache = ? where rowid = ?;"
        cxn.executemany(update, params)


def vacuum(gbif_db):
    logging.info("Vacuuming")
    with sqlite3.connect(gbif_db) as cxn:
        cxn.executescript("vacuum;")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Create a GBIF database from downloaded TSV files."""
        ),
    )

    arg_parser.add_argument(
        "--gbif-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the merged GBIF data to this SQLite DB.""",
    )

    arg_parser.add_argument(
        "--multimedia-tsv",
        metavar="PATH",
        type=Path,
        help="""Read multimedia data from this TSV file.""",
    )

    arg_parser.add_argument(
        "--occurrence-tsv",
        metavar="PATH",
        type=Path,
        help="""Read occurrence data from this TSV file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
