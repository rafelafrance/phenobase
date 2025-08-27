#!/usr/bin/env python3

import argparse
import csv
import glob
import logging
import sqlite3
import textwrap
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from phenobase.pylib import gbif, log


def main(args) -> None:
    log.started(args=args)

    bad_taxa = gbif.get_bad_taxa(args.bad_taxa)

    inferred = get_inferred(args.glob_1, bad_taxa)

    not_inferred = get_not_inferred(args.gbif_db, bad_taxa, inferred)

    df = pd.DataFrame(not_inferred)
    df.to_csv(args.not_inferred, index=False)

    log.finished()


def get_inferred(glob_: str, bad_taxa: list[tuple]) -> set:
    inferred = set()

    for path in sorted(glob.glob(glob_)):  # noqa: PTH207
        path = Path(path)
        with path.open() as f:
            reader = csv.DictReader(f)

            rows = [gbif.GbifRec(r) for r in reader]
            rows = gbif.filter_bad_images(rows)
            rows = gbif.filter_bad_taxa(rows, bad_taxa)
            logging.info(f"    {path.stem} file count {len(rows):,d}")

            inferred |= {r.stem for r in rows}

    logging.info(f"Inferred count {len(inferred):,d}")

    return inferred


def get_not_inferred(gbif_db: Path, bad_taxa: list[tuple], inferred: set) -> list[dict]:
    total = 0
    not_inferred = {}

    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row

        cursor = cxn.cursor()
        cursor.arraysize = 100_000

        cursor.execute(
            """
            select gbifID, tiebreaker, state, family, genus, scientificName
              from multimedia
              join occurrence using (gbifID)
            """
        )

        while True:
            rows = cursor.fetchmany()
            if not rows:
                break

            if total % 2_000_000 == 0:
                cxn.execute("PRAGMA optimize")
                cxn.commit()

            total += len(rows)

            rows = [gbif.GbifRec(r) for r in rows]
            rows = gbif.filter_bad_images(rows)
            rows = gbif.filter_bad_taxa(rows, bad_taxa)
            rows = {r.stem: asdict(r) for r in rows if r.stem not in inferred}
            logging.info(f"    batch {total:,d} not inferred {len(rows):,d}")

            not_inferred |= rows

    logging.info(f"Not inferred count {len(not_inferred):,d}")

    return list(not_inferred.values())


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Get all of the not inferred records."""),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""This SQLite DB data contains image links.""",
    )

    arg_parser.add_argument(
        "--glob-1",
        required=True,
        metavar="GLOB",
        help="""A glob for all the output CSVs for model 1.""",
    )

    arg_parser.add_argument(
        "--not-inferred",
        required=True,
        metavar="PATH",
        help="""Write not inferred IDs to this CSV file.""",
    )

    arg_parser.add_argument(
        "--bad-taxa",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A path to the CSV file with the list of bad families and genera.""",
    )
    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
