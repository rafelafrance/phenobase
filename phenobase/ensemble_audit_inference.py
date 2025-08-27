#!/usr/bin/env python3

import argparse
import csv
import glob
import logging
import sqlite3
import textwrap
from pathlib import Path

from phenobase.pylib import gbif, log


def main(args) -> None:
    log.started(args=args)

    bad_taxa = gbif.get_bad_taxa(args.bad_taxa)

    # db_good, db_total = count_db(args.gbif_db, bad_taxa)
    # logging.info(f"Database records {db_total:,d}")
    # logging.info(f"Database good    {db_good:,d}")

    g1_good = count_infer(args.glob_1, bad_taxa)
    logging.info(f"Inference good one   {g1_good:,d}")

    log.finished()


def count_infer(glob_: str, bad_taxa: list[tuple]) -> int:
    good = set()

    for path in sorted(glob.glob(glob_)):  # noqa: PTH207
        path = Path(path)
        with path.open() as f:
            reader = csv.DictReader(f)

            rows = [gbif.GbifRec(r) for r in reader]
            logging.info(f"starting count     {len(rows):,d}")
            rows = gbif.filter_bad_images(rows)
            logging.info(f"bad images removed {len(rows):,d}")
            rows = gbif.filter_bad_taxa(rows, bad_taxa)
            logging.info(f"bad taxa removed   {len(rows):,d}")

            new = {r.stem for r in rows}
            good |= new

            logging.info(f"{path.stem}  new good {len(new):,d}")

    return len(good)


def count_db(gbif_db: Path, bad_taxa: list[tuple]) -> tuple[int, int]:
    total = 0
    good = set()

    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row

        cxn.execute("PRAGMA optimize")
        cxn.commit()

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
            total += len(rows)

            rows = [gbif.GbifRec(r) for r in rows]
            rows = gbif.filter_bad_images(rows)
            rows = gbif.filter_bad_taxa(rows, bad_taxa)

            good |= {r.stem for r in rows}

            logging.info(f"good {len(good):,d}   total {total:,d}")

    return len(good), total


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Calculate how many records did we expected vs how many we actually got.
            """
        ),
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
        "--glob-2",
        metavar="GLOB",
        help="""A glob for all the output CSVs for model 2.""",
    )

    arg_parser.add_argument(
        "--glob-3",
        metavar="GLOB",
        help="""A glob for all the output CSVs for model 3.""",
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
