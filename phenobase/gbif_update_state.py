#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

from pylib import log

UPDATE = """update multimedia set state = ? where gbifid = ? and tiebreaker = ?;"""


def main(args):
    log.started(args=args)

    counts: dict[str, int] = defaultdict(int)

    sub_dirs = sorted(args.image_dir.glob(args.subdir_glob))

    for dir_ in sub_dirs:
        msg = f"Updating {dir_.stem}"
        logging.info(msg)
        params = []
        for path in dir_.glob("*.jpg"):
            gbifid, tiebreaker, *_ = path.stem.split("_")

            state = get_state(path)

            counts[state] += 1
            params.append((state, gbifid, tiebreaker))

        if not args.count_only:
            with sqlite3.connect(args.gbif_db) as cxn:
                cxn.executemany(UPDATE, params)

    log_counts(counts)

    log.finished()


def log_counts(counts: dict[str, int]) -> None:
    success, errors = 0, 0
    for state, count in counts.items():
        if "error" in state:
            errors += count
        else:
            success += count
        msg = f"State '{state}' count {count}"
        logging.info(msg)

    msg = f"Successful downloads {success:7,d}"
    logging.info(msg)

    msg = f"Download errors      {errors:7,d}"
    logging.info(msg)


def get_state(path: Path) -> str:
    state = path.parent.stem
    if path.stem.endswith("_download_error"):
        state = f"{state} download error"
    elif path.stem.endswith("_image_error"):
        state = f"{state} image error"
    elif path.stem.endswith("_small"):
        state = f"{state} small"
    return state


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Update the directory field in the multimedia table."""
        ),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        metavar="PATH",
        help="""Update the multimedia table in this SQLite DB.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Search subdirectories of this directory for image files.""",
    )

    arg_parser.add_argument(
        "--subdir-glob",
        default="images_*",
        metavar="GLOB",
        help="""Use this to find subdirectories within the --image-dir.""",
    )

    arg_parser.add_argument(
        "--count-only",
        action="store_true",
        help="Do not update the database, only count the results.",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
