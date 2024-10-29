#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

from pylib import log


def main():
    log.started()

    args = parse_args()

    params = []
    counts = defaultdict(int)

    for path in args.image_dir.glob("*.jpg"):
        gbifid, tiebreaker, *_ = path.stem.split("_")

        state = path.parent.stem
        if path.stem.endswith("_download_error"):
            state = "download error"
        elif path.stem.endswith("_image_error"):
            state = "image error"

        counts[state] += 1

        params.append((state, gbifid, tiebreaker))

    for state, count in counts.items():
        msg = f"State {state} count {count}"
        logging.info(msg)

    update = """update multimedia set state = ? where gbifid = ? and tiebreaker = ?;"""
    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.executemany(update, params)

    log.finished()


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
        required=True,
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

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
