#!/usr/bin/env python3

import argparse
import sqlite3
import textwrap
from pathlib import Path

from pylib import log


def main():
    log.started()

    args = parse_args()

    params = []

    for path in args.image_dir.glob("cache_*/*.jpg"):
        dir_ = int(path.parent.stem.split("_")[-1])
        gbifid, tiebreaker = path.stem.split("_")
        params.append((dir_, gbifid, tiebreaker))

    update = """update multimedia set dir = ? where gbifid = ? and tiebreaker = ?"""
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
