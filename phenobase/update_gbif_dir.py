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
        dir_ = int(path.parent.stem.split("_")[1])
        gbifid, tiebreaker = path.stem.split("_")
        params.append((dir_, gbifid, tiebreaker))

    update = """update multimedia set dir = ? where gbifid = ? and tiebreaker = ?"""
    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.executemany(update, params)

    log.finished()


def parse_args():
    description = """Download GBIF images."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output the merged GBIF data to this SQLite DB.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Place downloaded images into subdirectories of this directory.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
