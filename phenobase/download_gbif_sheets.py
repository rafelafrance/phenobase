#!/usr/bin/env python3

import argparse
import logging
import sqlite3
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pylib import log

DELAY = 5  # Seconds to delay between attempts to download an image

ERRORS = (TimeoutError, ConnectionError, requests.exceptions.ReadTimeout)


def main():
    log.started()
    args = parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        select = """
            select gbifid, tiebreaker, cache
            from multimedia
            where dir = 0
            limit ? offset ?;
            """
        rows = [dict(r) for r in cxn.execute(select, (args.limit, args.offset))]

    results = {"exists": 0, "download": 0, "error": 0}

    next_dir = get_last_dir(args.image_dir) + 1
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i, row in enumerate(rows):
            idx = int(i / args.per_dir) + next_dir
            subdir = f"cache_{idx:04d}"
            futures.append(
                executor.submit(
                    download,
                    row["gbifID"],
                    row["tiebreaker"],
                    row["cache"],
                    args.image_dir / subdir,
                    args.attempts,
                )
            )
        for future in as_completed(futures):
            results[future.result()] += 1

    for key, value in results.items():
        msg = f"Count {key} = {value}"
        logging.info(msg)

    log.finished()


def get_last_dir(image_dir) -> int:
    # Names formatted like "cache_0001"
    dirs = sorted(image_dir.glob("cache_*"))
    return int(dirs[-1].stem.split("_")[-1])


def download(gbifid, tiebreaker, url, image_dir, attempts):
    path = image_dir / f"{gbifid}_{tiebreaker}.jpg"

    if path.exists():
        return "exists"

    for _attempt in range(attempts):
        try:
            image = requests.get(url, timeout=DELAY).content
            with path.open("wb") as out_file:
                out_file.write(image)
        except ERRORS:  # noqa: PERF203
            time.sleep(DELAY)
        else:
            return "download"
    return "error"


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Download GBIF images."""),
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""This SQLite DB data contains the cached image links.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Place downloaded images into subdirectories of this directory.""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        metavar="INT",
        help="""Limit to this many completed downloads. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="INT",
        help="""Start counting records in the --limit from this offset.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--per-dir",
        type=int,
        default=10_000,
        metavar="INT",
        help="""How many images to put into each subdirectory.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-workers",
        metavar="INT",
        type=int,
        default=1,
        help="""How many worker threads to spawn. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        metavar="INT",
        help="""How many times to try downloading the image.
          (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
