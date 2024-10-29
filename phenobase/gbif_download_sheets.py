#!/usr/bin/env python3

import argparse
import logging
import socket
import sqlite3
import textwrap
import time
import warnings
from collections import defaultdict
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import requests
from PIL import Image
from pylib import log

Image.MAX_IMAGE_PIXELS = 300_000_000

# Set a timeout for requests
TIMEOUT = 10
socket.setdefaulttimeout(TIMEOUT)

DELAY = 5  # Seconds to delay between attempts to download an image

ERRORS = (
    AttributeError,
    BufferError,
    ConnectionError,
    Image.DecompressionBombError,
    EOFError,
    FileNotFoundError,
    IOError,
    IndexError,
    OSError,
    RuntimeError,
    SyntaxError,
    TimeoutError,
    TypeError,
    Image.UnidentifiedImageError,
    ValueError,
    requests.exceptions.ReadTimeout,
)


def main():
    log.started()
    args = parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)

    for _i in args.batches:
        rows = get_multimedia_recs(args.gbif_db, args.limit)
        subdir = mk_subdir(args.image_dir)

        results = parallel_download(
            subdir, rows, args.processes, args.attempts, args.max_width
        )
        for result in results:
            counts[result.get()] += 1

        update_multimedia_states(subdir, args.gbif_db)

    for key, value in counts.items():
        msg = f"Count {key} = {value}"
        logging.info(msg)

    log.finished()


def update_multimedia_states(subdir, gbif_db):
    params = []
    for path in subdir.glob("*.jpg"):
        gbifid, tiebreaker, *_ = path.stem.split("_")

        state = path.parent.stem
        if path.stem.endswith("_download_error"):
            state = "download error"
        elif path.stem.endswith("_image_error"):
            state = "image error"

        params.append((state, gbifid, tiebreaker))

    update = """update multimedia set state = ? where gbifid = ? and tiebreaker = ?;"""
    with sqlite3.connect(gbif_db) as cxn:
        cxn.executemany(update, params)


def parallel_download(subdir, rows, processes, attempts, max_width):
    with Pool(processes=processes) as pool:
        results = [
            pool.apply_async(
                download,
                (
                    row["gbifID"],
                    row["tiebreaker"],
                    row["identifier"],
                    subdir,
                    attempts,
                    max_width,
                ),
            )
            for row in rows
        ]
        return results


def get_multimedia_recs(gbif_db, limit):
    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        select = """
            select gbifid, tiebreaker, identifier
            from multimedia
            where state is null
            order by random()
            limit ?;
            """
        return [dict(r) for r in cxn.execute(select, (limit,))]


def mk_subdir(image_dir):
    # Names formatted like "images_0001"
    dirs = sorted(image_dir.glob("images_*"))

    next_dir = int(dirs[-1].stem.split("_")[-1]) if dirs else 0
    next_dir += 1

    subdir = image_dir / f"images_{next_dir:04d}"
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def download(gbifid, tiebreaker, url, subdir, attempts, max_width):
    path: Path = subdir / f"{gbifid}_{tiebreaker}.jpg"

    if path.exists():
        return "exists"

    for _attempt in range(attempts):
        try:
            req = requests.get(url, timeout=DELAY)
            break
        except ERRORS:
            time.sleep(DELAY)
    else:
        error = path.with_stem(f"{path.stem}_download_error")
        error.touch()
        return "download_error"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings

        try:
            with Image.open(BytesIO(req.content)) as image:
                width, height = image.size

                if width * height < 10_000:
                    raise ValueError

                if max_width < width < height:
                    height = int(height * (max_width / width))
                    width = max_width

                elif max_width < height:
                    width = int(width * (max_width / height))
                    height = max_width

                image = image.resize((width, height))
                image.save(path)

        except ERRORS:
            error = path.with_stem(f"{path.stem}_image_error")
            error.touch()
            return "image_error"

    return "download"


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
        help="""This SQLite DB data contains the image image links.""",
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
        "--batches",
        type=int,
        default=10,
        metavar="INT",
        help="""How many batches of size --limit to run. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        metavar="PIXELS",
        help="""Resize the image to this width, short edge. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--processes",
        metavar="INT",
        type=int,
        default=1,
        help="""How many worker processes to spawn. (default: %(default)s)""",
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
