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

    with sqlite3.connect(args.gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        select = """
            select gbifid, tiebreaker, identifier
            from multimedia
            where state is null
            order by random()
            limit ?;
            """
        rows = [dict(r) for r in cxn.execute(select, (args.limit,))]

    counts = defaultdict(int)

    next_dir = get_last_dir(args.image_dir) + 1
    subdir = args.image_dir / f"images_{next_dir:04d}"
    subdir.mkdir(parents=True, exist_ok=True)

    with Pool(processes=args.processes) as pool:
        results = [
            pool.apply_async(
                download,
                (
                    row["gbifID"],
                    row["tiebreaker"],
                    row["identifier"],
                    subdir,
                    args.attempts,
                    args.max_width,
                ),
            )
            for row in rows
        ]
        for result in results:
            counts[result.get()] += 1

    # results = [
    #     download(
    #         row["gbifID"],
    #         row["tiebreaker"],
    #         row["identifier"],
    #         subdir,
    #         args.attempts,
    #         args.max_width,
    #     )
    #     for row in rows
    # ]
    # for result in results:
    #     counts[result] += 1

    for key, value in counts.items():
        msg = f"Count {key} = {value}"
        logging.info(msg)

    log.finished()


def get_last_dir(image_dir) -> int:
    # Names formatted like "images_0001"
    dirs = sorted(image_dir.glob("images_*"))
    return int(dirs[-1].stem.split("_")[-1]) if dirs else 0


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
