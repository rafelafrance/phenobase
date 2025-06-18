#!/usr/bin/env python3

import argparse
import logging
import random
import re
import socket
import sqlite3
import textwrap
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from pylib import log, util

TIMEOUT = 10  # Seconds to wait for a server reply
DELAY = 5  # Seconds to delay between attempts to download an image

IMAGES_PER_DIR = 100_000

ERRORS = (
    AttributeError,
    BufferError,
    ConnectionError,
    EOFError,
    FileNotFoundError,
    IOError,
    Image.DecompressionBombError,
    Image.UnidentifiedImageError,
    IndexError,
    OSError,
    RuntimeError,
    SyntaxError,
    TimeoutError,
    TypeError,
    ValueError,
    requests.exceptions.ReadTimeout,
)

Image.MAX_IMAGE_PIXELS = 300_000_000

# Set a timeout for requests
socket.setdefaulttimeout(TIMEOUT)


def main(args):
    log.started(args=args)

    args.image_dir.mkdir(parents=True, exist_ok=True)

    row_chunks = get_multimedia_recs(args.gbif_db, args.limit, args.offset)

    for row_chunk in row_chunks:
        subdir = mk_subdir(args.image_dir, args.dir_suffix)

        multitasking_download(
            subdir, row_chunk, args.max_workers, args.attempts, args.max_width
        )

    log.finished()


def multitasking_download(subdir, rows, max_workers, attempts, max_width):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                download,
                row["gbifID"],
                row["tiebreaker"],
                row["identifier"],
                subdir,
                attempts,
                max_width,
            )
            for row in rows
        ]
        results = list(as_completed(futures))
        return results


def get_multimedia_recs(gbif_db, limit, offset):
    """Get records and put them into batches/chunks that will fit in a directory."""
    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        select = """
            select gbifid, tiebreaker, identifier
            from multimedia
            where state is null
            limit ? offset ?;
            """
        rows = [dict(r) for r in cxn.execute(select, (limit, offset))]
        random.shuffle(rows)  # A feeble attempt to not overload image servers;

        row_chunks = [
            rows[i : i + IMAGES_PER_DIR] for i in range(0, len(rows), IMAGES_PER_DIR)
        ]
        return row_chunks


def mk_subdir(image_dir, dir_suffix):
    # Names formatted like "images_0001a"
    dirs = sorted(image_dir.glob("images_*"))

    dir_next = int(re.sub(r"\D", "", dirs[-1].stem)) if dirs else 0
    dir_next += 1

    subdir = image_dir / f"images_{dir_next:04d}{dir_suffix}"
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def update_multimedia_states(subdir, gbif_db):
    params = []
    for path in subdir.glob("*.jpg"):
        gbifid, tiebreaker, *_ = path.stem.split("_")

        state = path.parent.stem
        if path.stem.endswith("_download_error"):
            state = "download error"
        elif path.stem.endswith("_image_error"):
            state = "image error"
        # Being a small image is not really an error, but should be noted and fixed
        elif path.stem.endswith("_small"):
            state = f"{state} small"

        params.append((state, gbifid, tiebreaker))

    update = """update multimedia set state = ? where gbifid = ? and tiebreaker = ?;"""
    with sqlite3.connect(gbif_db) as cxn:
        cxn.executemany(update, params)


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

                if width * height < util.TOO_DAMN_SMALL:
                    raise ValueError

                if max_width < width < height:
                    height = int(height * (max_width / width))
                    width = max_width

                elif max_width < height:
                    width = int(width * (max_width / height))
                    height = max_width

                if width < max_width and height < max_width:
                    path = path.with_stem(f"{path.stem}_small")
                else:
                    image = image.resize((width, height))

                image.save(path)

        except ERRORS:
            error = path.with_stem(f"{path.stem}_image_error")
            error.touch()
            return "image_error"

    return "download"


def log_counts(counts):
    for key, value in counts.items():
        msg = f"Count {key} = {value}"
        logging.info(msg)


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
        "--dir-suffix",
        default="",
        help="""Prevent directory names colliding with multiple jobs.""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=1_000_000,
        metavar="INT",
        help="""Limit to this many completed downloads per batch.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="INT",
        help="""Read records after this offset. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-width",
        type=int,
        default=1024,
        metavar="PIXELS",
        help="""Resize the image to this width, short edge. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-workers",
        metavar="INT",
        type=int,
        default=1,
        help="""How many workers to spawn. (default: %(default)s)""",
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
    ARGS = parse_args()
    main(ARGS)
