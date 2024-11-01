#!/usr/bin/env python3

import argparse
import logging
import textwrap
from collections import defaultdict
from pathlib import Path

from pylib import log


def main():
    log.started()

    args = parse_args()

    dupes: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for dir_ in args.image_dir.glob(args.subdir_glob):
        for path in dir_.glob("*.jpg"):
            gbifid, tiebreaker, *_ = path.stem.split("_")
            dupes[(gbifid, tiebreaker)].append(path)

    total = sum(c for v in dupes.values() if (c := len(v)) > 1)
    msg = f"Total duplicates {total}"
    logging.info(msg)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Update the directory field in the multimedia table."""
        ),
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

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
