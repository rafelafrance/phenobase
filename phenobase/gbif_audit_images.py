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
    logging.info(args)

    dupes: dict[tuple[str, str], list[str]] = defaultdict(list)

    for dir_ in args.image_dir.glob(args.subdir_glob):
        for path in dir_.glob("*.jpg"):
            gbifid, tiebreaker, *_ = path.stem.split("_", maxsplit=2)

            tail = f"{path.parent}/{path.name}"
            dupes[(gbifid, tiebreaker)].append(tail)

    total = sum(c for v in dupes.values() if (c := len(v)))
    msg = f"Total           {total:,8d}"
    logging.info(msg)

    dupe_count = sum(c for v in dupes.values() if (c := len(v)) > 1)
    msg = f"Duplicates      {dupe_count:,8d}"
    logging.info(msg)

    small = sum(c for d in dupes.values() for p in d if p.find("small") > -1)
    msg = f"Small images    {small:,8d}"
    logging.info(msg)

    down = sum(c for d in dupes.values() for p in d if p.find("download_error") > -1)
    msg = f"Download errors {down:,8d}"
    logging.info(msg)

    image = sum(c for d in dupes.values() for p in d if p.find("image_error") > -1)
    msg = f"Image errors    {image:,8d}"
    logging.info(msg)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Audit the downloaded images."""),
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
