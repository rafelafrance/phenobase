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

            dupes[(gbifid, tiebreaker)].append(path.name)

    total = len(dupes)
    msg = f"Total           {total:8,d}"
    logging.info(msg)

    dupe_count = sum(1 for v in dupes.values() if len(v) > 1)
    msg = f"Duplicates      {dupe_count:8,d}"
    logging.info(msg)

    differ = sum(1 for v in dupes.values() if len(v) > 1 and any(x != v[0] for x in v))
    msg = f"Different       {differ:8,d}"
    logging.info(msg)

    small = sum(
        1 for d in dupes.values() for p in d if any(x.find("small") > -1 for x in d)
    )
    msg = f"Small images    {small:8,d}"
    logging.info(msg)

    down = sum(
        1
        for d in dupes.values()
        for p in d
        if any(x.find("download_error") > -1 for x in d)
    )
    msg = f"Download errors {down:8,d}"
    logging.info(msg)

    image = sum(
        1
        for d in dupes.values()
        for p in d
        if any(x.find("image_error") > -1 for x in d)
    )
    msg = f"Image errors    {image:8,d}"
    logging.info(msg)

    for k, v in dupes.items():
        if len(v) > 1 and any(x != v[0] for x in v):
            print(k, v)
            break

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
