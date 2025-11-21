#!/usr/bin/env python3

import argparse
import csv
import textwrap
from pathlib import Path
from shutil import copyfile

from phenobase.pylib import log


def main(args: argparse.Namespace) -> None:
    log.started(args=args)

    if args.archive_csv:
        archive_images(args)

    log.finished()


def archive_images(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.archive_csv.open() as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            subdir, *_ = row["state"].split(" ")

            dst_dir = args.output_dir / subdir
            dst_dir.mkdir(parents=True, exist_ok=True)

            src: Path = args.image_dir / subdir / f"{row['id']}.jpg"
            dst: Path = dst_dir / src.name
            copyfile(src, dst)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Ad hoc utilities on images.

            * Copy positive flower images into a separate directory.
            """
        ),
    )

    arg_parser.add_argument(
        "--archive-csv",
        type=Path,
        metavar="PATH",
        help="""A csv file with positive flower images.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        help="""Root directory for images on UFIT.""",
    )

    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="PATH",
        help="""Copy images to this directory.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
