#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

from pylib import log

DELAY = 10  # Seconds to delay between attempts to download an image


def main():
    log.started()
    log.finished()


def parse_args():
    description = """Download images for observations."""

    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--observations",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read from this parquet file containing observations.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        metavar="PATH",
        required=True,
        type=Path,
        help="""Place downloaded images into this directory.""",
    )

    arg_parser.add_argument(
        "--max-images",
        metavar="INT",
        type=int,
        default=1000,
        help="""How many images to download for each classification group.
            Groups are: phylogenetic order x phenology class
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--max-workers",
        metavar="INT",
        type=int,
        default=100,
        help="""How many worker threads to spawn. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--image-size",
        choices=["medium", "large", "original"],
        default="medium",
        help="""Download image size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--attempts",
        metavar="INT",
        type=int,
        default=3,
        help="""How many times to try downloading the image.
          (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print result categories instead of a tqdm status bar.",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
