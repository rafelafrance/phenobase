#!/usr/bin/env python3
import argparse
import csv
import textwrap
from pathlib import Path

from phenobase.pylib import log


def main():
    log.started()
    args = parse_args()

    with args.split_csv.open() as in_file:
        reader = csv.DictReader(in_file)
        good = {Path(r["path"]).stem for r in reader}

    for path in args.image_dir.glob("*"):
        if path.stem not in good:
            path.rename(args.cull_dir / path.name)

    log.finished()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent("""Cull bad images from the image directory."""),
    )

    arg_parser.add_argument(
        "--image-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read images for the MAE in this directory.""",
    )

    arg_parser.add_argument(
        "--cull-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Move bad images into this directory.""",
    )

    arg_parser.add_argument(
        "--split-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get the MAE splits to this CSV file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
