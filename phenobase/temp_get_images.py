#!/usr/bin/env python3

import argparse
import csv
import textwrap
from pathlib import Path


def main(args):
    src = "/blue/guralnick/share/phenobase_specimen_data/images"
    dst = "/home/rafe.lafrance/blue/phenobase/data/images/xfer"

    with args.dataset_csv.open() as f:
        reader = csv.DictReader(f)
        missing = [r for r in reader if not (args.image_dir / r["name"]).exists()]

    for m in missing:
        dir_ = m["state"].split()[0]
        print(f"cp {src}/{dir_}/{m['name']} {dst}")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Get images."""),
    )

    arg_parser.add_argument(
        "--dataset-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A CSV file with images and trait labels.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A path to the directory where the images are.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
