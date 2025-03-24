#!/usr/bin/env python

import argparse
import csv
import textwrap
from collections import defaultdict
from pathlib import Path

from pylib import log


def main(args):
    log.started()

    flowers = defaultdict(int)
    fruits = defaultdict(int)

    with args.splits_csv.open() as inf:
        reader = csv.DictReader(inf)
        for row in reader:
            if row["old_flowers"] in "01":
                flowers[row["split"]] += 1

            if row["old_fruits"] in "01":
                fruits[row["split"]] += 1

    for split, count in flowers.items():
        print(f"flowers {split:<6} {count: 5,d}")

    for split, count in fruits.items():
        print(f"fruits  {split:<6} {count: 5,d}")

    log.finished()


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """
            Move thresholds to find the best modes and thresholds for each trait.

            Find high and low score thresholds for each trait model that maximize its
            scores. That is we will throw out records that are outside of the
            thresholds because the model wasn't really "sure" about its classifiations.
            """
        ),
    )

    arg_parser.add_argument(
        "--splits-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""This CSV contains all of splits.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
