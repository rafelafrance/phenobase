#!/usr/bin/env python3

import argparse
import csv
import textwrap
from pathlib import Path

import pandas as pd


def main():
    args = parse_args()

    flowers, fruits = get_expert_data(args.ant_csv)

    df = pd.read_csv(args.in_csv)
    df = df.set_index("name")

    df = df.merge(flowers.rename("old_flowers"), left_index=True, right_index=True)
    df = df.merge(fruits.rename("old_fruits"), left_index=True, right_index=True)

    df = df.fillna("")
    df["name"] = df.index

    df.to_csv(args.out_csv, index=False)


def get_expert_data(ant_csvs: list[Path]):
    flowers, fruits = {}, {}
    for csv_file in ant_csvs:
        with csv_file.open() as inf:
            reader = csv.DictReader(inf)
            for row in reader:
                if flower := row.get("flowers"):
                    flowers[row["file"]] = flower
                if fruit := row.get("fruits"):
                    fruits[row["file"]] = fruit
    return pd.Series(flowers), pd.Series(fruits)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """A one-off script foe appending the orginal trait values for traits
            removed via bad families."""
        ),
    )

    arg_parser.add_argument(
        "--ant-csv",
        metavar="PATH",
        type=Path,
        required=True,
        action="append",
        help="""These input CSV files hold expert classifications of the traits.
            Use this argument once for every ant CSV file.""",
    )

    arg_parser.add_argument(
        "--in-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""The original split CSV file.""",
    )

    arg_parser.add_argument(
        "--out-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""The new split CSV file with added columns.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
