#!/usr/bin/env python3

import argparse
import csv
import textwrap
from pathlib import Path


def main():
    args = parse_args()

    with args.split_csv.open() as f:
        reader = csv.DictReader(f)
        ori = [dict(r) for r in reader]

    with args.remove_csv.open() as f:
        reader = csv.DictReader(f)
        remove = {r["family"] for r in reader}

    rem = [r for r in ori if r["family"] not in remove]

    if args.output_csv:
        fields = next(list(r.keys()) for r in rem)
        with args.output_csv.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in rem:
                writer.writerow(rec)

    print(f"Original total rows: {len(ori)}")
    print(f"Filtered total rows: {len(rem)}")
    print()

    print(f"Original flowers total   {len([r for r in ori if r['flowers'] in '01'])}")
    print(f"Filtered flowers total   {len([r for r in rem if r['flowers'] in '01'])}")
    print()
    print(f"Original flowers present {len([r for r in ori if r['flowers'] == '1'])}")
    print(f"Filtered flowers present {len([r for r in rem if r['flowers'] == '1'])}")
    print()
    print(f"Original flowers absent  {len([r for r in ori if r['flowers'] == '0'])}")
    print(f"Filtered flowers absent  {len([r for r in rem if r['flowers'] == '0'])}")
    print()

    print(f"Original fruits total    {len([r for r in ori if r['fruits'] in '01'])}")
    print(f"Filtered fruits total    {len([r for r in rem if r['fruits'] in '01'])}")
    print()
    print(f"Original fruits present  {len([r for r in ori if r['fruits'] == '1'])}")
    print(f"Filtered fruits present  {len([r for r in rem if r['fruits'] == '1'])}")
    print()
    print(f"Original fruits absent   {len([r for r in ori if r['fruits'] == '0'])}")
    print(f"Filtered fruits absent   {len([r for r in rem if r['fruits'] == '0'])}")
    print()

    print(f"Original leaves total    {len([r for r in ori if r['leaves'] in '01'])}")
    print(f"Filtered leaves total    {len([r for r in rem if r['leaves'] in '01'])}")
    print()
    print(f"Original leaves present  {len([r for r in ori if r['leaves'] == '1'])}")
    print(f"Filtered leaves present  {len([r for r in rem if r['leaves'] == '1'])}")
    print()
    print(f"Original leaves absent   {len([r for r in ori if r['leaves'] == '0'])}")
    print(f"Filtered leaves absent   {len([r for r in rem if r['leaves'] == '0'])}")
    print()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Remove a set of families from the split data."""
        ),
    )

    arg_parser.add_argument(
        "--split-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""The split data with attached metadata.""",
    )

    arg_parser.add_argument(
        "--remove-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""The list of families to remove.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="""Output filtered split data to this file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
