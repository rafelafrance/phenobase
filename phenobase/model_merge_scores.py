#!/usr/bin/env python3

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd


def main(args: argparse.Namespace) -> None:
    records: dict[str, dict] = {}

    for path in args.score_dir.glob("*.json"):
        with path.open() as f:
            data = json.load(f)
        model = data["score_args"]["model_dir"].split("/")[-1]
        trait = data["score_args"]["trait"]

        for name, fields in data["records"].items():
            if name not in records:
                records[name] = {k: v for k, v in fields.items() if k != "scores"}
                records[name]["name"] = name
                records[name]["trait"] = trait
                records[name][trait] = float(records[name][trait])

            for checkpoint, score in fields["scores"].items():
                key = f"{model}/{checkpoint}"
                records[name][key] = score

    df = pd.DataFrame(records.values())
    df.to_csv(args.output_csv, index=False)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Merge single model JSON score files into a single CSV file."""
        ),
    )

    arg_parser.add_argument(
        "--score-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing JSON score files to merge.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output merged JSON scores to the CSV file.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
