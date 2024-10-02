#!/usr/bin/env python3
import argparse
import csv
import sqlite3
import textwrap
from pathlib import Path

from phenobase.pylib import log


def main():
    log.started()
    args = parse_args()

    metadata = get_angiosperm_data(args.angiosperm_db)

    merged = link_metadata(args.splits_csv, metadata)

    write_csv(args.merged_csv, merged)

    log.finished()


def get_angiosperm_data(angiosperm_db) -> dict[str, dict]:
    sql = """select * from angiosperms order by filter_set desc"""
    with sqlite3.connect(angiosperm_db) as cxn:
        cxn.row_factory = sqlite3.Row
        metadata = {row["coreid"]: dict(row) for row in cxn.execute(sql)}
    return metadata


def link_metadata(splits_csv, metadata) -> list[dict]:
    with splits_csv.open() as f:
        reader = csv.DictReader(f)
        merged = [r | metadata[Path(r["file"]).stem] for r in reader]
    return merged


def write_csv(merged_csv: Path, merged: list[dict]) -> None:
    fields = next(list(r.keys()) for r in merged)

    with merged_csv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rec in merged:
            writer.writerow(rec)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent("""Get metadata for split records."""),
    )

    arg_parser.add_argument(
        "--angiosperm-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the angiosperm database.""",
    )

    arg_parser.add_argument(
        "--splits-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""CSV file containing the splits.""",
    )

    arg_parser.add_argument(
        "--merged-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the merged file to this CSV.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
