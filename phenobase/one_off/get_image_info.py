#!/usr/bin/env python3
import argparse
import csv
import sqlite3
import textwrap
from pathlib import Path

from tqdm import tqdm

from phenobase.pylib import log


def main():
    log.started()
    args = parse_args()

    sheet_stems = get_sheet_stems(args.sheets_dir)

    info = link_idigbio_sheets(args.idigbio_db, sheet_stems)

    write_csv(args.info_csv, info)

    log.finished()


def get_sheet_stems(sheets_dir) -> list[str]:
    stems = [s.stem for s in sheets_dir.glob("*")]
    return stems


def link_idigbio_sheets(idigbio_db, sheet_stems) -> dict[str, dict]:
    sql = """select * from multimedia join occurrence using (coreid) where coreid = ?"""

    info = {}
    with sqlite3.connect(idigbio_db) as cxn:
        cxn.row_factory = sqlite3.Row
        for stem in tqdm(sheet_stems):
            res = cxn.execute(sql, (stem,))
            row = res.fetchone()
            info[stem] = dict(row) if row else {}
    return info


def write_csv(info_csv: Path, info: dict[str, dict]) -> None:
    fields = next(list(r.keys()) for r in info.values() if r)

    with info_csv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for coreid, rec in info.items():
            rec = rec if rec else {"coreid": coreid}
            writer.writerow(rec)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        fromfile_prefix_chars="@",
        description=textwrap.dedent("""Cull bad images from the image directory."""),
    )

    arg_parser.add_argument(
        "--idigbio-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Path to the iDigBio database.""",
    )

    arg_parser.add_argument(
        "--sheets-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Read herbarium sheets from this directory.""",
    )

    arg_parser.add_argument(
        "--info-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the information about the images to this CSV.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
