#!/usr/bin/env python3

import argparse
import csv
import logging
import sqlite3
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from phenobase.pylib import log


@dataclass
class Stats:
    total: int = 0
    event_date: int = 0
    lat_long: int = 0
    good: int = 0
    pos: int = 0
    neg: int = 0
    equiv: int = 0
    dups: int = 0
    families: set[str] = field(default_factory=set)
    genera: set[str] = field(default_factory=set)

    def positive_only(self, row: dict) -> int:
        result = 1 if row["winner"] and float(row["winner"]) == 1.0 else 0
        return result

    def has_event_date(self, data: dict) -> int:
        result = 1 if data["eventDate"] else 0
        self.event_date += result
        return result

    def has_lat_long(self, data: dict) -> int:
        result = 1 if data["decimalLatitude"] and data["decimalLatitude"] else 0
        self.lat_long += result
        return result

    def add_taxa(self, family: str, genus: str) -> None:
        self.families.add(family)
        self.genera.add(genus)

    def log_results(self) -> None:
        logging.info(f"Total records with votes     = {self.total:12,d}")
        logging.info(f"Missing event date           = {self.event_date:12,d}")
        logging.info(f"Missing lat/long             = {self.lat_long:12,d}")
        logging.info(f"Duplicate gbifid/tiebreaker  = {self.dups:12,d}")
        logging.info(f"Good records                 = {self.good:12,d}")
        logging.info(f"Positive records             = {self.pos:12,d}")
        logging.info(f"Negative records             = {self.neg:12,d}")
        logging.info(f"Equivocal records            = {self.equiv:12,d}")
        logging.info(f"Unique families              = {len(self.families)}")
        logging.info(f"Unique genera                = {len(self.genera)}")


def main(args: argparse.Namespace) -> None:
    log.started(args=args)

    sql = """
        select * from multimedia join occurrence using (gbifid)
        where gbifid = ? and tiebreaker = ?
        """

    records: list[dict] = []
    unique: set[str] = set()

    base_url = "https://www.gbif.org/occurrence/"

    stats = Stats()

    with sqlite3.connect(args.gbif_db) as cxn, args.winners_csv.open() as f:
        cxn.row_factory = sqlite3.Row

        reader = csv.DictReader(f)

        for row in tqdm(reader):
            if args.limit and stats.total >= args.limit:
                break

            stats.total += 1

            gbif_id, tiebreaker = row["id"].split("_")

            data = dict(cxn.execute(sql, (gbif_id, tiebreaker)).fetchone())

            trait = format_trait(row, stats)  # Count trait before skipping records

            keep = stats.positive_only(row)
            keep &= stats.has_event_date(data)
            keep &= stats.has_lat_long(data)

            if row["id"] in unique:
                stats.dups += 1
                keep = False

            if not keep:
                continue

            stats.good += 1

            unique.add(row["id"])

            stats.add_taxa(data["family"], data["genus"])

            data_source = "GBIF"
            if data.get("institutionCode"):
                data_source += "-" + data["institutionCode"]

            rec = {
                "dataSource": data_source,
                "scientificName": data["scientificName"],
                "trait": trait,
                "family": data["family"],
                "year": data["year"],
                "dayOfYear": data["startDayOfYear"],
                "latitude": data["decimalLatitude"],
                "longitude": data["decimalLongitude"],
                "observedMetadataUrl": f"{base_url}{data['gbifID']}",
                "annotationID": uuid.uuid5(uuid.NAMESPACE_URL, data["identifier"]),
                "annotationMethod": "machine",
                "occurrenceID": gbif_id,
                "organismID": data["organismID"],
                "genus": data["genus"],
                "date": data["eventDate"],
                "recordedBy": data["recordedBy"],
                "coordinateUncertaintyInMeters": data["coordinateUncertaintyInMeters"],
                "verbatimTrait": row["trait"],
                "modelUri": args.model_uri,
            }

            records.append(rec)

            if args.filter_limit and stats.good >= args.filter_limit:
                break

    stats.log_results()

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    log.finished()


def format_trait(row: dict, stats: Stats) -> str:
    match row["winner"]:
        case "1.0":
            present = "present"
            stats.pos += 1
        case "0.0":
            present = "absent"
            stats.neg += 1
        case _:
            present = "unknown"
            stats.equiv += 1
    return f"{row['trait']} {present}"


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Format ensemble results for use in other databases."""
        ),
    )

    arg_parser.add_argument(
        "--winners-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""The CSV file that contains all of the ensemble winners.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output the formatted results to this CSV file.""",
    )

    arg_parser.add_argument(
        "--gbif-db",
        type=Path,
        required=True,
        metavar="PATH",
        help="""This SQLite DB with information about herbarium sheets.""",
    )

    arg_parser.add_argument(
        "--model-uri",
        default="Link TBD",
        metavar="DOI",
        help="""Where is the ensemble stored.""",
    )

    arg_parser.add_argument(
        "--filter-limit",
        type=int,
        default=0,
        metavar="INT",
        help="""Limit to this many filtered records. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="INT",
        help="""Limit to this many total records. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
