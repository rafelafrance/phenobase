#!/usr/bin/env python3

import argparse
import csv
import logging
import sqlite3
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from phenobase.pylib import log


@dataclass
class Stats:
    total: int = 0
    matching_vote: int = 0
    event_date: int = 0
    lat_long: int = 0
    good: int = 0

    def positive_only(self, row: dict) -> int:
        result = 1 if row["winner"] and float(row["winner"]) == 1.0 else 0
        self.matching_vote += result
        return result

    def negative_only(self, row: dict) -> int:
        result = 1 if row["winner"] and float(row["winner"]) == 0.0 else 0
        self.matching_vote += result
        return result

    def has_event_date(self, data: dict) -> int:
        result = 1 if data["eventDate"] else 0
        self.event_date += result
        return result

    def has_lat_long(self, data: dict) -> int:
        result = data["decimalLatitude"] and data["decimalLatitude"]
        self.lat_long += result
        return result

    def missing_event_date(self, data: dict) -> int:
        result = 1 if not data["eventDate"] else 0
        self.event_date += result
        return result

    def missing_lat_long(self, data: dict) -> int:
        result = not (data["decimalLatitude"] and data["decimalLatitude"])
        self.lat_long += result
        return result

    def log_results(self, subset) -> None:
        logging.info(f"Total records                = {self.total:12,d}")
        logging.info(f"Matching vote {subset:14} = {self.matching_vote:12,d}")
        logging.info(f"Missing event date           = {self.event_date:12,d}")
        logging.info(f"Missing lat/long             = {self.lat_long:12,d}")
        logging.info(f"Good records                 = {self.good:12,d}")


def main(args):
    log.started(args=args)

    sql = """
        select * from multimedia join occurrence using (gbifid)
        where gbifid = ? and tiebreaker = ?
        """

    records = []

    base_url = "https://www.gbif.org/occurrence/"

    stats = Stats()

    with sqlite3.connect(args.gbif_db) as cxn, args.winners_csv.open() as f:
        cxn.row_factory = sqlite3.Row

        reader = csv.DictReader(f)

        for row in tqdm(reader):
            stats.total += 1

            gbif_id, tiebreaker = row["id"].split("_")

            data = dict(cxn.execute(sql, (gbif_id, tiebreaker)).fetchone())

            match args.subset:
                case "positives_only":
                    keep = stats.positive_only(row)
                    keep &= stats.has_event_date(data)
                    keep &= stats.has_lat_long(data)

                case "missing_data":
                    keep = stats.positive_only(row)
                    keep &= stats.missing_event_date(data) or stats.missing_lat_long(
                        data
                    )

                case _:
                    keep = False

            if not keep:
                continue

            stats.good += 1

            rec = {
                "dataSource": args.data_source,
                "scientificName": data["scientificName"],
                "trait": format_trait(row),
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

            if args.limit and stats.good >= args.limit:
                break

    stats.log_results(args.subset)

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    log.finished()


def format_trait(row) -> str:
    present = "unknown"
    match row["winner"]:
        case "1.0":
            present = "present"
        case "0.0":
            present = "absent"
    return f"{row['trait']} {present}"


def parse_args():
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
        "--subset",
        choices=["positives_only", "missing_data"],
        default="positives_only",
        help="""What results to include in the output. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--data-source",
        default="gbif",
        help="""Where did we get the raw data.""",
    )

    arg_parser.add_argument(
        "--model-uri",
        default="Zenodo link TBD",
        metavar="URI",
        help="""Where is the ensemble stored.""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="INT",
        help="""Limit to this many filtered records. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
