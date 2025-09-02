#!/usr/bin/env python3

import argparse
import csv
import sqlite3
import textwrap
import uuid
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from phenobase.pylib import log


def main(args):
    log.started(args=args)

    sql = """
        select * from multimedia join occurrence using (gbifid)
        where gbifid = ? and tiebreaker = ?
        """

    records = []

    base_url = "https://www.gbif.org/occurrence/"

    mode = "w"
    header = True
    if args.output_csv.exists() and args.append:
        mode = "a"
        header = False

    with sqlite3.connect(args.gbif_db) as cxn, args.winners_csv.open() as f:
        cxn.row_factory = sqlite3.Row

        reader = csv.DictReader(f)

        for i, row in enumerate(tqdm(reader)):
            formatted_trait = format_trait(row, args.trait, args.results)
            if not formatted_trait:
                continue

            data = dict(cxn.execute(sql, (row["gbifid"], row["tiebreaker"])).fetchone())

            if args.date_required and not data["eventDate"]:
                continue

            if args.location_required and not (
                data["decimalLatitude"] and data["decimalLatitude"]
            ):
                continue

            rec = {
                "dataSource": args.data_source,
                "scientificName": data["scientificName"],
                "trait": formatted_trait,
                "family": data["family"],
                "year": data["year"],
                "dayOfYear": data["startDayOfYear"],
                "latitude": data["decimalLatitude"],
                "longitude": data["decimalLongitude"],
                "observedMetadataUrl": f"{base_url}{data['gbifID']}",
                "annotationID": uuid.uuid5(uuid.NAMESPACE_URL, data["identifier"]),
                "annotationMethod": "machine",
                "occurrenceID": data["gbifID"],
                "organismID": data["organismID"],
                "genus": data["genus"],
                "date": data["eventDate"],
                "recordedBy": data["recordedBy"],
                "coordinateUncertaintyInMeters": data["coordinateUncertaintyInMeters"],
                "verbatimTrait": args.trait,
                "modelUri": args.model_uri,
            }

            records.append(rec)

            if args.limit and i >= args.limit:
                break

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, mode=mode, header=header, index=False)

    log.finished()


def format_prediction_class(row):
    if row["winner"] == "1.0":
        return "Detected"
    if row["winner"] == "0.0":
        return "Not detected"
    return ""


def format_prediction_probability(row):
    count = sum(1 for v in row["votes"] if v == row["winner"])
    return 1.0 if count == 3 else 0.67


def format_trait(row, trait, results):
    code = row["winner"]
    formatted_trait = ""
    key = f"{results} {code}"
    match key:
        case "positive 1.0":
            formatted_trait = f"{trait} present"
        case "negative 0.0":
            formatted_trait = f"{trait} absent"
        case "pos/neg 1.0":
            formatted_trait = f"{trait} present"
        case "pos/neg 0.0":
            formatted_trait = f"{trait} absent"
        case "all 1.0":
            formatted_trait = f"{trait} present"
        case "all 0.0":
            formatted_trait = f"{trait} absent"
        case "all nan":
            formatted_trait = f"{trait} unknown"

    return formatted_trait


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
        "--results",
        choices=["positive", "negative", "pos/neg", "all"],
        default="positive",
        help="""What results to include in the output. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--date-required",
        action="store_true",
        help="""Only include records that have event dates.""",
    )

    arg_parser.add_argument(
        "--location-required",
        action="store_true",
        help="""Only include records that have decimal latitudes and longitudes.""",
    )

    arg_parser.add_argument(
        "--append",
        action="store_true",
        help="""Append formatted records to an already existing output CSV file.""",
    )

    arg_parser.add_argument(
        "--data-source",
        default="gbif",
        help="""Where is the ensemble stored.""",
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
        help="""Limit to this many records, 0 = unlimited. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
