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

# from collections import defaultdict


def main(args):
    log.started(args=args)

    sql = """
        select * from multimedia join occurrence using (gbifid)
        where gbifid = ? and tiebreaker = ?
        """

    records = []

    # family_counts = defaultdict(int)

    with sqlite3.connect(args.gbif_db) as cxn, args.winners_csv.open() as f:
        cxn.row_factory = sqlite3.Row

        reader = csv.DictReader(f)

        for row in tqdm(reader):
            formatted_trait = format_trait(row, args.trait, args.results)
            if not formatted_trait:
                continue

            data = dict(cxn.execute(sql, (row["gbifid"], row["tiebreaker"])).fetchone())

            rec = {
                "dataSource": "gbif",
                "scientificName": data["scientificName"],
                "trait": formatted_trait,
                "family": data["family"],
                "year": data["year"],
                "dayOfYear": data["startDayOfYear"],
                "latitude": data["decimalLatitude"],
                "longitude": data["decimalLongitude"],
                "observedMetadataUrl": data["identifier"],
                "annotationID": uuid.uuid5(uuid.NAMESPACE_URL, data["identifier"]),
                "annotationMethod": "machine",
                "occurrenceID": data["occurrenceID"],
                "organismID": data["organismID"],
                "genus": data["genus"],
                "taxonRank": data["taxonRank"],
                "date": data["eventDate"],
                "recordedBy": data["recordedBy"],
                "coordinateUncertaintyInMeters": data["coordinateUncertaintyInMeters"],
                "certainty": format_certainty(row),
                "predictionProbability": format_prediction_probability(row),
                "predictionClass": format_prediction_class(row),
                # "proportionCertaintyFamily": data[""],
                # "countFamily": 0,
                # "accuracyFamily": data[""],
            }
            # family_counts[data["family"]] += 1

            records.append(rec)

    df = pd.DataFrame(records)
    mode, header = ("a", False) if args.output_csv.exists() else ("w", True)
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


def format_certainty(row):
    count = sum(1 for v in row["votes"] if v == row["winner"])
    return "High" if count == 3 else "Medium"


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
        case "any 1.0":
            formatted_trait = f"{trait} present"
        case "any 0.0":
            formatted_trait = f"{trait} absent"
        case "any nan":
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
        "--trait",
        choices=["flowers", "fruits", "leaves"],
        default="flowers",
        help="""What trait was classified. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--results",
        choices=["positive", "negative", "pos/neg", "all"],
        default="positive",
        help="""What results to include in the output. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
