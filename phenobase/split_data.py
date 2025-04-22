#!/usr/bin/env python3

import argparse
import csv
import logging
import random
import sqlite3
import textwrap
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pylib import log, util

FILTER_COUNT = 5
FILTER_RATE = 0.25


def main(args):
    log.started(args=args)
    random.seed(args.seed)

    gbif_recs = get_expert_gbif_data(args.ant_gbif)
    idigbio_recs = get_expert_idigbio_data(args.ant_idigbio)

    append_gbif_metadata(gbif_recs, args.gbif_db)
    append_idigbio_metadata(idigbio_recs, args.idigbio_db)

    records = gbif_recs + idigbio_recs
    msg = f"Total records before filtering = {len(records)}"
    logging.info(msg)

    format_taxa(records)

    filter_records(records, "flowers")
    # filter_records(records, "fruits")

    # split_data(records, args.train_split, args.val_split)
    # missing_images(records)
    # write_csv(args.split_csv, records)

    log.finished()


def get_expert_gbif_data(ant_gbif: Path) -> list[dict]:
    recs: dict[str, dict] = defaultdict(
        lambda: {
            "file": f,
            "db": "gbif",
            "split": "",
            "formatted_genus": "",
            "formatted_family": "",
        }
        | dict.fromkeys(util.TRAITS, "")
    )

    for path in ant_gbif.glob("*.csv"):
        with path.open() as ant:
            reader = csv.DictReader(ant)
            for row in reader:
                f = row["file"]
                if f.find("_small") > -1:
                    continue
                for trait in util.TRAITS:
                    if label := row.get(trait):
                        recs[f][trait] = label

    return list(recs.values())


def get_expert_idigbio_data(ant_idigbio: Path) -> list[dict]:
    recs: dict[str, dict] = defaultdict(
        lambda: {
            "file": f,
            "db": "idigbio",
            "split": "",
            "formatted_genus": "",
            "formatted_family": "",
        }
        | dict.fromkeys(util.TRAITS, "")
    )

    for path in ant_idigbio.glob("*.csv"):
        with path.open() as ant:
            reader = csv.DictReader(ant)
            for row in reader:
                f = row["file"]
                for trait in util.TRAITS:
                    if label := row.get(trait):
                        recs[f][trait] = label

    return list(recs.values())


def append_gbif_metadata(records, gbif_db) -> None:
    sql = """
        select * from multimedia join occurrence using (gbifid)
        where gbifid = ? and tiebreaker = ?
        """
    with sqlite3.connect(gbif_db) as cxn:
        cxn.row_factory = sqlite3.Row
        for rec in records:
            name = rec["file"].split(".")[0]
            gbifid, tiebreaker, *_ = name.split("_")
            data = dict(cxn.execute(sql, (gbifid, tiebreaker)).fetchone())
            data = {k.lower(): v for k, v in data.items()}
            rec |= data


def append_idigbio_metadata(records, idigbio_db) -> None:
    sql = "select * from angiosperms where coreid = ?"
    with sqlite3.connect(idigbio_db) as cxn:
        cxn.row_factory = sqlite3.Row
        for rec in records:
            coreid = rec["file"].split(".")[0]
            data = dict(cxn.execute(sql, (coreid,)).fetchone())
            data = {k.lower(): v for k, v in data.items() if k != "filter_set"}
            rec |= data


def format_taxa(records):
    for rec in records:
        rec["formatted_family"] = rec["family"].lower()

        genus = rec["genus"] if rec["genus"] else rec["scientificname"].split()[0]
        genus = genus.lower()
        rec["formatted_genus"] = genus


def filter_records(records, trait):
    by_genus = group_by_genus(records, trait)
    genus_keep, genus_remove, genus_out = filter_by_genus(by_genus)

    by_family = group_by_family(genus_keep)
    family_keep, family_remove, family_out = filter_by_family(by_family)

    out = sorted(genus_out) + sorted(family_out)
    for ln in out:
        print(ln)

    total = sum(len(g) for g in by_genus.values())

    genus_kept = sum(len(k) for k in genus_keep.values())
    genus_removed = sum(len(r) for r in genus_remove.values())

    family_kept = sum(len(k) for k in family_keep.values())
    family_removed = sum(len(r) for r in family_remove.values())
    total_removed = genus_removed + family_removed
    percent_removed = total_removed / total * 100.0

    print(f"total count {total}")
    print(f"genus kept {genus_kept}")
    print(f"genus removed {genus_removed}")
    print(f"family kept {family_kept}")
    print(f"family removed {family_removed}")
    print(f"total removed {total_removed}")
    print(f"percent removed {percent_removed:5.1f} %")

    return records


def group_by_genus(records, trait):
    by_genus = defaultdict(list)
    for rec in records:
        label = rec[trait].lower()
        if not label or label not in "u01":
            continue
        by_genus[(rec["formatted_family"], rec["formatted_genus"])].append(label)

    by_genus = dict(sorted(by_genus.items()))
    return by_genus


def group_by_family(genus_keep):
    by_family = defaultdict(list)
    for (family, _), labels in genus_keep.items():
        by_family[family] += labels
    return by_family


def filter_by_genus(by_genus):
    genus_keep = {}
    genus_remove = {}
    genus_out = []

    for (family, genus), labels in by_genus.items():
        unknown = sum(1 for lb in labels if lb == "u")
        fract = unknown / len(labels)
        percent = fract * 100.0

        if len(labels) <= FILTER_COUNT:
            genus_out.append(
                f"keep genus {family} {genus} {percent:5.1f} {' '.join(labels)}"
            )
            genus_keep[(family, genus)] = labels
        elif fract < FILTER_RATE:
            genus_out.append(
                f"keep genus {family} {genus} {percent:5.1f} {' '.join(labels)}"
            )
            genus_keep[(family, genus)] = labels
        else:
            genus_out.append(
                f"remove genus {family} {genus} {percent:5.1f} {' '.join(labels)}"
            )
            genus_remove[(family, genus)] = labels

    return genus_keep, genus_remove, genus_out


def filter_by_family(by_family):
    family_keep = {}
    family_remove = {}
    family_out = []

    for family, labels in by_family.items():
        unknown = sum(1 for lb in labels if lb == "u")
        fract = unknown / len(labels)
        percent = fract * 100.0

        if len(labels) <= FILTER_COUNT:
            family_out.append(f"keep family {family} {percent:5.1f} {' '.join(labels)}")
            family_keep[family] = labels
        elif fract < FILTER_RATE:
            family_out.append(f"keep family {family} {percent:5.1f} {' '.join(labels)}")
            family_keep[family] = labels
        else:
            family_out.append(
                f"remove family {family} {percent:5.1f} {' '.join(labels)}"
            )
            family_remove[family] = labels

    return family_keep, family_remove, family_out


def missing_images(records):
    _ = records


def split_data(records: list[dict], train_split: float, val_split: float) -> None:
    random.shuffle(records)

    total = len(records)
    split1 = round(total * train_split)
    split2 = split1 + round(total * val_split)

    for i, rec in enumerate(records):
        if i < split1:
            rec["split"] = "train"
        elif i < split2:
            rec["split"] = "val"
        else:
            rec["split"] = "test"


def write_csv(split_csv: Path, records: list[dict]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(split_csv, index=False)


def validate_splits(args: argparse.Namespace) -> None:
    splits = (args.train_split, args.val_split, args.test_split)

    if sum(splits) != 1.0:
        msg = "train, val, and test splits must sum to 1.0"
        raise ValueError(msg)

    if any(s < 0.0 or s > 1.0 for s in splits):
        msg = "All splits must be in the interval [0.0, 1.0]"
        raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Split the data into training, testing, & validation datasets."""
        ),
    )

    arg_parser.add_argument(
        "--ant-gbif",
        type=Path,
        metavar="PATH",
        help="""This directory contains all of the expert classifications from the
            GBIF database.""",
    )

    arg_parser.add_argument(
        "--ant-idigbio",
        type=Path,
        metavar="PATH",
        help="""This directory contains all of the expert classifications from the
            iDigBio database.""",
    )

    arg_parser.add_argument(
        "--gbif-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get metadata for classifications from the GBIF DB.""",
    )

    arg_parser.add_argument(
        "--idigbio-db",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get metadata for classifications from the iDigBio DB.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Get herbarium sheet images from this directory.""",
    )

    arg_parser.add_argument(
        "--filter-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Put lists of genera and families to filter in this directory.""",
    )

    arg_parser.add_argument(
        "--split-csv",
        metavar="PATH",
        type=Path,
        required=True,
        help="""Output the split data to thisCSV file.""",
    )

    arg_parser.add_argument(
        "--train-split",
        type=float,
        metavar="FRACTION",
        default=0.6,
        help="""What fraction of records to use for training the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--val-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for validating the model between epochs.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--test-split",
        type=float,
        metavar="FRACTION",
        default=0.2,
        help="""What fraction of records to use for testing the model.
            (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=2114987,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    validate_splits(args)

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
