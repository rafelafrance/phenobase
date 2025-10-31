#!/usr/bin/env python3

import argparse
import csv
import random
import sqlite3
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from phenobase.pylib import log, util

FILTER_COUNT = 5
FILTER_RATE = 0.25


@dataclass
class Taxon:
    family: str = ""
    genus: str = ""
    labels: list[str] = field(default_factory=list)

    @property
    def keep(self) -> bool:
        count = len(self.labels)
        unknown = sum(1 for lb in self.labels if lb == "u")
        fract = unknown / count
        return count <= FILTER_COUNT or fract < FILTER_RATE

    @property
    def remove(self) -> bool:
        return not self.keep


def main(args: argparse.Namespace) -> None:
    log.started(args=args)
    random.seed(args.seed)

    gbif_recs = get_expert_gbif_data(args.ant_gbif)
    idigbio_recs = get_expert_idigbio_data(args.ant_idigbio)

    append_gbif_metadata(gbif_recs, args.gbif_db)
    append_idigbio_metadata(idigbio_recs, args.idigbio_db)

    records = gbif_recs + idigbio_recs

    format_taxa(records)

    filter_records(records, "flowers", args.split_csv)
    filter_records(records, "fruits", args.split_csv)

    split_data(records, args.train_split, args.val_split)
    write_csvs(args.split_csv, records)

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


def append_gbif_metadata(records: list[dict], gbif_db: Path) -> None:
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


def append_idigbio_metadata(records: list[dict], idigbio_db: Path) -> None:
    sql = "select * from angiosperms where coreid = ?"
    with sqlite3.connect(idigbio_db) as cxn:
        cxn.row_factory = sqlite3.Row
        for rec in records:
            coreid = rec["file"].split(".")[0]
            data = dict(cxn.execute(sql, (coreid,)).fetchone())
            data = {k.lower(): v for k, v in data.items() if k != "filter_set"}
            rec |= data


def format_taxa(records: list[dict]) -> None:
    for rec in records:
        rec["formatted_family"] = rec["family"].lower()

        genus = rec["genus"] if rec["genus"] else rec["scientificname"].split()[0]
        genus = genus.lower()
        rec["formatted_genus"] = genus


def filter_records(
    records: list[dict], trait: str, split_csv: Path | None = None
) -> None:
    by_genus = group_by_genus(records, trait)
    by_family = group_by_family(by_genus)

    to_remove = {(g.family, g.genus) for g in by_genus if g.remove}
    to_remove |= {(f.family, "") for f in by_family if f.remove}

    for rec in records:
        family = rec["formatted_family"]
        genus = rec["formatted_genus"]
        if (family, genus) in to_remove or (family, "") in to_remove:
            rec[trait] = f"x{rec[trait]}"

    if split_csv:
        removes = [{"family": p[0], "genus": p[1]} for p in to_remove]
        path = split_csv.with_stem(f"remove_{trait}")
        df = pd.DataFrame(removes)
        df.to_csv(path, index=False)


def group_by_genus(records: list[dict], trait: str) -> list[Taxon]:
    by_genus = {}
    for rec in records:
        label = rec[trait].lower()
        if not label or label not in "01u":
            continue
        family = rec["formatted_family"]
        genus = rec["formatted_genus"]
        key = (family, genus)
        if key not in by_genus:
            by_genus[key] = Taxon(family=family, genus=genus)
        by_genus[key].labels.append(label)

    return list(by_genus.values())


def group_by_family(by_genus: list[Taxon]) -> list[Taxon]:
    by_family = {}
    for taxon in by_genus:
        if taxon.keep:
            if taxon.family not in by_family:
                by_family[taxon.family] = Taxon(taxon.family)
            by_family[taxon.family].labels += taxon.labels
    return list(by_family.values())


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


def write_csvs(split_csv: Path, records: list[dict]) -> None:
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
        help="""Get herbarium sheet images from this directory.""",
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
