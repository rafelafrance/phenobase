import csv
import sqlite3
from pathlib import Path

from datasets import Dataset, Image, NamedSplit, Split

LABEL_VALUES = {"0": 0.0, "1": 1.0, "U": 0.5, "u": 0.5}


def get_dataset(
    split: NamedSplit,
    dataset_csv: Path,
    image_dir: Path,
    trait: str,
    *,
    use_unknowns: bool = False,
) -> Dataset:
    recs = get_records(split, dataset_csv, trait, use_unknowns=use_unknowns)

    labels = [LABEL_VALUES[r[trait]] for r in recs]

    images = [str(image_dir / r["name"]) for r in recs]
    ids = [r["name"] for r in recs]

    split = Split.TRAIN if split == "train" else Split.VALIDATION

    dataset = Dataset.from_dict(
        {"image": images, "label": labels, "id": ids}, split=split
    ).cast_column("image", Image())

    return dataset


def get_records(
    split: NamedSplit, dataset_csv: Path, trait: str, *, use_unknowns: bool = False
) -> list[dict]:
    with dataset_csv.open() as inp:
        reader = csv.DictReader(inp)
        recs = list(reader)

    recs = [r for r in recs if r["split"] == split]

    keeps = "01Uu" if use_unknowns else "01"
    recs = [r for r in recs if r[trait] and r[trait] in keeps]

    return recs


def get_inference_records(db, limit, offset):
    with sqlite3.connect(db) as cxn:
        cxn.row_factory = sqlite3.Row
        sql = """select gbifid, tiebreaker, state, family
            from multimedia join occurrence using (gbifid)
            limit ? offset ?"""
        rows = [dict(r) for r in cxn.execute(sql, (limit, offset))]
    return rows


def filter_bad_images(rows):
    rows = [r for r in rows if not r["state"].endswith("error")]
    return rows


def filter_bad_families(rows, bad_family_csv):
    bad_families = []
    if bad_family_csv:
        with bad_family_csv.open() as bad:
            reader = csv.DictReader(bad)
            bad_families = [r["family"].lower() for r in reader]
    rows = [r for r in rows if r["family"].lower() not in bad_families]
    return rows


def get_inference_dataset(rows, image_dir):
    images = []
    ids = []
    families = []
    for row in rows:
        ids.append(row["gbifID"])
        families.append(row["family"])

        parts = row["state"].split()

        name = f"{row['gbifID']}_{row['tiebreaker']}"
        name += f"_{parts[-1]}" if len(parts) > 1 else ""

        path = image_dir / parts[0] / (name + ".jpg")

        images.append(str(path))

    dataset = Dataset.from_dict(
        {"image": images, "id": ids, "path": images, "family": families}
    ).cast_column("image", Image())

    return dataset
