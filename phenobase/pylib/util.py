import csv
from enum import StrEnum
from pathlib import Path
from typing import Literal

from datasets import Dataset, Image, NamedSplit

TRAITS = ["flowers", "fruits", "leaves", "buds"]
SPLIT = Literal["train", "val", "test"]
SPLITS = ["train", "val", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

LABELS = ["without", "with"]
WITH = [0.0, 1.0]
WITHOUT = [1.0, 0.0]
ID2LABEL = {str(i): v for i, v in enumerate(LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class ProblemType(StrEnum):
    REGRESSION = "regression"
    SINGLE_LABEL = "single_label_classification"
    MULTI_LABEL = "multi_label_classification"


PROBLEM_TYPES = [
    ProblemType.REGRESSION,
    ProblemType.SINGLE_LABEL,
    ProblemType.MULTI_LABEL,
]


def get_dataset(
    split: str,
    dataset_csv: Path,
    image_dir: Path,
    trait: str,
    problem_type: ProblemType,
    *,
    limit=0,
) -> Dataset:
    recs = get_records(split, dataset_csv, trait, limit=limit)

    images = [str(image_dir / r["name"]) for r in recs]
    if problem_type == ProblemType.REGRESSION:
        labels = [float(r[trait]) for r in recs]
    elif problem_type == ProblemType.MULTI_LABEL:
        labels = [WITHOUT if r[trait] == "0" else WITH for r in recs]
    else:
        labels = [int(r[trait]) for r in recs]

    ids = [r["name"] for r in recs]

    dataset = Dataset.from_dict(
        {"image": images, "label": labels, "id": ids}, split=NamedSplit(split)
    ).cast_column("image", Image())

    return dataset


def get_records(split: str, dataset_csv: Path, trait: str, *, limit=0) -> list[dict]:
    with dataset_csv.open() as f:
        reader = csv.DictReader(f)
        recs = list(reader)

    recs = [
        r for r in recs if r["split"] == split and r.get(trait) and r[trait] in "01"
    ]
    recs = recs[:limit] if limit else recs
    return recs
