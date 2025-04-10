import csv
from pathlib import Path
from typing import Literal

from datasets import Dataset, Image, NamedSplit

TRAITS = ["flowers", "fruits", "leaves", "buds"]
SPLIT = Literal["train", "val", "test"]
SPLITS = ["train", "val", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

LABELS = ["without", "with"]
ID2LABEL = {str(i): v for i, v in enumerate(LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

REGRESSION = "regression"
SINGLE_LABEL = "single_label_classification"
MULTI_LABEL = "multi_label_classification"
PROBLEM_TYPES = [REGRESSION, SINGLE_LABEL, MULTI_LABEL]


def get_dataset(
    split: str,
    dataset_csv: Path,
    image_dir: Path,
    trait: str,
    *,
    limit=0,
    regression: bool = False,
) -> Dataset:
    recs = get_records(split, dataset_csv, trait, limit=limit)

    images = [str(image_dir / r["name"]) for r in recs]
    labels = [float(r[trait]) if regression else int(r[trait]) for r in recs]
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
