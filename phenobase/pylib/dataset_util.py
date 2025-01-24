import csv
from pathlib import Path

from datasets import Dataset, Image, Split
from phenobase.pylib import const


def get_num_labels(problem_type: str, traits: list[str]) -> int:
    if problem_type == const.REGRESSION:
        return 1
    if problem_type == const.SINGLE_LABEL:
        return len(const.LABELS)
    if problem_type == const.MULTI_LABEL:
        return len(traits)
    raise ValueError(problem_type)


def get_dataset(
    split: str, dataset_csv: Path, image_dir: Path, traits: list[str], problem_type: str
) -> Dataset:
    trait = traits[0]

    recs = get_records(split, dataset_csv, traits)

    if problem_type == const.REGRESSION:
        labels = [int(r[trait]) for r in recs]

    elif problem_type == const.SINGLE_LABEL:
        labels = [const.WITHOUT if r[trait] == "0" else const.WITH for r in recs]

    elif problem_type == const.MULTI_LABEL:
        labels = [[int(r[t]) for t in traits] for r in recs]

    else:
        raise ValueError(problem_type)

    images = [str(image_dir / r["name"]) for r in recs]
    ids = [r["name"] for r in recs]

    split = Split.TRAIN if split == "train" else Split.VALIDATION
    dataset = Dataset.from_dict(
        {"image": images, "label": labels, "id": ids}, split=split
    ).cast_column("image", Image())

    return dataset


def get_records(split: str, dataset_csv: Path, traits: list[str]) -> list[dict]:
    with dataset_csv.open() as inp:
        reader = csv.DictReader(inp)
        recs = list(reader)

    recs = [r for r in recs if r["split"] == split]

    for trait in traits:
        recs = [r for r in recs if r[trait] in "01"]

    return recs
