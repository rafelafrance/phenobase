import csv
import logging
import sqlite3
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from transformers import AutoModelForImageClassification

from datasets import Dataset, Image
from phenobase.pylib import util

BATCH = 100_000


def filter_bad_images(rows) -> list[dict]:
    rows = [
        r
        for r in rows
        if r["state"].startswith("images")
        and not (r["state"].endswith("error") or r["state"].endswith("small"))
    ]
    return rows


def filter_bad_taxa(rows: list[dict], bad_taxa: Path | None = None) -> list[dict]:
    to_remove = []
    if not bad_taxa:
        return rows

    # Get the bad taxa
    with bad_taxa.open() as bad:
        reader = csv.DictReader(bad)
        to_remove = [
            (r["family"].lower(), r["genus"].lower() if r["genus"] else "")
            for r in reader
        ]

    # Filter bad taxa from the rows
    filtered = []
    for row in rows:
        family = row["family"].lower()
        genus = row["genus"] if row["genus"] else row["scientificName"].split()[0]
        genus = genus.lower()
        if (family, genus) in to_remove or (family, "") in to_remove:
            continue
        filtered.append(row)
    return filtered


def get_inference_records(db, limit, offset) -> list[dict]:
    with sqlite3.connect(db) as cxn:
        cxn.row_factory = sqlite3.Row
        sql = """
            select gbifID, tiebreaker, state, family, genus, scientificName
            from multimedia join occurrence using (gbifid)
            limit ? offset ?"""
        rows = [dict(r) for r in cxn.execute(sql, (limit, offset))]
    return rows


def get_inference_dataset(rows, image_dir) -> Dataset:
    images = []
    ids = []
    for row in rows:
        parts = row["state"].split()

        name = f"{row['gbifID']}_{row['tiebreaker']}"
        name += f"_{parts[-1]}" if len(parts) > 1 else ""

        path = image_dir / parts[0] / (name + ".jpg")

        ids.append(row["gbifID"])
        images.append(str(path))

    dataset = Dataset.from_dict(
        {"image": images, "id": ids, "path": images}
    ).cast_column("image", Image())

    return dataset


def get_data_loader(dataset, image_size):
    infer_xforms = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
        ]
    )

    dataset.set_transform(infer_xforms)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)
    return loader


def infer_records(
    records,
    checkpoint,
    problem_type,
    image_dir,
    image_size,
    trait,
    threshold_low,
    threshold_high,
    output_csv,
):
    softmax = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    logging.info(f"Device = {device}")

    model = AutoModelForImageClassification.from_pretrained(
        str(checkpoint),
        num_labels=1 if problem_type == util.ProblemType.REGRESSION else 2,
    )

    model.to(device)
    model.eval()

    base_recs = {r["gbifID"]: r for r in records}

    dataset = get_inference_dataset(records, image_dir)

    loader = get_data_loader(dataset, image_size)

    logging.info("Starting inference")

    output = []
    mode = "w"
    header = True

    with torch.no_grad():
        for sheet in loader:
            image = sheet["image"].to(device)
            result = model(image)

            if problem_type == util.ProblemType.REGRESSION:
                scores = torch.sigmoid(result.logits)
                score = scores[0, 0]
            else:
                # I'm filtering on the score.
                # I can use softmax and take the positive class for the predicted value
                scores = softmax(result.logits)
                score = scores[0, 1]

            if threshold_low < score < threshold_high:
                continue

            rec = base_recs[sheet["id"][0]]
            rec[trait] = torch.round(score).item()
            rec[f"{trait}_score"] = score.item()
            rec["path"] = sheet["path"][0]
            output.append(rec)

            if len(output) >= BATCH:
                df = pd.DataFrame(output)
                df.to_csv(output_csv, mode=mode, header=header, index=False)
                output = []
                mode = "a"
                header = False

    df = pd.DataFrame(output)
    df.to_csv(output_csv, mode=mode, header=header, index=False)
    logging.info(f"Remaining {len(output)}")
    return output


def common_args(arg_parser):
    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A path to the directory where the images are.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output inference results to this CSV.""",
    )

    arg_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing the best checkpoint.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        metavar="INT",
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        required=True,
        help="""Infer this trait.""",
    )

    arg_parser.add_argument(
        "--problem-type",
        choices=util.PROBLEM_TYPES,
        default=util.ProblemType.SINGLE_LABEL,
        help="""What kind of a model are we scoring. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.05,
        help="""Low threshold for being in the negative class (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--threshold-high",
        type=float,
        default=0.95,
        help="""high threshold for being in the positive class
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args
