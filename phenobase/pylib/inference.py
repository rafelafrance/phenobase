import logging
import sqlite3
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from datasets import Dataset, Image
from phenobase.pylib import gbif, util

BATCH = 1_000_000


def get_inference_records(db, limit, offset) -> list[gbif.GbifRec]:
    with sqlite3.connect(db) as cxn:
        cxn.row_factory = sqlite3.Row
        sql = """
            select gbifID, tiebreaker, state, family, genus, scientificName
            from multimedia join occurrence using (gbifID)
            limit ? offset ?"""
        records = [gbif.GbifRec(r) for r in cxn.execute(sql, (limit, offset))]
    return records


def get_inference_dataset(records, image_dir, *, debug: bool = False) -> Dataset:
    images = []
    ids = []
    for rec in records:
        if debug:
            path = rec.local_path(image_dir)
            if not path.exists():
                continue
        else:
            path = rec.hipergator_path(image_dir)
            if path.stat().st_size < util.TOO_DAMN_SMALL:
                logging.warning(f"Bad image file {path.stem}")
                continue

        ids.append(rec.stem)
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
    records: list[gbif.GbifRec],
    checkpoint,
    problem_type,
    image_dir,
    image_size,
    trait,
    output_csv,
    *,
    debug: bool = False,
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

    base_recs = {r.id: r.as_dict() for r in records}

    dataset = get_inference_dataset(records, image_dir, debug=debug)

    loader = get_data_loader(dataset, image_size)

    logging.info("Starting inference")

    output = []
    mode = "w"
    header = True

    with torch.no_grad():
        for sheet in tqdm(loader):
            image = sheet["image"].to(device)
            result = model(image)

            if problem_type == util.ProblemType.REGRESSION:
                scores = torch.sigmoid(result.logits)
                score = scores[0, 0]
            else:
                # I can use softmax and take the positive class for the predicted value
                scores = softmax(result.logits)
                score = scores[0, 1]

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
    logging.info("Finished inference")
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
        "--debug",
        action="store_true",
        help="""Use when debugging locally.""",
    )

    args = arg_parser.parse_args()

    return args
