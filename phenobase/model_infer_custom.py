#!/usr/bin/env python3

import argparse
import csv
import logging
import textwrap
from pathlib import Path

import pandas as pd
import torch
from pylib import log, util
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from datasets import Dataset, Image

BATCH = 100_000


def main(args):
    log.started(args=args)

    softmax = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    logging.info(f"Device = {device}")

    model = AutoModelForImageClassification.from_pretrained(
        str(args.checkpoint),
        num_labels=1 if args.problem_type == util.ProblemType.REGRESSION else 2,
    )

    model.to(device)
    model.eval()

    records = get_records(args.custom_dataset)
    base_recs = {r["gbifID"]: r for r in records}

    dataset = get_inference_dataset(records, args.image_dir)

    infer_xforms = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
        ]
    )

    dataset.set_transform(infer_xforms)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    output = []
    mode = "w"
    header = True

    logging.info("Starting inference")

    with torch.no_grad():
        for sheet in tqdm(loader):
            image = sheet["image"].to(device)
            result = model(image)

            if args.problem_type == util.ProblemType.REGRESSION:
                score = torch.sigmoid(torch.tensor(result.logits))
            else:
                # I'm filtering on the score.
                # I can use softmax and take the positive class for the predicted value
                scores = softmax(result.logits)
                score = scores[0, 1]

            if args.use_thresholds and args.thresh_low < score < args.thresh_high:
                continue

            rec = base_recs[sheet["id"][0]]
            rec[args.trait] = torch.round(score).item()
            rec[f"{args.trait}_score"] = score.item()
            rec["path"] = sheet["path"][0]
            output.append(rec)

            if len(output) >= BATCH:
                df = pd.DataFrame(output)
                df.to_csv(args.output_csv, mode=mode, header=header, index=False)
                output = []
                mode = "a"
                header = False

    df = pd.DataFrame(output)
    df.to_csv(args.output_csv, mode=mode, header=header, index=False)

    logging.info(f"Remaining {len(output)}")

    log.finished()


def get_records(custom_dataset):
    with custom_dataset.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    total = len(records)
    records = [
        r
        for r in records
        if r["state"].startswith("images")
        and not (r["state"].endswith("error") or r["state"].endswith("small"))
    ]
    good = len(records)
    logging.info(f"Total records {total}, good images {good}")
    return records


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


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Infer phenology traits from a custom dataset."""
        ),
    )

    arg_parser.add_argument(
        "--custom-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A CSV file with GBIF records about the custom herbarium sheets.""",
    )

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
        "--use-thresholds",
        action="store_true",
        help="""Filter sheets based upon the thresholds.""",
    )

    arg_parser.add_argument(
        "--thresh-low",
        type=float,
        default=0.05,
        help="""Low threshold for being in the negative class (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--thresh-high",
        type=float,
        default=0.95,
        help="""high threshold for being in the positive class
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
