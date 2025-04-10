#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pylib import log, util
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from phenobase.pylib.util import get_records

TEST_XFORMS: Callable = None


def main(args):
    global TEST_XFORMS

    log.started()

    records = get_records("test", args.dataset_csv, args.trait)

    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    checkpoints = [p for p in args.model_dir.glob("checkpoint-*") if p.is_dir()]
    for checkpoint in sorted(checkpoints):
        print(checkpoint)

        model = AutoModelForImageClassification.from_pretrained(
            str(checkpoint),
            num_labels=1 if args.regression else 2,
        )
        model.to(device)

        dataset = util.get_dataset(
            "test",
            args.dataset_csv,
            args.image_dir,
            args.trait,
            regression=args.regression,
        )

        TEST_XFORMS = v2.Compose(
            [
                v2.Resize((args.image_size, args.image_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
            ]
        )

        dataset.set_transform(test_transforms)

        loader = DataLoader(dataset, batch_size=1, pin_memory=True)
        total = len(loader)
        correct = 0
        output = []

        with torch.no_grad():
            for item, sheet in tqdm(zip(loader, records, strict=True)):
                pixels = item["pixel_values"].to(device)
                result = model(pixels)

                out = deepcopy(sheet)
                out |= {"checkpoint": checkpoint}

                if args.regression:
                    true = item["labels"].item()
                    pred = torch.sigmoid(result.logits)
                    pred = pred.detach().cpu().tolist()[0]

                    out |= {f"{args.trait}_score": pred[0]}
                    correct += int(round(pred[0]) == true)

                else:
                    # I want scores here vs classes because I do threshold moving later
                    # in an attempt to improve the final results.
                    # If there are 2 classes of neg/pos we can use softmax and take
                    # the positive class (scores[1]) for the predicted value. I.e.
                    # I treat scores[1] as the "real" score. This will only work when
                    # there are exactly two classes and only when they are organized as
                    # util.LABELS = [without, with].
                    scores = softmax(result.logits).detach().cpu().tolist()[0]
                    out |= {f"{args.trait}_score": scores[1]}

                    true = torch.argmax(sheet["label"].clone().detach()).item()
                    pred = np.argmax(result.logits.detach().cpu(), axis=-1).item()
                    correct += int(pred == true)

                output.append(out)

        print(f"Accuracy {correct}/{total} = {(correct / total):0.3f}\n")

        if args.score_csv:
            df = pd.DataFrame(records)
            if args.score_csv.exists():
                df_old = pd.read_csv(args.score_csv, low_memory=False)
                df = pd.concat((df_old, df))
            df.to_csv(args.score_csv, index=False)

    log.finished()


def test_transforms(examples):
    pixel_values = [TEST_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


# def get_dataset(split: str, dataset_csv: Path, image_dir: Path, limit=0) -> Dataset:
#     with dataset_csv.open() as inp:
#         reader = csv.DictReader(inp)
#         recs = list(reader)
#
#     recs = [r for r in recs if r["split"] == split]
#     recs = recs[:limit] if limit else recs
#
#     images = [str(image_dir / r["name"]) for r in recs]
#     labels = [int(r["label"]) for r in recs]
#     ids = [r["name"] for r in recs]
#
#     dataset = Dataset.from_dict(
#         {"image": images, "label": labels, "id": ids}, split=NamedSplit(split)
#     ).cast_column("image", Image())
#
#     return dataset


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Score a model trained to classify phenology traits."""
        ),
    )

    arg_parser.add_argument(
        "--dataset-csv",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A CSV file with images and trait labels.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A path to the directory where the images are.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        required=True,
        help="""Score the classification of this trait.""",
    )

    arg_parser.add_argument(
        "--score-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Append scores to this CSV.""",
    )

    arg_parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing the training checkpoints.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--regression",
        action="store_true",
        help="""Handle regression scoring.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
