#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from pylib import const, dataset_util, image_util, log
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageClassification

TEST_XFORMS: Callable | None = None


def main(args):
    global TEST_XFORMS

    log.started()

    base_recs = {
        d["name"]: d
        for d in dataset_util.get_records("test", args.dataset_csv, args.trait)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = [p for p in args.model_dir.glob("checkpoint-*") if p.is_dir()]
    for checkpoint in sorted(checkpoints):
        print(checkpoint)

        model = AutoModelForImageClassification.from_pretrained(
            str(checkpoint),
            num_labels=1,
        )

        model.to(device)
        model.eval()

        dataset = dataset_util.get_dataset(
            "test", args.dataset_csv, args.image_dir, args.trait
        )

        TEST_XFORMS = image_util.build_transforms(args.image_size, augment=False)
        dataset.set_transform(TEST_XFORMS)

        loader = DataLoader(dataset, batch_size=1, pin_memory=True)
        total = len(loader)

        correct = 0
        records = []

        with torch.no_grad():
            for sheet in tqdm(loader):
                image = sheet["image"].to(device)
                result = model(image)

                # Append result record
                rec = deepcopy(base_recs[sheet["id"][0]])
                rec |= {"checkpoint": checkpoint}

                score = torch.sigmoid(result.logits.clone().detach())
                rec |= {f"{args.trait}_score": score.item()}

                true = sheet["label"].item()
                pred = torch.round(score).item()
                correct += int(pred == true)

                records.append(rec)

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
        choices=const.TRAITS,
        required=True,
        help="""Score the classification of this trait.""",
    )

    arg_parser.add_argument(
        "--score-csv",
        type=Path,
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

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
