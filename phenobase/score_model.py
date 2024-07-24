#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import torch
from pylib import util
from pylib.datasets.labeled_dataset import LabeledDataset
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViTForImageClassification.from_pretrained(
        str(args.pretrained_dir),
        num_labels=len(args.trait),
    )

    model.to(device)
    model.eval()

    dataset = LabeledDataset(
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        traits=args.trait,
        split="test",
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True
    )

    with torch.no_grad():
        tp, tn, fp, fn = 0, 0, 0, 0
        for sheets in loader:
            images = sheets["pixel_values"].to(device)
            preds = model(images)
            preds = torch.sigmoid(preds.logits)
            preds = torch.round(preds)
            preds = preds.detach().cpu()
            for pred, actual, name in zip(
                preds, sheets["labels"], sheets["name"], strict=False
            ):
                pred = pred.tolist()
                actual = actual.tolist()
                for a, p in zip(actual, pred, strict=False):
                    tp += int(a == 1.0 and p == 1.0)
                    tn += int(a == 0.0 and p == 0.0)
                    fp += int(a == 0.0 and p == 1.0)
                    fn += int(a == 1.0 and p == 0.0)
                print(f"{name} {actual} {pred}")

        print(f"tp = {tp:3d}  fp = {fp:3d}")
        print(f"fn = {fn:3d}  tn = {tn:3d}")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Test a model trained to classify phenology traits."""
        ),
    )

    arg_parser.add_argument(
        "--trait-csv",
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
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="""Save inference results here.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        action="append",
        required=True,
        help="""Infer this trait. Repeat this argument to infer multiple traits.""",
    )

    arg_parser.add_argument(
        "--pretrained-dir",
        type=Path,
        metavar="PATH",
        help="""The directory containing the pretrained model.""",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="INT",
        help="""Input batch size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="INT",
        help="""Number of workers for loading data. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
