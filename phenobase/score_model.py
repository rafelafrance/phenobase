#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import pandas as pd
import torch
from pylib import util
from pylib.datasets.labeled_dataset import LabeledDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    data = []
    with torch.no_grad():
        for sheets in tqdm(loader):
            images = sheets["pixel_values"].to(device)
            preds = model(images)
            preds = torch.sigmoid(preds.logits)
            preds = preds.detach().cpu()
            for pred, actual, name in zip(
                preds, sheets["labels"], sheets["name"], strict=False
            ):
                rec = {
                    "name": name,
                    "pretrained": args.pretrained_dir,
                }

                pred = pred.tolist()
                actual = actual.tolist()
                for a, p, t in zip(actual, pred, args.trait, strict=False):
                    rec[f"{t}_pred"] = p
                    rec[f"{t}_actual"] = a
                data.append(rec)

    df = pd.DataFrame(data)
    mode, header = "w", True
    if args.output_csv.exists():
        mode, header = "a", False
    df.to_csv(args.output_csv, mode=mode, header=header, index=False)


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
        "--output-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output inference results to this CSV.""",
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
