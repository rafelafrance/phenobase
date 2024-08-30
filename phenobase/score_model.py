#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import pandas as pd
import torch
from pylib import util
from pylib.labeled_dataset import LabeledDataset
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for checkpoint in sorted(args.pretrained_dir.glob("**/checkpoint-*")):
        print(checkpoint)

        model = ViTForImageClassification.from_pretrained(
            str(checkpoint),
            num_labels=len(args.trait),
        )

        model.to(device)
        model.eval()

        dataset = LabeledDataset(
            trait_csv=args.trait_csv,
            image_dir=args.image_dir,
            image_size=args.image_size,
            split="test",
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )

        new = []
        with torch.no_grad():
            for sheets in loader:
                images = sheets["pixel_values"].to(device)
                preds = model(images)
                preds = torch.sigmoid(preds.logits)
                preds = preds.detach().cpu()
                for pred, true, name in zip(
                    preds, sheets["labels"], sheets["name"], strict=False
                ):
                    rec = {"name": name, "pretrained": checkpoint}

                    pred = pred.tolist()
                    true = true.tolist()
                    for a, p, t in zip(true, pred, util.TRAITS, strict=False):
                        rec[f"{t}_pred"] = p
                        rec[f"{t}_true"] = a
                    new.append(rec)

        print(checkpoint)
        total = len(new)
        print(f"total = {total}")
        for trait in util.TRAITS:
            correct = sum(1 for r in new if r[f"{trait}_pred"] == r[f"{trait}_true"])
            accuracy = round(correct / total, 3)
            print(f"{trait:<8} correct = {correct}   accuracy = {accuracy}")
        print()

        df = pd.DataFrame(new)

        if args.output_csv.exists():
            df_old = pd.read_csv(args.output_csv)
            df = pd.concat((df_old, df))
        df.to_csv(args.output_csv, index=False)


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
        required=True,
        metavar="PATH",
        help="""Output inference results to this CSV.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--pretrained-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing the training checkpoints.""",
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
