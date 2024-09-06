#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import pandas as pd
import torch
from pylib import util
from pylib.binary_metrics import Metrics
from pylib.labeled_dataset import LabeledDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = [p for p in args.model_dir.glob("checkpoint-*") if p.is_dir()]
    for checkpoint in sorted(checkpoints):
        model = AutoModelForImageClassification.from_pretrained(
            str(checkpoint),
            num_labels=len(util.TRAITS),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
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

        new_rows = []
        y_true = []
        y_pred = []

        with torch.no_grad():
            for sheets in loader:
                images = sheets["pixel_values"].to(device)
                preds = model(images)
                preds = torch.sigmoid(preds.logits)
                preds = preds.detach().cpu()
                for pred, true, name in zip(
                    preds, sheets["labels"], sheets["name"], strict=False
                ):
                    new_row = {"name": name, "pretrained": checkpoint}

                    true = true.tolist()
                    pred = pred.tolist()

                    y_true.append(true)
                    y_pred.append(pred)

                    for t, p, trait in zip(true, pred, util.TRAITS, strict=False):
                        pred_key = f"{trait}_pred"
                        true_key = f"{trait}_true"
                        new_row[true_key] = float(t)
                        new_row[pred_key] = p

                    new_rows.append(new_row)

        print(checkpoint, "\n")
        metrics = Metrics()
        for i, trait in enumerate(util.TRAITS):
            for true, pred in zip(y_true, y_pred, strict=True):
                metrics.add(true[i], pred[i])
            metrics.count()
            print(trait)
            metrics.display()
            print(f"accuracy = {metrics.accuracy:0.3f}")
            print(f"f1       = {metrics.f1:0.3f}")
            print(f"ppv      = {metrics.ppv:0.3f}")
            print(f"npv      = {metrics.npv:0.3f}")
            print()

        if args.output_csv:
            df = pd.DataFrame(new_rows)
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
        metavar="PATH",
        help="""Output inference results to this CSV.""",
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
