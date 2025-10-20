#!/usr/bin/env python3

import argparse
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pylib import log, util
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageClassification


def main(args: argparse.Namespace) -> None:
    log.started(args=args)

    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    logging.info(f"Device {device}")

    model = AutoModelForImageClassification.from_pretrained(
        str(args.checkpoint),
        num_labels=1 if args.problem_type == util.ProblemType.REGRESSION else 2,
    )
    model.to(device)
    model.eval()

    records = util.get_records("test", args.dataset_csv, trait=args.trait)
    base_rec = {r["image"]: r for r in records}

    dataset = util.get_dataset(
        "test",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        problem_type=args.problem_type,
    )

    test_xforms = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
        ]
    )

    dataset.set_transform(test_xforms)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)
    total = len(loader)
    correct = 0

    with torch.no_grad():
        for sheet in tqdm(loader):
            image = sheet["image"].to(device)
            result = model(image)

            file_ = sheet["id"][0]

            if args.problem_type == util.ProblemType.REGRESSION:
                true = sheet["label"].item()
                pred = torch.sigmoid(result.logits)
                pred = pred.detach().cpu().tolist()[0]
                base_rec[file_]["scores"][args.checkpoint.stem] = pred[0]
                correct += int(round(pred[0]) == true)

            else:
                # I want scores here vs classes because I do threshold moving later
                # in an attempt to improve the final results.
                # I can softmax the logits and take scores[0, 1] as the score
                scores = softmax(result.logits).detach().cpu().tolist()[0]
                base_rec[file_]["scores"][args.checkpoint.stem] = scores[1]

                true = float(sheet["label"])
                pred = np.argmax(result.logits.detach().cpu(), axis=-1).item()
                correct += int(pred == true)

    print(f"Accuracy {correct}/{total} = {(correct / total):0.3f}\n")

    df = pd.DataFrame(base_rec)
    df.to_csv(args.output_csv, index=False)

    log.finished()


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Score a model trained to classify phenology traits on the holdout set."""
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
        "--output-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output scores to this CSV file.""",
    )

    arg_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing the model checkpoint.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--problem-type",
        choices=util.PROBLEM_TYPES,
        default=util.ProblemType.SINGLE_LABEL,
        help="""What kind of a model are we scoring. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
