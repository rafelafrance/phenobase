#!/usr/bin/env python3

import argparse
import logging
import textwrap
from collections.abc import Callable
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    model = AutoModelForImageClassification.from_pretrained(
        str(args.checkpoint),
        problem_type=const.SINGLE_LABEL,
        num_labels=len(const.LABELS),
    )

    model.to(device)
    model.eval()

    logging.info("Getting dataset limit {args.limit} offset {args.offset}")

    dataset = dataset_util.get_inference_dataset(
        args.db, args.image_dir, args.bad_families, args.limit, args.offset
    )

    TEST_XFORMS = image_util.build_transforms(args.image_size, augment=False)
    dataset.set_transform(TEST_XFORMS)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    records = []

    logging.info("Starting inference")

    with torch.no_grad():
        for sheet in tqdm(loader):
            image = sheet["image"].to(device)
            result = model(image)

            # Note: Softmax for 2 classes is symmetric, so I can use the
            # positive class (scores[1]) for the predicted value. I.e.
            # scores[1] IS always the real score in this case, but only
            # when there are exactly two classes and only when they are
            # organized as const.labels = [WITHOUT, WITH].
            scores = softmax(result.logits).detach().cpu().tolist()[0]

            rec = {"gbifid": sheet["id"][0], "score": scores[1]}

            records.append(rec)

    df = pd.DataFrame(records)
    if args.output_csv.exists():
        df_old = pd.read_csv(args.output_csv, low_memory=False)
        df = pd.concat((df_old, df))
    df.to_csv(args.output_csv, index=False)

    log.finished()


def test_transforms(examples):
    pixel_values = [TEST_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Infer phenology traits."""),
    )

    arg_parser.add_argument(
        "--db",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A database with information about herbarium sheets.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""A path to the directory where the images are.""",
    )

    arg_parser.add_argument(
        "--bad-families",
        metavar="PATH",
        type=Path,
        help="""Make sure records in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output inference results to this CSV.""",
    )

    arg_parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing the training checkpoint.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=1_000_000,
        metavar="INT",
        help="""Limit to this many images. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="INT",
        help="""Read records after this offset. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
