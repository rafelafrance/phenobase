#!/usr/bin/env python3

import argparse
import logging
import textwrap
from collections.abc import Callable
from pathlib import Path
from shutil import copyfile

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

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    model = AutoModelForImageClassification.from_pretrained(
        str(args.checkpoint),
        problem_type=const.REGRESSION,
        num_labels=1,
    )

    model.to(device)
    model.eval()

    logging.info(f"Getting dataset limit {args.limit} offset {args.offset}")

    dataset = dataset_util.get_inference_records(args.db, args.limit, args.offset)
    total = len(dataset)
    logging.info(f"{total} records retrieved from database")

    dataset = dataset_util.filter_bad_inference_images(dataset)
    images = len(dataset)
    logging.info(f"{images} records have images")

    dataset = dataset_util.filter_bad_families(dataset, args.bad_families)
    families = len(dataset)
    logging.info(f"{families} records remain after filtering 'bad' families")

    dataset = dataset_util.get_inference_dataset(dataset, args.image_dir)

    TEST_XFORMS = image_util.build_transforms(args.image_size, augment=False)
    dataset.set_transform(TEST_XFORMS)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    records = []

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
            score = scores[1]

            if score <= args.thresh_low or score >= args.thresh_high:
                rec = {
                    "gbifid": sheet["id"][0],
                    "score": score,
                    "family": sheet["family"][0],
                    args.trait: int(round(score)),
                }
                records.append(rec)

                if args.save_dir:
                    src = Path(sheet["path"][0])
                    dst = args.save_dir / src.name
                    copyfile(src, dst)

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    logging.info(f"{len(records)} records remain after removing equivocal scores")
    logging.info(f"Low threshold: {args.thresh_low}")
    logging.info(f"High threshold: {args.thresh_high}")

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
        required=True,
        metavar="PATH",
        help="""A database with information about herbarium sheets.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
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

    arg_parser.add_argument(
        "--trait",
        choices=const.TRAITS,
        required=True,
        help="""Infer this trait.""",
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

    arg_parser.add_argument(
        "--save-dir",
        type=Path,
        metavar="PATH",
        help="""Where to put images for further analysis.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
