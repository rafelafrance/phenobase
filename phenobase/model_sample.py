#!/usr/bin/env python3

import argparse
import random
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

    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    model = AutoModelForImageClassification.from_pretrained(
        str(args.checkpoint),
        problem_type=const.SINGLE_LABEL,
        num_labels=len(const.LABELS),
    )

    present_unequiv = []
    absent_unequiv = []
    present_equiv = []
    absent_equiv = []

    model.to(device)
    model.eval()

    dataset = dataset_util.get_inference_records(args.db, args.limit, args.offset)
    dataset = dataset_util.filter_bad_inference_images(dataset)
    dataset = dataset_util.filter_bad_inference_families(dataset, args.bad_families)
    dataset = dataset_util.get_inference_dataset(dataset, args.image_dir)

    TEST_XFORMS = image_util.build_transforms(args.image_size, augment=False)
    dataset.set_transform(TEST_XFORMS)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    with torch.no_grad():
        for sheet in tqdm(loader):
            image = sheet["image"].to(device)
            result = model(image)

            # Note: Softmax for 2 classes is symmetric, so I can use the
            # positive class (scores[1]) for the predicted value. I.e.
            # scores[1] is always the real score in this case, but only
            # when there are exactly two classes and only when they are
            # organized as const.labels = [WITHOUT, WITH].
            scores = softmax(result.logits).detach().cpu().tolist()[0]
            score = scores[1]

            rec = {
                "gbifid": sheet["id"][0],
                "family": sheet["family"][0],
                "path": sheet["path"][0],
                "score": score,
                args.trait: int(round(score)),
            }

            if score <= args.thresh_low:
                absent_unequiv.append(rec)
            elif score >= args.thresh_high:
                present_unequiv.append(rec)
            elif score < 0.5:
                absent_equiv.append(rec)
            elif score > 0.5:
                present_equiv.append(rec)

    write(present_unequiv, "present_unequivocal", args)
    write(absent_unequiv, "absent_unequivocal", args)
    write(present_equiv, "present_equivocal", args)
    write(absent_equiv, "absent_equivocal", args)

    log.finished()


def write(dataset, name, args):
    dataset = random.sample(dataset, k=args.sample)
    df = pd.DataFrame(dataset)
    df.to_csv(args.output_dir / f"{args.trait}_{name}.csv")
    dir_ = args.output_dir / f"{args.trait}_{name}"
    dir_.mkdir(parents=True, exist_ok=True)
    for rec in dataset:
        src = rec["path"]
        dst = dir_ / src.name
        copyfile(src, dst)


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
        help="""A database with information about herbarium sheets.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="""A path to the root directory where the images are.""",
    )

    arg_parser.add_argument(
        "--bad-families",
        type=Path,
        help="""Make sure records in these families are skipped.""",
    )

    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="""Output inference results to this directory.""",
    )

    arg_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="""Directory containing the best checkpoint.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
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
        "--sample",
        type=int,
        default=250,
        help="""How many records to sample for each category. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
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
        "--seed",
        type=int,
        metavar="INT",
        default=2114987,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
