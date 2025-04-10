#!/usr/bin/env python3

import argparse
import csv
import logging
import sqlite3
import textwrap
from pathlib import Path
from shutil import copyfile

import pandas as pd
import torch
from pylib import log, util
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from datasets import Dataset, Image


def main(args):
    log.started()

    softmax = torch.nn.Softmax(dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForImageClassification.from_pretrained(
        str(args.checkpoint),
        num_labels=1 if args.regression else 2,
    )

    model.to(device)
    model.eval()

    logging.info(f"Getting dataset limit {args.limit} offset {args.offset}")

    records = get_inference_records(args.db, args.limit, args.offset)
    total = len(records)
    records = filter_bad_images(records)
    good = len(records)
    records = filter_bad_families(records, args.bad_families)
    families = len(records)

    dataset = get_inference_dataset(records, args.image_dir, debug=args.debug)

    logging.info(f"Total records {total}, good images {good}, good families {families}")

    if args.debug:
        logging.info(f"Found {len(dataset)} files for debugging.")

    infer_xforms = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
        ]
    )

    dataset.set_transform(infer_xforms)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    output = []

    logging.info("Starting inference")
    logging.info(f"Low threshold: {args.thresh_low}")
    logging.info(f"High threshold: {args.thresh_high}")

    with torch.no_grad():
        for sheet in tqdm(loader):
            image = sheet["image"].to(device)
            result = model(image)

            if args.regression:
                score = torch.sigmoid(torch.tensor(result.logits))
            else:
                # I'm filtering on the score.
                # I can use softmax and take the positive class for the predicted value
                scores = softmax(result.logits)
                score = scores[0, 1]

            if score <= args.thresh_low or score >= args.thresh_high:
                rec = {
                    "gbifid": sheet["id"][0],
                    "score": score,
                    "family": sheet["family"][0],
                    "path": sheet["path"][0],
                    f"{args.trait}_score": score.item(),
                    args.trait: torch.round(score).item(),
                }
                output.append(rec)

                if args.save_dir:
                    src = Path(sheet["path"][0])
                    dst = args.save_dir / src.name
                    if not dst.exists():
                        copyfile(src, dst)

    df = pd.DataFrame(output)
    df.to_csv(args.output_csv, index=False)

    logging.info(f"Remaining {len(output)}")

    log.finished()


def get_inference_records(db, limit, offset):
    with sqlite3.connect(db) as cxn:
        cxn.row_factory = sqlite3.Row
        sql = """select gbifid, tiebreaker, state, family
            from multimedia join occurrence using (gbifid)
            limit ? offset ?"""
        rows = [dict(r) for r in cxn.execute(sql, (limit, offset))]
    return rows


def filter_bad_images(rows):
    rows = [
        r
        for r in rows
        if r["state"].startswith("images") and not r["state"].endswith("error")
    ]
    return rows


def filter_bad_families(rows, bad_family_csv):
    bad_families = []
    if bad_family_csv:
        with bad_family_csv.open() as bad:
            reader = csv.DictReader(bad)
            bad_families = [r["family"].lower() for r in reader]
    rows = [r for r in rows if r["family"].lower() not in bad_families]
    return rows


def get_inference_dataset(rows, image_dir, *, debug: bool = False):
    images = []
    ids = []
    families = []
    for row in rows:
        parts = row["state"].split()

        name = f"{row['gbifID']}_{row['tiebreaker']}"
        name += f"_{parts[-1]}" if len(parts) > 1 else ""

        if debug:
            path = image_dir / (name + ".jpg")
            if not path.exists():
                continue
        else:
            path = image_dir / parts[0] / (name + ".jpg")

        ids.append(row["gbifID"])
        families.append(row["family"])
        images.append(str(path))

    dataset = Dataset.from_dict(
        {"image": images, "id": ids, "path": images, "family": families}
    ).cast_column("image", Image())

    return dataset


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
        choices=util.TRAITS,
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

    arg_parser.add_argument(
        "--regression",
        action="store_true",
        help="""Handle regression scoring.""",
    )

    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="""Testing this module locally.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
