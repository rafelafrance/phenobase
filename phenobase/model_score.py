#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from pathlib import Path

import torch
from pylib import const, dataset_util, image_util
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageClassification

TEST_XFORMS: Callable | None = None


def main(args):
    global TEST_XFORMS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_dir in args.model_dir:
        checkpoints = [p for p in model_dir.glob("checkpoint-*") if p.is_dir()]
        for checkpoint in sorted(checkpoints):
            print(checkpoint)

            model = AutoModelForImageClassification.from_pretrained(
                str(checkpoint),
                problem_type=args.problem_type,
                num_labels=dataset_util.get_num_labels(args.problem_type, args.trait),
            )

            model.to(device)
            model.eval()

            dataset = dataset_util.get_dataset(
                "test", args.dataset_csv, args.image_dir, args.trait, args.problem_type
            )

            TEST_XFORMS = image_util.build_transforms(args.image_size, augment=False)
            dataset.set_transform(TEST_XFORMS)

            loader = DataLoader(dataset, batch_size=1, pin_memory=True)
            total = len(loader)

            correct = 0
            with torch.no_grad():
                for sheet in tqdm(loader):
                    image = sheet["image"].to(device)
                    result = model(image)
                    # single_label
                    pred = torch.argmax(result.logits).item()
                    true = torch.argmax(torch.tensor(sheet["label"])).item()
                    correct += int(pred == true)
                    # regression
                    # multi_label
            print(f"{correct} / {total} = {correct/total}")

    #         with torch.no_grad():
    #             for sheets in loader:
    #                 images = sheets["pixel_values"].to(device)
    #                 preds = model(images)
    #                 preds = torch.sigmoid(preds.logits)
    #                 preds = preds.detach().cpu()
    #                 for pred, true, name in zip(
    #                     preds, sheets["labels"], sheets["name"], strict=False
    #                 ):
    #                     true = true.tolist()
    #                     pred = pred.tolist()
    #
    #                     for t, p, trait in zip(true, pred, args.traits, strict=False):
    #                         rows.append(
    #                             {
    #                                 "name": name,
    #                                 "checkpoint": checkpoint,
    #                                 "trait": trait,
    #                                 "y_true": float(t),
    #                                 "y_pred": p,
    #                             }
    #                         )
    #
    # if args.output_csv:
    #     df = pd.DataFrame(rows)
    #     if args.output_csv.exists():
    #         df_old = pd.read_csv(args.output_csv)
    #         df = pd.concat((df_old, df))
    #     df.to_csv(args.output_csv, index=False)


def test_transforms(examples):
    pixel_values = [TEST_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Test a model trained to classify phenology traits."""
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
        "--output-csv",
        type=Path,
        metavar="PATH",
        help="""Output inference results to this CSV.""",
    )

    arg_parser.add_argument(
        "--model-dir",
        type=Path,
        action="append",
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
        "--trait",
        choices=const.TRAITS,
        action="append",
        help="""Train to classify this trait. Repeat this argument to train
            multiple trait labels.""",
    )

    arg_parser.add_argument(
        "--problem-type",
        choices=list(const.PROBLEM_TYPES.keys()),
        default="regression",
        help="""This chooses the appropriate scoring function for the type of problem.
            (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    args.trait = args.trait if args.trait else const.TRAITS

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
