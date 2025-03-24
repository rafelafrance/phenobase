#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from copy import deepcopy
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

    base_recs = {
        d["name"]: d
        for d in dataset_util.get_records("test", args.dataset_csv, args.traits)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    checkpoints = [p for p in args.model_dir.glob("checkpoint-*") if p.is_dir()]
    for checkpoint in sorted(checkpoints):
        print(checkpoint)

        model = AutoModelForImageClassification.from_pretrained(
            str(checkpoint),
            problem_type=const.SINGLE_LABEL,
            num_labels=len(const.LABELS),
        )

        model.to(device)
        model.eval()

        dataset = dataset_util.get_dataset(
            "test", args.dataset_csv, args.image_dir, args.traits
        )

        TEST_XFORMS = image_util.build_transforms(args.image_size, augment=False)
        dataset.set_transform(TEST_XFORMS)

        loader = DataLoader(dataset, batch_size=1, pin_memory=True)
        total = len(loader)

        correct = 0
        records = []

        with torch.no_grad():
            for sheet in tqdm(loader):
                image = sheet["image"].to(device)
                result = model(image)

                # Append result record
                rec = deepcopy(base_recs[sheet["id"][0]])
                rec |= {"checkpoint": checkpoint}

                trait = args.traits[0]

                # Note: Softmax for 2 classes is symmetric, so I can use the
                # positive class (scores[1]) for the predicted value. I.e.
                # scores[1] IS always the real score in this case, but only
                # when there are exactly two classes and only when they are
                # organized as const.labels = [WITHOUT, WITH].
                scores = softmax(result.logits).detach().cpu().tolist()[0]
                rec |= {f"{trait}_score": scores[1]}

                true = torch.argmax(torch.tensor(sheet["label"])).item()
                pred = torch.argmax(result.logits).item()
                correct += int(pred == true)

                records.append(rec)

        print(f"Accuracy {correct}/{total} = {(correct / total):0.3f}\n")

        if args.output_csv:
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
        "--traits",
        choices=const.TRAITS,
        action="append",
        help="""Train to classify this trait. Repeat this argument to train
            multiple trait labels.""",
    )

    args = arg_parser.parse_args()

    args.traits = args.traits if args.traits else const.TRAITS

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
