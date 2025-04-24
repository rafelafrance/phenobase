#!/usr/bin/env python3

import argparse
import json
import logging
import textwrap
from pathlib import Path

import numpy as np
import torch
from pylib import log, util
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoModelForImageClassification


def main(args):
    log.started(args=args)

    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    softmax = torch.nn.Softmax(dim=1)

    logging.info(f"Device {device}")

    checkpoints = sorted([p for p in args.model_dir.glob("checkpoint-*") if p.is_dir()])

    skel = skeleton_start(args.dataset_csv, args.trait, checkpoints)

    for checkpoint in sorted(checkpoints):
        print(checkpoint)

        model = AutoModelForImageClassification.from_pretrained(
            str(checkpoint),
            num_labels=1 if args.problem_type == util.ProblemType.REGRESSION else 2,
        )
        model.to(device)

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
                    skel[file_]["scores"][checkpoint.stem] = pred[0]
                    correct += int(round(pred[0]) == true)

                else:
                    # I want scores here vs classes because I do threshold moving later
                    # in an attempt to improve the final results.
                    # I can softmax the logits and take scores[0, 1] as the score
                    scores = softmax(result.logits).detach().cpu().tolist()[0]
                    skel[file_]["scores"][checkpoint.stem] = scores[1]

                    true = float(sheet["label"])
                    pred = np.argmax(result.logits.detach().cpu(), axis=-1).item()
                    correct += int(pred == true)

        print(f"Accuracy {correct}/{total} = {(correct / total):0.3f}\n")

    skeleton_finish(skel, args)

    log.finished()


def skeleton_start(dataset_csv, trait, checkpoints) -> dict[str, dict]:
    fields = (
        f"scientificname formatted_genus formatted_family file split db {trait}".split()
    )
    recs = util.get_records("test", dataset_csv, trait)
    skel = {
        r["file"].split(".")[0]: {f: r[f] for f in fields}
        | {"scores": {c.stem: 0.0 for c in checkpoints}}
        for r in recs
    }
    for rec in skel.values():
        rec[trait] = float(rec[trait])
    return skel


def skeleton_finish(skel, args):
    out = {
        "score_args": {},
        "train_args": [],
        "records": skel,
    }

    for key, val in sorted(vars(args).items()):
        out["score_args"][key] = str(val)

    if args.training_log:
        with args.training_log.open() as f:
            for ln in f.readlines():
                if ln.find("Argument:") > -1:
                    out["train_args"].append(ln.strip())

    with args.score_json.open("w") as f:
        json.dump(out, f, indent=4)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent(
            """Score a model trained to classify phenology traits."""
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
        "--score-json",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Output scores to this JSON file.""",
    )

    arg_parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Directory containing the training checkpoints.""",
    )

    arg_parser.add_argument(
        "--training-log",
        type=Path,
        metavar="PATH",
        help="""Get the training arguments from this log file.""",
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
