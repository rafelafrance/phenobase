#!/usr/bin/env python3

import argparse
import csv
import textwrap
from collections.abc import Callable
from pathlib import Path

import evaluate
import numpy as np
import torch
import transformers
from pylib import const
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

from datasets import Dataset, Image, Split
from phenobase.pylib.labeled_dataset import LabeledDataset

TRAIN_XFORMS: Callable | None = None
VALID_XFORMS: Callable | None = None

METRICS = evaluate.combine(["f1", "precision", "recall", "accuracy"])


def main():
    global TRAIN_XFORMS, VALID_XFORMS

    args = parse_args()

    transformers.set_seed(args.seed)

    problem_type = "regression"  # "single_label_classification"
    if len(args.traits) > 1:
        problem_type = "multi_label_classification"

    model = AutoModelForImageClassification.from_pretrained(
        args.finetune,
        problem_type=problem_type,
        num_labels=len(args.trait),
        id2label={str(i): v for i, v in enumerate(args.trait)},
        label2id={v: str(i) for i, v in enumerate(args.trait)},
        ignore_mismatched_sizes=True,
    )

    train_dataset = get_dataset("train", args.dataset_csv, args.image_dir)
    valid_dataset = get_dataset("valid", args.dataset_csv, args.image_dir)

    TRAIN_XFORMS = LabeledDataset.build_transforms(args.image_size, augment=True)
    VALID_XFORMS = LabeledDataset.build_transforms(args.image_size, augment=False)

    train_dataset.set_transform(train_transforms)
    valid_dataset.set_transform(valid_transforms)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
        weight_decay=0.01,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        save_total_limit=5,
        push_to_hub=False,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def train_transforms(examples):
    pixel_values = [TRAIN_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def valid_transforms(examples):
    pixel_values = [VALID_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def get_dataset(split: str, dataset_csv: Path, image_dir: Path) -> Dataset:
    with dataset_csv.open() as inp:
        reader = csv.DictReader(inp)
        recs = list(reader)

    recs = [r for r in recs if r["split"] == split]  # [:16]  # !!!!!!!!!!!!!!!!!!!!!!!

    split = Split.TRAIN if split == "train" else Split.VALIDATION

    images = [str(image_dir / r["name"]) for r in recs]
    labels = [int(r["label"]) for r in recs]
    dataset = Dataset.from_dict(
        {"image": images, "label": labels}, split=split
    ).cast_column("image", Image())

    return dataset


def compute_metrics(eval_pred):
    logits, trues = eval_pred
    preds = np.argmax(logits, axis=-1)
    return METRICS.compute(predictions=preds, references=trues)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Train a model to classify phenology traits."""),
    )

    arg_parser.add_argument(
        "--dataset-csv",
        type=Path,
        required=True,
        metavar="PATH",
        help="""A path to dataset directory.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""images are in this directory.""",
    )

    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="PATH",
        help="""Save training results here.""",
    )

    arg_parser.add_argument(
        "--finetune",
        type=Path,
        default="google/vit-base-patch16-224",
        metavar="PATH",
        help="""Finetune this model.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=const.TRAITS,
        action="append",
        help=f"""Train to classify this trait. Repeat this argument for multi-label
            classification not single label multi-class training.
            (default: all traits {', '.join(const.TRAITS)})""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=5e-5,
        metavar="FLOAT",
        help="""Initial learning rate. (default: %(default)s)""",
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

    arg_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="INT",
        help="""How many epochs to train. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--use-weights",
        action="store_true",
        help="""Use positive weights when training.""",
    )

    arg_parser.add_argument(
        "--best-metric",
        choices=["precision", "f1", "accuracy", "loss"],
        default="precision",
        help="""Model evaluation strategy.""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=8174997,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )
    args = arg_parser.parse_args()

    args.traits = args.trait if args.trait else const.TRAITS

    return args


if __name__ == "__main__":
    main()
