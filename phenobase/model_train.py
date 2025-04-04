#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from pathlib import Path

import evaluate
import numpy as np
import torch
import transformers
from pylib import const, dataset_util, image_util
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

TRAIN_XFORMS: Callable | None = None
VALID_XFORMS: Callable | None = None

METRICS = evaluate.combine(["f1", "precision", "recall", "accuracy"])


def main(args):
    global TRAIN_XFORMS, VALID_XFORMS

    transformers.set_seed(args.seed)

    model = AutoModelForImageClassification.from_pretrained(
        args.finetune,
        problem_type=args.problem_type,
        num_labels=1 if args.problem_type == const.REGRESSION else 2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = dataset_util.get_dataset(
        "train",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        args.problem_type,
        use_unknowns=args.use_unknowns,
        limit=args.limit,
    )
    valid_dataset = dataset_util.get_dataset(
        "val",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        args.problem_type,
        use_unknowns=args.use_unknowns,
        limit=args.limit,
    )

    TRAIN_XFORMS = image_util.build_transforms(args.image_size, augment=True)
    VALID_XFORMS = image_util.build_transforms(args.image_size, augment=False)

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
        greater_is_better=args.best_metric != "loss",
        weight_decay=0.01,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        save_total_limit=5,
        push_to_hub=False,
    )

    if args.problem_type == const.REGRESSION:
        compute_metrics = compute_metrics_regression
    else:
        compute_metrics = compute_metrics_single_label

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def compute_metrics_regression(eval_pred):
    logits, trues = eval_pred
    preds = torch.sigmoid(torch.tensor(logits))
    preds = torch.round(preds).flatten()
    trues = torch.tensor(trues).flatten()
    return METRICS.compute(predictions=preds, references=trues)


def compute_metrics_single_label(eval_pred):
    logits, trues = eval_pred
    preds = np.argmax(logits, axis=-1)
    return METRICS.compute(predictions=preds, references=trues)


def train_transforms(examples):
    pixel_values = [TRAIN_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def valid_transforms(examples):
    pixel_values = [VALID_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


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
        required=True,
        metavar="PATH",
        help="""Save training results here.""",
    )

    arg_parser.add_argument(
        "--finetune",
        type=Path,
        required=True,
        default="google/vit-base-patch16-224",
        metavar="PATH",
        help="""Finetune this model.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=const.TRAITS,
        required=True,
        help="""Train to classify this trait.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        metavar="INT",
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--problem-type",
        choices=list(const.PROBLEM_TYPES),
        default=const.REGRESSION,
        help="""This chooses the appropriate scoring function for the problem_type.""",
    )

    arg_parser.add_argument(
        "--use-unknowns",
        action="store_true",
        help="""Use samples with traits identified as unknown by experts, and set their
            expected value to 0.5. Only valid for regression problem types.""",
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
        "--best-metric",
        choices=["f1", "precision", "accuracy", "loss"],
        default="f1",
        help="""Model evaluation strategy.""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        default=8174997,
        metavar="INT",
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit dataset size for testing.""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
