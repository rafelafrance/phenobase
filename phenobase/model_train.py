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
from torchvision.transforms import v2
from transformers import (
    AutoModelForImageClassification,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from datasets import Dataset, Image, Split

TRAIN_XFORMS: Callable | None = None
VALID_XFORMS: Callable | None = None

METRICS = evaluate.combine(["f1", "precision", "recall", "accuracy"])


class BestMetricsCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        super().__init__()
        self.trainer = trainer
        self.bests = []

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.bests.append(state.log_history[-1])

        sign = -1.0 if args.greater_is_better else 1.0
        metric = args.metric_for_best_model
        self.bests = sorted(self.bests, key=lambda b: sign * b[metric])
        self.bests = self.bests[: args.save_total_limit]

        print(
            [f"(epoch: {b['epoch']:4d}, metric: {b[metric]:6.4f})" for b in self.bests]
        )

        epoch = state.log_history[-1]["epoch"]
        control.should_save = any(epoch == b["epoch"] for b in self.bests)


def main(args):
    global TRAIN_XFORMS, VALID_XFORMS

    transformers.set_seed(args.seed)

    model = AutoModelForImageClassification.from_pretrained(
        args.finetune,
        num_labels=len(const.LABELS),
        id2label=const.ID2LABEL,
        label2id=const.LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = get_dataset("train", args.dataset_csv, args.image_dir, args.limit)
    valid_dataset = get_dataset("valid", args.dataset_csv, args.image_dir, args.limit)

    TRAIN_XFORMS = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.AutoAugment(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=const.IMAGENET_MEAN, std=const.IMAGENET_STD_DEV),
        ]
    )
    VALID_XFORMS = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=const.IMAGENET_MEAN, std=const.IMAGENET_STD_DEV),
        ]
    )

    train_dataset.set_transform(train_transforms)
    valid_dataset.set_transform(valid_transforms)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
        greater_is_better=args.best_metric != "loss",
        save_total_limit=5,
        overwrite_output_dir=True,
        seed=args.seed,
        push_to_hub=False,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    callback = BestMetricsCallback(trainer)
    trainer.add_callback(callback)

    trainer.train()


def train_transforms(examples):
    pixel_values = [TRAIN_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def valid_transforms(examples):
    pixel_values = [VALID_XFORMS(image.convert("RGB")) for image in examples["image"]]
    labels = torch.tensor(examples["label"])
    return {"pixel_values": pixel_values, "labels": labels}


def get_dataset(split: str, dataset_csv: Path, image_dir: Path, limit=0) -> Dataset:
    with dataset_csv.open() as inp:
        reader = csv.DictReader(inp)
        recs = list(reader)

    recs = [r for r in recs if r["split"] == split]
    recs = recs[:limit] if limit else recs

    split = Split.TRAIN if split == "train" else Split.VALIDATION

    images = [str(image_dir / r["name"]) for r in recs]
    labels = [int(r["label"]) for r in recs]
    ids = [r["name"] for r in recs]

    dataset = Dataset.from_dict(
        {"image": images, "label": labels, "id": ids}, split=split
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
