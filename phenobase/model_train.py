#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from pathlib import Path

import evaluate
import numpy as np
import torch
import transformers
from pylib import util
from torchvision.transforms import v2
from transformers import (
    AutoModelForImageClassification,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

TRAIN_XFORMS: Callable = None
VALID_XFORMS: Callable = None

METRICS = evaluate.combine(["f1", "precision", "recall", "accuracy"])


class BestMetricsCallback(TrainerCallback):
    """
    Save the top n models using the given metric as the only criterion.

    The problem is that if you set metric_for_best_model the trainer only saves the top
    model with that metric, and it then goes back to saving the last n-1 models not
    considering the metric. This may be useful if we do early stopping, but it is not
    helpful in my case, so I patched into the trainer with a callback and save models
    using only the given metric.
    """

    def __init__(self):
        super().__init__()
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

        short = metric.removeprefix("eval_")
        msg = [
            f"(epoch:{b['epoch']:5.1f}, {short}: {b[metric]:6.4f})" for b in self.bests
        ]
        msg = f"\nbest {metric}: " + " ".join(msg)
        print(msg)

        epoch = state.log_history[-1]["epoch"]
        control.should_save = any(epoch == b["epoch"] for b in self.bests)


def main(args):
    global TRAIN_XFORMS, VALID_XFORMS

    transformers.set_seed(args.seed)

    model = AutoModelForImageClassification.from_pretrained(
        args.finetune,
        num_labels=1 if args.regression else 2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = util.get_dataset(
        "train",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        limit=args.limit,
        regression=args.regression,
    )
    valid_dataset = util.get_dataset(
        "val",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        limit=args.limit,
        regression=args.regression,
    )

    TRAIN_XFORMS = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.AutoAugment(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
        ]
    )
    VALID_XFORMS = v2.Compose(
        [
            v2.Resize((args.image_size, args.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=util.IMAGENET_MEAN, std=util.IMAGENET_STD_DEV),
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

    if args.regression:
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

    callback = BestMetricsCallback()
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


def compute_metrics_single_label(eval_pred):
    logits, trues = eval_pred
    preds = np.argmax(logits, axis=-1)
    return METRICS.compute(predictions=preds, references=trues)


def compute_metrics_regression(eval_pred):
    logits, trues = eval_pred
    preds = torch.sigmoid(torch.tensor(logits))
    preds = torch.round(preds)
    trues = torch.tensor(trues)
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
        choices=util.TRAITS,
        required=True,
        help="""Train to classify this trait.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=224,
        help="""Images are this size (pixels). (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--regression",
        action="store_true",
        help="""Use MSE as the loss function.""",
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
