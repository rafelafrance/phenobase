#!/usr/bin/env python3

import argparse
import textwrap
from collections.abc import Callable
from pathlib import Path

import torch
import transformers
from pylib import const, dataset_util, image_util
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

TRAIN_XFORMS: Callable | None = None
VALID_XFORMS: Callable | None = None


def main(args):
    global TRAIN_XFORMS, VALID_XFORMS

    transformers.set_seed(args.seed)

    model = AutoModelForImageClassification.from_pretrained(
        args.finetune,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    train_dataset = dataset_util.get_dataset(
        "train",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        use_unknowns=args.use_unknowns,
    )
    valid_dataset = dataset_util.get_dataset(
        "val",
        args.dataset_csv,
        args.image_dir,
        args.trait,
        use_unknowns=args.use_unknowns,
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
        weight_decay=0.01,
        overwrite_output_dir=True,
        logging_strategy="epoch",
        save_total_limit=3,
        push_to_hub=False,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
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
        "--use-unknowns",
        action="store_true",
        help="""Use samples with traits identified as unknown by experts, and set their
            expected value to 0.5.""",
    )

    arg_parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=8174997,
        help="""Seed used for random number generator. (default: %(default)s)""",
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS)
