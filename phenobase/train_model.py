#!/usr/bin/env python3

import argparse
import logging
import textwrap
from pathlib import Path

import torch
from pylib import util
from pylib.datasets.labeled_traits import LabeledTraits
from torch import FloatTensor, nn
from transformers import Trainer, TrainingArguments, ViTForImageClassification


class VitTrainer(Trainer):
    def __init__(self, *args, pos_weight: FloatTensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(self.args.device)

        msg = f"Using multi-label classification with class weights: {pos_weight}"
        logging.info(msg)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False):  # noqa: FBT002
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    if args.pretrained_dir:
        model = ViTForImageClassification.from_pretrained(
            str(args.pretrained_dir),
            num_labels=len(args.trait),
        )
    else:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=len(args.trait),
            ignore_mismatched_sizes=True,
        )

    train_dataset = LabeledTraits(
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        traits=args.trait,
        split="train",
        augment=True,
    )

    eval_dataset = LabeledTraits(
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        traits=args.trait,
        split="eval",
        augment=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=3,
        logging_strategy="epoch",
    )

    trainer = VitTrainer(
        pos_weight=train_dataset.pos_weight(),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


def parse_args():
    arg_parser = argparse.ArgumentParser(
        allow_abbrev=True,
        description=textwrap.dedent("""Train a model to classify phenology traits."""),
    )

    arg_parser.add_argument(
        "--trait-csv",
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
        "--output-dir",
        type=Path,
        metavar="PATH",
        help="""Save training results here.""",
    )

    arg_parser.add_argument(
        "--trait",
        choices=util.TRAITS,
        action="append",
        required=True,
        help="""Train to classify this trait. Repeat this argument to train
            multiple trait labels.""",
    )

    arg_parser.add_argument(
        "--pretrained-dir",
        type=Path,
        metavar="PATH",
        help="""Continue training the model in this directory.""",
    )

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=3e-4,
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

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
