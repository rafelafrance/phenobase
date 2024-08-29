#!/usr/bin/env python3

import argparse
import logging
import textwrap
from pathlib import Path

import evaluate
import torch
from pylib import util
from pylib.labeled_dataset import LabeledDataset
from torch import FloatTensor, nn
from transformers import Trainer, TrainingArguments, ViTForImageClassification

accuracy = evaluate.load("accuracy")


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


# def compute_accuracy(eval_pred):
#     logits, trues = eval_pred
#     preds = torch.sigmoid(torch.tensor(logits))
#     preds = torch.round(preds)
#     return accuracy.compute(predictions=preds, references=trues)


def main():
    args = parse_args()

    model = ViTForImageClassification.from_pretrained(
        args.finetune,
        num_labels=len(args.trait),
        ignore_mismatched_sizes=True,
    )

    train_dataset = LabeledDataset(
        augment=True,
        image_dir=args.image_dir,
        split="train",
        trait_csv=args.trait_csv,
        traits=args.trait,
        image_size=args.image_size,
    )

    eval_dataset = LabeledDataset(
        augment=False,
        image_dir=args.image_dir,
        split="eval",
        trait_csv=args.trait_csv,
        traits=args.trait,
        image_size=args.image_size,
    )

    training_args = TrainingArguments(
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = VitTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pos_weight=train_dataset.pos_weight(),
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
