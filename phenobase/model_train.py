#!/usr/bin/env python3

import argparse
import textwrap
from pathlib import Path

import evaluate
import torch
import transformers
from pylib import const
from pylib.labeled_dataset import LabeledDataset
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

metrics = evaluate.combine(["precision", "f1", "recall", "accuracy"])


class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.FloatTensor, **kwargs):
        super().__init__(*args, **kwargs)

        self.pos_weight = pos_weight.to(self.args.device)

        print(f"Using multi-label classification with positive weights: {pos_weight}")

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False):  # noqa: FBT002
        labels = inputs.get("labels")
        outputs = model(**inputs)

        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, trues = eval_pred
    preds = torch.sigmoid(torch.tensor(logits))
    preds = torch.round(preds).flatten()
    trues = torch.tensor(trues).flatten()
    return metrics.compute(predictions=preds, references=trues)


def main():
    args = parse_args()

    transformers.set_seed(args.seed)

    model = AutoModelForImageClassification.from_pretrained(
        args.finetune,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    train_dataset = LabeledDataset(
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        split="train",
        image_size=args.image_size,
        augment=True,
        trait=args.trait,
    )

    eval_dataset = LabeledDataset(
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        split="eval",
        image_size=args.image_size,
        augment=False,
        trait=args.trait,
    )

    training_args = TrainingArguments(
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
        greater_is_better=True,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        push_to_hub=False,
    )

    pos_weight = torch.ones(1)
    if args.use_weights:
        pos_weight = train_dataset.pos_weight()

    trainer = WeightedTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight,
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
        help="""A CSV file with images names and trait labels.""",
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
        "--trait",
        choices=const.TRAITS,
        help="""Train to classify this trait. Repeat this argument to train
            multiple trait labels.""",
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

    return args


if __name__ == "__main__":
    main()
