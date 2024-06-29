#!/usr/bin/env python3

import argparse
import logging
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import torch
from pylib import log, util
from pylib.datasets.labeled_traits import LabeledTraits
from pylib.models.simple_vit import SimpleVit
from torch import FloatTensor, nn, optim
from torch.utils.data import DataLoader
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


def new_main():
    args = parse_args()

    model = ViTForImageClassification.from_pretrained(
        str(args.pretrained_dir), num_labels=len(args.trait)
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
    )

    trainer = VitTrainer(
        pos_weight=train_dataset.pos_weight(),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


@dataclass
class Stats:
    is_best: bool = False
    best_loss: float = float("Inf")
    train_loss: float = float("Inf")
    val_loss: float = float("Inf")


def main():
    log.started()
    args = parse_args()

    model = SimpleVit(
        traits=args.trait,
        load_model=args.load_model,
        pretrained_dir=args.pretrained_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = get_loader(
        split="train",
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        traits=args.trait,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    val_loader = get_loader(
        split="eval",
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        traits=args.trait,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    optimizer = get_optimizer(model, args.learning_rate)
    loss_fn = get_loss_fn(train_loader.dataset, device)

    stats = Stats(best_loss=model.state.get("best_loss", float("Inf")))

    start_epoch, end_epoch = get_epoch_range(args, model)

    logging.info("Training started.")
    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_loss = one_epoch(model, device, train_loader, loss_fn, optimizer)

        model.eval()
        stats.val_loss = one_epoch(model, device, val_loader, loss_fn)

        save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(stats, epoch)

    log.finished()


def one_epoch(model, device, loader, loss_fn, optimizer=None):
    running_loss = 0.0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)

        loss = loss_fn(preds.logits, targets)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def get_loss_fn(dataset, device):
    pos_weight = dataset.pos_weight()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn


def get_epoch_range(args, model):
    start_epoch = model.state.get("epoch", 0) + 1
    end_epoch = start_epoch + args.epochs
    return start_epoch, end_epoch


def get_loader(
    split: util.SPLIT,
    trait_csv: Path,
    image_dir: Path,
    traits: list[util.TRAIT],
    batch_size: int,
    workers: int,
):
    msg = f"Loading {split} data."
    logging.info(msg)
    dataset = LabeledTraits(
        trait_csv=trait_csv,
        image_dir=image_dir,
        traits=traits,
        split=split,
        augment=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=split == "train",
        pin_memory=True,
    )


def get_optimizer(model, lr):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if model.state.get("optimizer_state"):
        logging.info("Loading the optimizer.")
        optimizer.load_state_dict(model.state["optimizer_state"])
    return optimizer


def save_checkpoint(model, optimizer, save_model, stats, epoch):
    stats.is_best = False
    if stats.val_loss <= stats.best_loss:
        stats.is_best = True
        stats.best_loss = stats.val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": stats.best_loss,
            },
            save_model,
        )


def log_stats(stats, epoch):
    # Must happen after save_checkpoint
    msg = (
        f"{epoch:4}: "
        f"Train: loss {stats.train_loss:0.6f} "
        f"Valid: loss {stats.val_loss:0.6f}"
        f"{' ++' if stats.is_best else ''}"
    )
    logging.info(msg)


def parse_args():
    description = """Train a model to classify phenology traits."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
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

    # arg_parser.add_argument(
    #     "--save-model",
    #     type=Path,
    #     metavar="PATH",
    #     required=True,
    #     help="""Save best models to this path.""",
    # )

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
        "--load-model",
        type=Path,
        metavar="PATH",
        help="""Continue training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--pretrained-dir",
        type=Path,
        metavar="PATH",
        help="""Use this pretrained model to kickoff training.""",
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

    if args.pretrained_dir and args.load_model:
        sys.exit("Only one of pretrained or load model can be used at a time.")

    return args


if __name__ == "__main__":
    # main()
    new_main()
