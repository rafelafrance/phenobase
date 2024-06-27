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
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Stats:
    is_best: bool = False
    best_acc: float = 0.0
    best_loss: float = float("Inf")
    train_acc: float = 0.0
    train_loss: float = float("Inf")
    val_acc: float = 0.0
    val_loss: float = float("Inf")


def main():
    log.started()
    args = parse_args()

    model = SimpleVit(
        traits=args.trait,
        pretrained_dir=args.pretrained_dir,
        load_model=args.load_model,
        model_config=args.model_config,
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
        split="val",
        trait_csv=args.trait_csv,
        image_dir=args.image_dir,
        traits=args.trait,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    optimizer = get_optimizer(model, args.learning_rate)
    scheduler = get_scheduler(optimizer)
    loss_fn = get_loss_fn(train_loader.dataset, device)

    writer = SummaryWriter(args.log_dir)

    stats = Stats(
        best_acc=model.state.get("accuracy", 0.0),
        best_loss=model.state.get("best_loss", float("Inf")),
    )

    start_epoch, end_epoch = get_epoch_range(args, model)

    logging.info("Training started.")
    for epoch in range(start_epoch, end_epoch):
        model.train()
        stats.train_acc, stats.train_loss = one_epoch(
            model, device, train_loader, loss_fn, optimizer
        )

        if scheduler:
            scheduler.step()

        model.eval()
        stats.val_acc, stats.val_loss = one_epoch(model, device, val_loader, loss_fn)

        save_checkpoint(model, optimizer, args.save_model, stats, epoch)
        log_stats(writer, stats, epoch)

    writer.close()
    log.finished()


def one_epoch(model, device, loader, loss_fn, optimizer=None):
    running_loss = 0.0
    running_acc = 0.0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)

        loss = loss_fn(preds, targets)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += util.accuracy(preds, targets)

    return running_acc / len(loader), running_loss / len(loader)


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


def get_scheduler(optimizer):
    return optim.lr_scheduler.CyclicLR(
        optimizer, 0.001, 0.01, step_size_up=10, cycle_momentum=False
    )


def get_loss_fn(dataset, device):
    pos_weight = dataset.pos_weight()
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn


def save_checkpoint(model, optimizer, save_model, stats, epoch):
    stats.is_best = False
    if (stats.val_acc, -stats.val_loss) >= (stats.best_acc, -stats.best_loss):
        stats.is_best = True
        stats.best_acc = stats.val_acc
        stats.best_loss = stats.val_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": stats.best_loss,
                "accuracy": stats.best_acc,
            },
            save_model,
        )


def log_stats(writer, stats, epoch):
    msg = (
        f"{epoch:4}: "
        f"Train: loss {stats.train_loss:0.6f} acc {stats.train_acc:0.6f} "
        f"Valid: loss {stats.val_loss:0.6f} acc {stats.val_acc:0.6f}"
        f"{' ++' if stats.is_best else ''}"
    )
    logging.info(msg)
    writer.add_scalars(
        "Training vs. Validation",
        {
            "Training loss": stats.train_loss,
            "Training accuracy": stats.train_acc,
            "Validation loss": stats.val_loss,
            "Validation accuracy": stats.val_acc,
        },
        epoch,
    )
    writer.flush()


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

    arg_parser.add_argument(
        "--save-model",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save best models to this path.""",
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
        "--model-config",
        type=Path,
        metavar="PATH",
        help="""The model configuration JSON file for the model.""",
    )

    arg_parser.add_argument(
        "--pretrained-dir",
        type=Path,
        metavar="PATH",
        help="""Use this pretrained model to kickoff training.""",
    )

    arg_parser.add_argument(
        "--log-dir",
        type=Path,
        metavar="DIR",
        help="""Save tensorboard logs to this directory.""",
    )

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=2e-5,
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
    main()
