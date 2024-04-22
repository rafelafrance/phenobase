#!/usr/bin/env python3
import argparse
import csv
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from phenobase.history.pylib import log, util
from phenobase.history.pylib.datasets.labeled_dataset import LabeledDataset


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

    device = torch.device("cuda" if torch.has_cuda else "cpu")

    model = timm.create_model(
        args.base_model, pretrained=True, num_classes=len(util.TARGETS)
    )
    model.to(device)

    with args.csv_path.open() as f:
        reader = csv.DictReader(f)
        csv_data = list(reader)

    train_loader = get_train_loader(
        csv_data,
        args.image_dir,
        args.image_size,
        args.batch_size,
        args.workers,
    )
    val_loader = get_val_loader(
        csv_data,
        args.image_dir,
        args.image_size,
        args.batch_size,
        args.workers,
    )

    optimizer = get_optimizer(model, args.learning_rate)
    scheduler = get_scheduler(optimizer)
    loss_fn = get_loss_fn(train_loader.dataset, device)

    stats = Stats(
        best_acc=model.state.get("accuracy", 0.0),
        best_loss=model.state.get("best_loss", float("Inf")),
    )

    end_epoch, start_epoch = get_epoch_range(args, model)

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
        log_stats(stats, epoch)

    log.finished()


def one_epoch(model, device, loader, loss_fn, optimizer=None):
    running_loss = 0.0
    running_acc = 0.0

    for images, orders, targets, _ in loader:
        images = images.to(device)
        orders = orders.to(device)
        targets = targets.to(device)

        preds = model(images, orders)
        # preds = preds[:, use_col].to(device)  # for hydra

        loss = loss_fn(preds, targets)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc += util.accuracy(preds, targets)

    return running_acc / len(loader), running_loss / len(loader)


def get_loss_fn(dataset, device):
    pos_weight = LabeledDataset.pos_weight(dataset)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn


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


def get_train_loader(csv_data, image_dir, image_size, batch_size, workers):
    logging.info("Loading training data.")
    data = LabeledDataset(csv_data, image_dir, "train", image_size, augment=True)
    return DataLoader(
        data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )


def get_val_loader(csv_data, image_dir, image_size, batch_size, workers):
    logging.info("Loading validation data.")
    dataset = LabeledDataset(csv_data, image_dir, "val", image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )


def get_epoch_range(args, model):
    start_epoch = model.state.get("epoch", 0) + 1
    end_epoch = start_epoch + args.epochs
    return end_epoch, start_epoch


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


def log_stats(stats, epoch):
    msg = (
        f"{epoch:4}: "
        f"Train: loss {stats.train_loss:0.6f} acc {stats.train_acc:0.6f} "
        f"Valid: loss {stats.val_loss:0.6f} acc {stats.val_acc:0.6f}"
        f"{' ++' if stats.is_best else ''}"
    )
    logging.info(msg)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Train a model to classify herbarium traits. The input uses images of
            herbarium sheets and a CSV file. The CSV file must contain:
                1. A column with the name of the image file being classified. The
                   directory portion of the image path is given as an argument.
                2. A column for each class being identified, for instance: flowering,
                   fruiting, and leaf-out. The cell values are: "1" = present,
                   "0" = absent, "U" & "N" for uncertain & N/A; the lat 2 get skipped.
                3. A column indicating the data set for the image (training, testing, or
               validation).
            """
        ),
    )

    arg_parser.add_argument(
        "--csv-path",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Path to the CSV file containing the image data described above.""",
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Images of herbarium sheets are in this directory.""",
    )

    arg_parser.add_argument(
        "--image-size",
        type=int,
        metavar="INT",
        default=512,
        help="""Image size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--base-model",
        required=True,
        help="""Use this pretrained model as the base model.""",
    )

    arg_parser.add_argument(
        "--save-model",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Save best models to this path.""",
    )

    arg_parser.add_argument(
        "--load-model",
        type=Path,
        metavar="PATH",
        help="""Continue training with weights from this model.""",
    )

    arg_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        metavar="FLOAT",
        default=0.001,
        help="""Initial learning rate. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        metavar="INT",
        default=16,
        help="""Input batch size. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--workers",
        type=int,
        metavar="INT",
        default=4,
        help="""Number of workers for loading data. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--epochs",
        type=int,
        metavar="INT",
        default=100,
        help="""How many epochs to train. (default: %(default)s)""",
    )

    arg_parser.add_argument(
        "--limit",
        type=int,
        metavar="INT",
        help="""Limit the input to this many records.""",
    )

    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    main()
