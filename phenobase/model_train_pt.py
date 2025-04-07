#!/usr/bin/env python3

import argparse
import logging
import textwrap
from pathlib import Path

import torch
from pylib import const, labeled_dataset, log, statistics
from torch import optim
from torch.nn import Dropout, Linear, MSELoss, Sequential
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B6_Weights, efficientnet_b6
from tqdm import tqdm


def main(args):
    log.started()

    torch.manual_seed(args.seed)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    model.classifier = Sequential(
        Dropout(p=0.5, inplace=True),
        Linear(in_features=2304, out_features=1, bias=True),
    )

    device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")
    model.to(device)

    train_dataset = labeled_dataset.LabeledDataset(
        args.dataset_csv,
        split="train",
        trait=args.trait,
        image_dir=args.image_dir,
        image_size=args.image_size,
        augment=True,
        limit=args.limit,
    )

    val_dataset = labeled_dataset.LabeledDataset(
        args.dataset_csv,
        split="val",
        trait=args.trait,
        image_dir=args.image_dir,
        image_size=args.image_size,
        augment=False,
        limit=args.limit,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.CyclicLR(
    #     optimizer, 0.001, 0.01, step_size_up=10, cycle_momentum=False
    # )
    loss_fn = MSELoss()

    bests = []

    logging.info("Training started.")

    for epoch in range(1, args.epochs + 1):
        stats = statistics.Statistics(epoch, args.checkpoint_dir)

        model.train()
        stats.train = one_epoch(model, device, train_dataloader, loss_fn, optimizer)

        # scheduler.step()

        model.eval()
        with torch.no_grad():
            stats.val = one_epoch(model, device, val_dataloader, loss_fn)

        bests = stats.end_of_epoch(bests, model, optimizer, args.checkpoint_dir)


def one_epoch(model, device, loader, loss_fn_, optimizer=None):
    stats = statistics.Stats()

    for images, trues, _ in tqdm(loader):
        images = images.to(device)
        trues = trues.to(device)

        preds = model(images)

        loss = loss_fn_(preds, trues)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        stats.running_totals(loss.item(), preds.detach().cpu(), trues.detach().cpu())

    return stats


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
        "--checkpoint-dir",
        type=Path,
        required=True,
        metavar="PATH",
        help="""Save best training checkpoints in this dir.""",
    )

    # arg_parser.add_argument(
    #     "--finetune",
    #     type=Path,
    #     required=True,
    #     default="google/vit-base-patch16-224",
    #     metavar="PATH",
    #     help="""Finetune this model.""",
    # )

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
