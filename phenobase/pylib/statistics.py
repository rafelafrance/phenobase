import logging
from dataclasses import dataclass, field

import torch


@dataclass()
class Stats:
    trues: list[float] = field(default_factory=list)
    preds: list[float] = field(default_factory=list)
    loss: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def running_totals(self, batch_loss, trues, preds):
        self.loss += batch_loss
        self.trues += trues
        self.preds += preds

    def calculate(self):
        trues = torch.tensor(self.trues).sigmoid().round()
        preds = torch.tensor(self.preds).sigmoid().round()

        tp = ((preds == 1.0) & (trues == 1.0)).sum().item()
        fp = ((preds == 1.0) & (trues == 0.0)).sum().item()
        fn = ((preds == 0.0) & (trues == 1.0)).sum().item()
        tn = ((preds == 0.0) & (trues == 0.0)).sum().item()

        count = tp + tn + fn + fp

        accuracy = (tp + tn) / count if count > 0.0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        self.f1 = 2.0 * (pre * recall) / (pre + recall) if (pre + recall) > 0.0 else 0.0
        self.accuracy = accuracy
        self.precision = pre
        self.recall = recall
        self.loss = self.loss / count if count else 1000.0

    def report(self, prefix, epoch):
        line = (
            f"epoch {epoch:4d}, {prefix:<5}, loss {self.loss:6.4f}, "
            f"accuracy {self.accuracy:6.4f}, f1 {self.f1:6.4f}, "
            f"precision {self.precision:6.4f}, recall {self.recall:6.4f}"
        )
        logging.info(line)


class Statistics:
    def __init__(self, epoch, checkpoint_dir):
        self.epoch: int = epoch
        self.path = checkpoint_dir / f"checkpoint_{epoch:04d}.pt"
        self.train: Stats = Stats()
        self.val: Stats = Stats()

    def end_of_epoch(
        self,
        bests,
        model,
        optimizer,
        checkpoint_dir,
        metric: str = "f1",
        sign: float = -1.0,
        keep: int = 5,
    ):
        self.train.calculate()
        self.val.calculate()

        bests = self.get_best(bests, keep, metric, sign)

        self.train.report("train", self.epoch)
        self.val.report("val", self.epoch)
        self.report_bests(bests, metric)

        self.save_checkpoint(bests, model, optimizer)
        self.clean_checkpoints(bests, checkpoint_dir)
        return bests

    @staticmethod
    def report_bests(bests, metric):
        tops = [f"epoch: {b.epoch:4d} = {getattr(b.val, metric):6.4f}" for b in bests]
        tops = ", ".join(tops)
        line = f"best {metric} [{tops}]"
        logging.info(line)

    def get_best(self, bests, keep=5, metric: str = "f1", sign: float = -1.0):
        bests.append(self)
        bests = sorted(bests, key=lambda b: sign * getattr(b.val, metric))
        bests = bests[:keep]
        return bests

    def save_checkpoint(self, bests, model, optimizer):
        if any(self.path == b.path for b in bests):
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": self.train.loss,
                    "train_accuracy": self.train.accuracy,
                    "train_precision": self.train.precision,
                    "train_recall": self.train.recall,
                    "train_f1": self.train.f1,
                    "val_loss": self.val.loss,
                    "val_accuracy": self.val.accuracy,
                    "val_precision": self.val.precision,
                    "val_recall": self.val.recall,
                    "val_f1": self.val.f1,
                },
                self.path,
            )

    @staticmethod
    def clean_checkpoints(bests, checkpoint_dir):
        for path in checkpoint_dir.glob("checkpoint_*.pt"):
            if all(path != b.path for b in bests):
                path.unlink()
