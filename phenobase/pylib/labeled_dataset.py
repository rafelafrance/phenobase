import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from phenobase.pylib import util
from phenobase.pylib.util import TRAITS


@dataclass
class LabeledSheet:
    path: Path
    target: torch.tensor


class LabeledDataset(Dataset):
    def __init__(
        self,
        *,
        trait_csv: Path,
        image_dir: Path,
        split: util.SPLIT,
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.transform = self.build_transforms(image_size, augment=augment)
        with trait_csv.open() as csv_in:
            reader = csv.DictReader(csv_in)
            sheets = [
                s
                for s in reader
                if s["split"] == split and all(s[t] in "01" for t in TRAITS)
            ]
            self.sheets = [
                LabeledSheet(
                    path=image_dir / s["file"],
                    target=torch.tensor([int(s[t]) for t in TRAITS], dtype=torch.float),
                )
                for s in sheets
            ]

    def __len__(self) -> int:
        return len(self.sheets)

    def __getitem__(self, index) -> dict:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(sheet.path).convert("RGB")
            image = self.transform(image)
        return {"pixel_values": image, "labels": sheet.target, "name": sheet.path.name}

    @staticmethod
    def build_transforms(image_size, *, augment=False):
        xform = [transforms.Resize(image_size)]

        if augment:
            xform += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.AutoAugment(),
            ]

        xform += [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(util.IMAGENET_MEAN, util.IMAGENET_STD_DEV),
        ]

        return transforms.Compose(xform)

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the traits."""
        weights = []
        for i in range(len(TRAITS)):
            pos = sum(s.target[i] for s in self.sheets)
            neg = len(self) - pos
            pos_wt = neg / pos if pos > 0 else 1.0
            weights.append(pos_wt)
        return weights
