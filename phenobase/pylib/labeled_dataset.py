import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from phenobase.pylib import const


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
        split: const.SPLIT,
        image_size: int,
        traits: list[str],
        augment: bool = False,
    ) -> None:
        self.transform = self.build_transforms(image_size, augment=augment)
        self.traits = traits if traits else const.TRAITS

        with trait_csv.open() as csv_in:
            reader = csv.DictReader(csv_in)
            sheets = [
                s
                for s in reader
                if s["split"] == split and all(s[t] in "01" for t in self.traits)
            ]
            self.sheets = [
                LabeledSheet(
                    path=image_dir / s["file"],
                    target=torch.tensor(
                        [int(s[t]) for t in self.traits], dtype=torch.float
                    ),
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
        xform = [v2.Resize(image_size)]

        if augment:
            xform += [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.AutoAugment(),
            ]

        xform += [
            v2.ToTensor(),
            v2.ConvertImageDtype(torch.float),
            v2.Normalize(mean=const.IMAGENET_MEAN, std=const.IMAGENET_STD_DEV),
        ]

        return v2.Compose(xform)

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the traits."""
        weights = []
        for i in range(len(self.traits)):
            pos = sum(s.target[i] for s in self.sheets)
            neg = len(self) - pos
            pos_wt = neg / pos if pos > 0 else 1.0
            weights.append(pos_wt)
        return torch.tensor(weights, dtype=torch.float)
