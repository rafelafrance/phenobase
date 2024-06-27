import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from phenobase.pylib import util


@dataclass
class LabeledSheet:
    path: Path
    target: list[int]


class LabeledTraits(Dataset):
    def __init__(
        self,
        *,
        trait_csv: Path,
        image_dir: Path,
        traits: list[util.TRAIT],
        split: util.SPLIT,
        augment: bool = False,
    ) -> None:
        self.traits = traits
        self.transform = self.build_transforms(augment=augment)
        with trait_csv.open() as csv_in:
            reader = csv.DictReader(csv_in)
            self.sheets = [
                LabeledSheet(
                    path=image_dir / s["file"],
                    target=[int(s[t]) for t in traits],
                )
                for s in reader
                if s["split"] == split
            ]

    def __len__(self) -> int:
        return len(self.sheets)

    def __getitem__(self, index) -> tuple:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            sheet = self.sheets[index]
            image = Image.open(sheet.path).convert("RGB")
            image = self.transform(image)
        return image, sheet.target, sheet.path.name

    @staticmethod
    def build_transforms(*, augment=False):
        xform = [transforms.Resize(util.IMAGE_SIZE)]

        if augment:
            xform += [
                transforms.AutoAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        xform += [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(util.IMAGENET_MEAN, util.IMAGENET_STD_DEV),
        ]

        return transforms.Compose(xform)

    def pos_weight(self):
        """Calculate the weights for the positive & negative cases of the trait."""
        weights = []
        for i in range(len(self.traits)):
            pos = sum(s.target[i] for s in self.sheets)
            neg = len(self) - pos
            pos_wt = neg / pos if pos > 0 else 1.0
            weights.append(pos_wt)
        return weights
