import warnings
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from phenobase.pylib import util

Split = Literal["train", "val", "test"]


class MaskedImages(Dataset):
    def __init__(
        self,
        csv_data,
        image_dir: Path,
        split: Split,
        image_size: int,
        *,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = (image_size, image_size)
        self.transform = self.build_transforms(augment=augment)
        self.sheets: list[Path] = [s["path"] for s in csv_data if s["split"] == split]

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index) -> tuple:
        sheet: Path = self.sheets[index]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            image = Image.open(sheet.image_path).convert("RGB")
            image = self.transform(image)

        return image, sheet.targets

    def build_transforms(self, *, augment=False):
        xform = [transforms.Resize(self.image_size)]

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

    @staticmethod
    def pos_weight(dataset):
        weights = []
        for i in range(len(util.TARGETS)):
            pos = sum(s.targets[i] for s in dataset.sheets)
            pos_wt = (len(dataset) - pos) / pos if pos > 0.0 else 1.0
            weights.append(pos_wt)
        return weights
