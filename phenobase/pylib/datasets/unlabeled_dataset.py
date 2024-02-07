import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .. import util


@dataclass
class UnlabeledSheet:
    image_path: Path


class UnlabeledDataset:
    def __init__(self, csv_data, image_dir: Path, image_size: int) -> None:
        super().__init__()
        self.image_size = (image_size, image_size)
        self.transform = self.build_transforms()
        self.sheets: list[UnlabeledSheet] = [
            UnlabeledSheet(image_path=image_dir / s["file"]) for s in csv_data
        ]

    def __len__(self):
        return len(self.sheets)

    def __getitem__(self, index):
        sheet: UnlabeledSheet = self.sheets[index]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings
            image = Image.open(sheet.image_path).convert("RGB")
            image = self.transform(image)

        return image

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
