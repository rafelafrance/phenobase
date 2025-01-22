import torch
from torchvision.transforms import v2

from phenobase.pylib import const


def build_transforms(image_size, *, augment=False):
    xform = [v2.Resize((image_size, image_size))]

    if augment:
        xform += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.AutoAugment(),
        ]

    xform += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(const.IMAGENET_MEAN, const.IMAGENET_STD_DEV),
    ]

    return v2.Compose(xform)
