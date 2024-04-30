import logging
import warnings
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

TARGETS = """ flowers fruits leaves """.split()

IMAGE_EXCEPTIONS = (
    UnidentifiedImageError,
    ValueError,
    TypeError,
    FileNotFoundError,
    OSError,
)


def accuracy(preds, targets):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(preds))
    equals = (pred == targets).type(torch.float)
    return torch.mean(equals)


def get_sheet_image(path: Path) -> Image:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # No EXIF warnings

        try:
            image = Image.open(path).convert("RGB")

        except IMAGE_EXCEPTIONS as err:
            msg = f"Could not prepare {path.name}: {err}"
            logging.error(msg)  # noqa: TRY400
            return None

        return image


def resize_image(image: Image, size: int) -> Image:
    try:
        image = image.resize((size, size))
    except IMAGE_EXCEPTIONS:
        return None

    return image
