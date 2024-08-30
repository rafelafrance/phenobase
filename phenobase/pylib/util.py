import logging
import warnings
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

TRAITS = "flowers fruits leaves".split()
SPLIT = Literal["train", "eval", "test"]

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

IMAGE_EXCEPTIONS = (
    UnidentifiedImageError,
    ValueError,
    TypeError,
    FileNotFoundError,
    OSError,
)


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
