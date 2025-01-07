from typing import Literal

TRAITS = "flowers fruits leaves buds".split()
SPLIT = Literal["train", "valid", "test"]
SPLITS = ["train", "valid", "test"]

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)
