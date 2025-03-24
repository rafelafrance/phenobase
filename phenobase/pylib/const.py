from typing import Literal

TRAITS = "flowers fruits leaves buds old_flowers old_fruits".split()

SPLIT = Literal["train", "val", "test"]
SPLITS = ["train", "val", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

REGRESSION = "regression"
SINGLE_LABEL = "single_label_classification"
MULTI_LABEL = "multi_label_classification"
PROBLEM_TYPES = [REGRESSION, SINGLE_LABEL, MULTI_LABEL]
