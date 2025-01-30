from typing import Literal

TRAITS = "flowers fruits leaves buds old_flowers old_fruits".split()

SPLIT = Literal["train", "valid", "test"]
SPLITS = ["train", "valid", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

LABELS = ["without", "with"]
WITHOUT = [1, 0]
WITH = [0, 1]

REGRESSION = "regression"
SINGLE_LABEL = "single_label_classification"
MULTI_LABEL = "multi_label_classification"
PROBLEM_TYPES = [REGRESSION, SINGLE_LABEL, MULTI_LABEL]
