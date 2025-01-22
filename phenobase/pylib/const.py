from typing import Literal

TRAITS = "flowers fruits leaves buds".split()

SPLIT = Literal["train", "valid", "test"]
SPLITS = ["train", "valid", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

LABELS = ["without", "with"]
ID2LABEL = {str(i): v for i, v in enumerate(LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
WITHOUT = [1, 0]
WITH = [0, 1]

REGRESSION = "regression"
SINGLE_LABEL = "single_label"
MULTI_LABEL = "multi_label"
PROBLEM_TYPES = {
    REGRESSION: REGRESSION,
    SINGLE_LABEL: "single_label_classification",
    MULTI_LABEL: "multi_label_classification",
}
