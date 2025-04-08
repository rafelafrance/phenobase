from typing import Literal

TRAITS = ["flowers", "fruits", "leaves", "buds"]
SPLIT = Literal["train", "valid", "test"]
SPLITS = ["train", "valid", "test"]

IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

LABELS = ["without", "with"]
ID2LABEL = {str(i): v for i, v in enumerate(LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

REGRESSION = "regression"
SINGLE_LABEL = "single_label_classification"
MULTI_LABEL = "multi_label_classification"
PROBLEM_TYPES = [REGRESSION, SINGLE_LABEL, MULTI_LABEL]
