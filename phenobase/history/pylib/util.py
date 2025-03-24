import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

TRAITS = "flowers fruits leaves".split()
TARGETS = TRAITS


def accuracy(logits, targets):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(logits))
    equals = (pred == targets).type(torch.float)
    return torch.mean(equals)
