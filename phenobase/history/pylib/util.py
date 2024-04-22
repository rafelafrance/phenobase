import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD_DEV = (0.229, 0.224, 0.225)

TARGETS = """ flowers fruits leaves """.split()


def accuracy(preds, targets):
    """Calculate the accuracy of the model."""
    pred = torch.round(torch.sigmoid(preds))
    equals = (pred == targets).type(torch.float)
    return torch.mean(equals)
