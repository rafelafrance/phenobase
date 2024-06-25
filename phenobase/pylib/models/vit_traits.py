import torch
from transformers import ViTForImageClassification


class VitTraits(torch.nn.Module):
    def __init__(self, pretrained="data/pretraining_output", out_features=5):
        self.vit = ViTForImageClassification.from_pretrained(pretrained)
        self.vit.classifier.out_features = out_features
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)
        x = self.sigmoid(x)
        return x
