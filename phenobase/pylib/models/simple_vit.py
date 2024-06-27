import logging
from pathlib import Path

import torch
from transformers import ViTForImageClassification

from phenobase.pylib import util


def load_model_state(model, load_model):
    """Load a previous model."""
    model.state = torch.load(load_model) if load_model else {}
    if model.state.get("model_state"):
        logging.info("Loading a model.")
        model.load_state_dict(model.state["model_state"])


class SimpleVit(torch.nn.Module):
    def __init__(
        self,
        traits=list[util.TRAIT],
        pretrained_dir: Path | None = None,
        load_model: Path | None = None,
        model_config: Path | None = None,
    ):
        super().__init__()

        self.traits = traits

        if pretrained_dir:
            self.vit = ViTForImageClassification.from_pretrained(str(pretrained_dir))
            self.vit.classifier.out_features = len(traits)
        else:
            self.vit = ViTForImageClassification(model_config)

        load_model_state(self, load_model)

    def forward(self, x):
        x = self.vit(x)
        return x
