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
        *,
        traits=list[util.TRAIT],
        load_model: Path | None = None,
        pretrained_dir: Path | None = None,
    ):
        super().__init__()

        self.traits = traits

        if pretrained_dir:
            self.vit = ViTForImageClassification.from_pretrained(
                str(pretrained_dir), num_labels=len(traits)
            )
        else:
            load_model_state(self, load_model)

    def forward(self, x):
        x = self.vit(x)
        return x
