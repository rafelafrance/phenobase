#!/usr/bin/env python

from transformers import ViTMAEConfig, ViTMAEModel

config = ViTMAEConfig()

model = ViTMAEModel(config)

config = model.config

print(model)
print(config)
