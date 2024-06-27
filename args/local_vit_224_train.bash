#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./data/splits.csv \
  --image-dir ./data/images/images_224 \
  --save-model ./data/training_output \
  --pretrained-dir ./data/pretraining_output \
  --log-dir ./data/logs \
  --trait flowers \
  --trait fruits
