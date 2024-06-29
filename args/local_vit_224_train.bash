#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./data/split.csv \
  --image-dir ./data/images/images_224 \
  --output-dir ./data/training_output \
  --pretrained-dir ./data/pretraining_output \
  --trait flowers \
  --trait fruits \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 100
