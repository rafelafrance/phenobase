#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./data/split_all_3.csv \
  --image-dir ./data/images/images_600 \
  --output-dir ./data/local/effnet_456_flowers_pos \
  --finetune "google/efficientnet-b5" \
  --image-size 456 \
  --batch-size 12 \
  --lr 1e-4 \
  --use-weights \
  --traits flowers \
  --epochs 25
