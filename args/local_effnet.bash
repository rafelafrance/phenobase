#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./data/split_all_3.csv \
  --image-dir ./data/images/images_600 \
  --output-dir ./data/local/effnet_456_f1 \
  --finetune "google/efficientnet-b5" \
  --image-size 456 \
  --batch-size 12 \
  --lr 1e-4 \
  --epochs 50
