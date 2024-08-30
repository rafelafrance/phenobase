#!/bin/bash

python ./phenobase/train_vit.py \
  --trait-csv ./data/split_all_3.csv \
  --image-dir ./data/images/images_224 \
  --output-dir ./data/local/vit_224 \
  --finetune "google/vit-large-patch16-224" \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 50
