#!/bin/bash

python ./phenobase/train_vit.py \
  --trait-csv ./data/split_all_3.csv \
  --image-dir ./data/images/images_384 \
  --output-dir ./data/local/vit_384_hf \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 50
