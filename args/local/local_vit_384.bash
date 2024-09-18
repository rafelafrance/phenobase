#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./splits/split_all_3.csv \
  --image-dir ./data/images/images_384 \
  --output-dir ./data/local/vit_384_base_prec \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 25
