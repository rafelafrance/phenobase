#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./splits/split_all_3.csv \
  --image-dir ./data/images/images_224 \
  --output-dir ./data/local/vit_224_base_prec \
  --finetune "google/vit-base-patch16-224" \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 25
