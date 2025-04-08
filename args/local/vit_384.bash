#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/output/vit_384_base_flowers_f1_hf \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --batch-size 48 \
  --best-metric f1 \
  --epochs 50 \
  --trait flowers
