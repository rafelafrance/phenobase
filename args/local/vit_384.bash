#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/vit_384_base_prec \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --epochs 25 \
  --batch-size 48 \
  --use-unknowns \
  --trait flowers
