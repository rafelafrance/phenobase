#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/output/vit_384_base_f1 \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --epochs 10 \
  --batch-size 48 \
  --trait flowers
