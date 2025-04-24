#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./data/images \
  --output-dir ./data/models/vit_384_large_fruits_f1_loc_sl \
  --finetune "google/vit-large-patch16-384" \
  --image-size 384 \
  --batch-size 8 \
  --best-metric f1 \
  --epochs 50 \
  --trait fruits
