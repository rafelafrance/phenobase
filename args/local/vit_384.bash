#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/flowers.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/vit_384_base_prec \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --best-metric f1 \
  --epochs 50 \
  --batch-size 48
