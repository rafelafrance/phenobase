#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/train_data.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/vit_384_base_unk_f1 \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --epochs 25 \
  --batch-size 48 \
  --best-metric f1 \
  --use-unknowns \
  --trait flowers
