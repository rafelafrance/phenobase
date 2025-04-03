#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/train_data.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/vit_384_base_sl_f1 \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --epochs 50 \
  --batch-size 48 \
  --problem-type single_label_classification \
  --best-metric f1 \
  --trait flowers
