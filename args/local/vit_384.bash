#!/bin/bash

python ./phenobase/model_train_hf.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/vit_384_base_reg_f1 \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --epochs 50 \
  --batch-size 48 \
  --problem-type regression \
  --best-metric f1 \
  --trait flowers
