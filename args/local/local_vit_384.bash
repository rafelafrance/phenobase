#!/bin/bash

python ./phenobase/model_train.py \
  --trait-csv ./splits/splits.csv \
  --image-dir ./data/images/images_600 \
  --output-dir ./data/local/vit_384_base_prec \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 25
