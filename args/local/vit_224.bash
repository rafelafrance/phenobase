#!/bin/bash

python ./phenobase/model_train.py \
  --trait-csv ./splits/splits.csv \
  --image-dir ./data/images/images_600 \
  --output-dir ./data/local/vit_224_base_prec \
  --finetune "google/vit-base-patch16-224" \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 25 \
  --trait flowers
