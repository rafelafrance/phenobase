#!/bin/bash

python ./phenobase/train_model.py \
  --trait-csv ./splits/splits.csv \
  --image-dir ./data/images/images_600 \
  --output-dir ./data/local/effnet_456_flowers_prec_wt \
  --finetune "google/efficientnet-b5" \
  --image-size 456 \
  --batch-size 12 \
  --lr 1e-4 \
  --traits flowers \
  --use-weights \
  --best-metric precision \
  --epochs 50
