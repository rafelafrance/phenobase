#!/bin/bash

python ./phenobase/train_vit.py \
  --trait-csv ./data/split_all_3.csv \
  --image-dir ./data/images/images_224 \
  --output-dir ./data/local/flowers_no_mae \
  --trait flowers \
  --batch-size 48 \
  --lr 1e-4 \
  --epochs 100
