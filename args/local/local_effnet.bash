#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/flowers.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/output/effnet_528_flowers_f1 \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --best-metric f1 \
  --epochs 50 \
  --batch-size 8
