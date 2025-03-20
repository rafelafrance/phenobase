#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/output/effnet_528_flowers_f1 \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --epochs 10 \
  --batch-size 8 \
  --trait flowers
