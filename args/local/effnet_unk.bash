#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/train_data.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/effnet_528_flowers_unk_f1 \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --epochs 25 \
  --batch-size 8 \
  --best-metric f1 \
  --use-unknowns \
  --trait flowers
