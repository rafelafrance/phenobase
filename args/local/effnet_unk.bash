#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/train_data.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/effnet_528_flowers_unk_f1 \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --epochs 50 \
  --batch-size 8 \
  --best-metric f1 \
  --problem-type regression \
  --use-unknowns \
  --trait flowers
