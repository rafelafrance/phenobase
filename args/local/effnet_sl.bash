#!/bin/bash

python ./phenobase/model_train.py \
  --dataset-csv ./datasets/train_data.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/effnet_528_flowers_reg_f1 \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --epochs 5 \
  --batch-size 8 \
  --problem-type single_label_classification \
  --best-metric f1 \
  --trait flowers
