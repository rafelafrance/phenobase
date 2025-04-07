#!/bin/bash

python ./phenobase/model_train_hf.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --output-dir ./data/local/effnet_528_flowers_reg_f1_hf \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --epochs 50 \
  --batch-size 8 \
  --problem-type regression \
  --trait flowers
