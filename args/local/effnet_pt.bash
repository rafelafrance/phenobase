#!/bin/bash

python ./phenobase/model_train_pt.py \
  --dataset-csv ./datasets/all_traits.csv \
  --image-dir ./datasets/images \
  --checkpoint-dir ./data/local/effnet_528_flowers_reg_f1 \
  --image-size 528 \
  --epochs 100 \
  --batch-size 8 \
  --trait flowers
