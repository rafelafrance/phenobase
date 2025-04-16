#!/bin/bash

python ./phenobase/model_train.py \
    --dataset-csv ./datasets/all_traits.csv \
    --image-dir ./datasets/images \
    --output-dir ./data/models/effnet_528_fruits_f1_loc_sl \
    --finetune "google/efficientnet-b6" \
    --image-size 528 \
    --batch-size 8 \
    --best-metric f1 \
    --epochs 50 \
    --trait fruits
