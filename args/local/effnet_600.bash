#!/bin/bash

python ./phenobase/model_train.py \
    --dataset-csv ./datasets/splits_2025-04-22.csv \
    --image-dir ./data/images \
    --output-dir ./data/models/effnet_600_flowers_f1_loc_sl_a \
    --finetune "google/efficientnet-b7" \
    --image-size 600 \
    --batch-size 4 \
    --best-metric f1 \
    --epochs 50 \
    --trait flowers
