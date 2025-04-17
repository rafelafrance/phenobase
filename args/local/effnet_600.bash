#!/bin/bash

python ./phenobase/model_train.py \
    --dataset-csv ./datasets/all_traits.csv \
    --image-dir ./datasets/images \
    --output-dir ./data/models/effnet_600_flowers_f1_loc_ml \
    --finetune "google/efficientnet-b7" \
    --image-size 600 \
    --batch-size 4 \
    --best-metric f1 \
    --epochs 50 \
    --problem-type multi_label_classification \
    --trait flowers
