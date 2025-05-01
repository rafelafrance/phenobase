#!/bin/bash

python ./phenobase/model_train.py \
    --dataset-csv ./datasets/splits_2025-04-22.csv \
    --image-dir ./data/images \
    --output-dir ./data/models/flowers_2025-05-02/effnet_528_flowers_f1_loc_unk_51 \
    --finetune "google/efficientnet-b6" \
    --image-size 528 \
    --batch-size 8 \
    --best-metric f1 \
    --epochs 50 \
    --use-unknowns \
    --problem-type regression \
    --trait flowers
