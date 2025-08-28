#!/bin/bash

python3 phenobase/model_infer_custom.py \
    --custom-data data/inference/not_inferred_debug.csv \
    --image-dir data \
    --bad-taxa datasets/remove_flowers.csv \
    --output-csv data/inference/flower_inference_1_debug.csv \
    --checkpoint data/models/ensembles/best_3combo_fract/effnet_528_flowers_reg_f1_a_checkpoint-17424 \
    --image-size 528 \
    --problem-type regression \
    --trait flowers
