#!/bin/bash

python3 phenobase/model_infer.py \
    --db data/gbif_2024-10-28.sqlite \
    --image-dir data/images \
    --bad-taxa datasets/remove_flowers.csv \
    --output-csv data/inference/flower_inference_1_debug.csv \
    --checkpoint data/models/ensembles/best_3combo_fract/effnet_528_flowers_reg_f1_a_checkpoint-17424 \
    --image-size 528 \
    --limit 100 \
    --offset 0 \
    --problem-type regression \
    --trait flowers \
    --debug
