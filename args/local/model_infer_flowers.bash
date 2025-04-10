#!/bin/bash

python3 phenobase/model_infer.py \
    --db data/gbif_2024-10-28.sqlite \
    --image-dir datasets/images \
    --bad-families datasets/bad_families/bad_flower_fams.csv \
    --output-csv data/flower_inference_0_10000.csv \
    --checkpoint data/models/effnet_528_flowers_f1_sl/checkpoint-15260 \
    --image-size 528 \
    --limit 10000 \
    --offset 0 \
    --trait flowers \
    --thresh-low 0.05 \
    --thresh-high 0.95 \
    --debug
