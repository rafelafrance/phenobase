#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 528 \
    --regression \
    --model-dir data/models/flowers/effnet_528_flowers_f1_reg

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 528 \
    --model-dir data/models/flowers/effnet_528_flowers_f1_sl

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 528 \
    --regression \
    --model-dir data/models/flowers/effnet_528_flowers_reg_loc_f1_hf

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 600 \
    --model-dir data/models/flowers/effnet_600_flowers_f1_loc_sl

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 384 \
    --model-dir data/models/flowers/vit_384_base_flowers_f1_sl

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 384 \
    --regression \
    --model-dir data/models/flowers/vit_384_base_flowers_reg_loc_f1

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 384 \
    --model-dir data/models/flowers/vit_384_lg_flowers_f1_loc_sl

./phenobase/model_score.py \
    --dataset-csv datasets/all_traits.csv \
    --image-dir datasets/images \
    --score-csv data/score_2025_04_16a.csv \
    --trait flowers \
    --image-size 384 \
    --model-dir data/models/flowers/vit_384_lg_flowers_f1_slurm_sl
