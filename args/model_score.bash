#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
  --dataset-csv datasets/all_traits.csv \
  --image-dir datasets/images \
  --score-csv data/score_2025_04_10a.csv \
  --trait flowers \
  --image-size 528 \
  --regression \
  --model-dir data/models/effnet_528_flowers_f1_reg

./phenobase/model_score.py \
  --dataset-csv datasets/all_traits.csv \
  --image-dir datasets/images \
  --score-csv data/score_2025_04_10a.csv \
  --trait flowers \
  --image-size 528 \
  --model-dir data/models/effnet_528_flowers_f1_sl

./phenobase/model_score.py \
  --dataset-csv datasets/all_traits.csv \
  --image-dir datasets/images \
  --score-csv data/score_2025_04_10a.csv \
  --trait flowers \
  --image-size 384 \
  --model-dir data/models/vit_384_base_flowers_f1_sl

./phenobase/model_score.py \
  --dataset-csv datasets/all_traits.csv \
  --image-dir datasets/images \
  --score-csv data/score_2025_04_10a.csv \
  --trait flowers \
  --image-size 384 \
  --model-dir data/models/vit_384_lg_flowers_f1_sl
