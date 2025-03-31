#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
  --dataset-csv datasets/test_data.csv \
  --image-dir datasets/images \
  --score-csv data/score_2025_03_30a.csv \
  --trait flowers \
  --image-size 528 \
  --model-dir data/models/effnet_528_flowers

#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_03_30a.csv \
#  --trait flowers \
#  --image-size 528 \
#  --model-dir data/models/effnet_528_unk_flowers
#
#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_03_30a.csv \
#  --trait flowers \
#  --image-size 384 \
#  --model-dir data/models/vit_384_lg_flowers
#
#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_03_30a.csv \
#  --trait flowers \
#  --image-size 384 \
#  --model-dir data/models/vit_384_lg_unk_flowers
