#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
  --dataset-csv datasets/train_data.csv \
  --image-dir datasets/images \
  --score-csv data/score_2025_04_05a.csv \
  --trait flowers \
  --image-size 528 \
  --problem-type regression \
  --model-dir data/models/effnet_528_flowers_reg_f1

#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_04_05a.csv \
#  --trait flowers \
#  --image-size 528 \
#  --problem-type single_label_classification \
#  --model-dir data/models/effnet_528_flowers_sl_f1
#
#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_04_05a.csv \
#  --trait flowers \
#  --image-size 528 \
#  --problem-type regression \
#  --model-dir data/models/effnet_528_flowers_unk_f1
#
#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_04_05a.csv \
#  --trait flowers \
#  --image-size 384 \
#  --problem-type regression \
#  --model-dir data/models/vit_384_lg_flowers_f1
#
#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_04_05a.csv \
#  --trait flowers \
#  --image-size 384 \
#  --problem-type single_label_classification \
#  --model-dir data/models/vit_384_lg_flowers_sl_f1
#
#./phenobase/model_score.py \
#  --dataset-csv datasets/test_data.csv \
#  --image-dir datasets/images \
#  --score-csv data/score_2025_04_05a.csv \
#  --trait flowers \
#  --image-size 384 \
#  --problem-type regression \
#  --model-dir data/models/vit_384_lg_flowers_unk_f1
