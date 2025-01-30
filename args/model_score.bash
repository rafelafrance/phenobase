#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
  --dataset-csv datasets/all_traits.csv \
  --image-dir datasets/images \
  --output-csv data/score_single_label_unfiltered_2025_01_30a.csv \
  --problem-type regression \
  --trait old_flowers \
  --image-size 528 \
  --model-dir data/tuned/effnet_529_flowers_f1

# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait flowers \
#  --image-size 384 \
#  --model-dir data/tuned/vit_384_lg_flowers_f1
#
# #---------------------------------------------------------------------------------------------
# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait fruits \
#  --image-size 528 \
#  --model-dir data/tuned/effnet_528_fruits_f1
#
# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait fruits \
#  --image-size 384 \
#  --model-dir data/tuned/vit_384_lg_fruits_f1
#
# #---------------------------------------------------------------------------------------------
# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait leaves \
#  --image-size 528 \
#  --model-dir data/tuned/effnet_528_leaves_f1
#
# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait leaves \
#  --image-size 384 \
#  --model-dir data/tuned/vit_384_lg_leaves_f1
#
# #---------------------------------------------------------------------------------------------
# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait buds \
#  --image-size 528 \
#  --model-dir data/tuned/effnet_528_buds_f1
#
# ./phenobase/model_score.py \
#  --dataset-csv datasets/all_traits.csv \
#  --image-dir datasets/images \
#  --output-csv data/score_single_label_2025_01_28c.csv \
#  --problem-type single_label_classification \
#  --trait buds \
#  --image-size 384 \
#  --model-dir data/tuned/vit_384_lg_buds_f1
